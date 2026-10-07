"""
Sequencing-depth handling and the matched random-gene-set sampler, shared by the
Moran's I null test, co-localization and the gene-pair test.

  DepthModel              QC-column access; nuisance design matrix [1, log1p(total counts), ...];
                          OLS residuals through one QR decomposition
  MatchedGeneSetSampler   random gene sets with the same size and expression-bin composition
                          as a marker set
  moran_i, row_normalize  fast global Moran's I (row-standardized weights, as squidpy)
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


# ---------------------------------------------------------------------------
# Moran's I
# ---------------------------------------------------------------------------
def row_normalize(W) -> sp.csr_matrix:
    """Row-standardize a sparse weight matrix (each row sums to 1; isolated bins stay 0) - squidpy's default."""
    W = sp.csr_matrix(W, dtype=float)
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    inv = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
    return sp.diags(inv) @ W


def moran_i(score, W, row_normalized: bool = False) -> float:
    """
    Global Moran's I of ``score`` for weights ``W``. W is row-normalized first unless
    ``row_normalized`` (the null loops normalize once, for speed), so the value equals the I
    in {celltype}_moran_stats.csv. Implemented directly because the null tests need
    thousands of evaluations.
    """
    if not row_normalized:
        W = row_normalize(W)
    z = np.asarray(score, dtype=float)
    z = z - z.mean()
    n = z.shape[0]
    s0 = W.sum()
    if s0 == 0:
        raise ValueError("Spatial weight matrix has zero total weight - check the spatial_neighbors graph.")
    denom = (z ** 2).sum()
    if denom == 0:
        return 0.0
    numerator = z @ (W @ z)
    return float((n / s0) * (numerator / denom))


# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------
class DepthModel:
    """
    Nuisance model for sequencing depth: an intercept, ``covariates`` and (optionally) one-hot
    dummies of ``compartment_col``. Residuals of y ~ X are what depth cannot explain.

    covariates : any of "log_total_counts" (log1p total counts, the main library-size covariate)
        and "log_n_genes" (detection efficiency).
    compartment_col : obs column (e.g. "tissue_section", Leiden clusters); its dummies remove
        between-compartment mean differences, so remaining structure must exist WITHIN compartments.
    """

    QC_HINT = ("Run scanpy's QC on the RAW counts before the pipeline, e.g. "
               "sc.pp.calculate_qc_metrics(adata, layer='counts', inplace=True) "
               "(adata.X holds log-normalized data, which would give wrong totals).")
    COVARIATES = ("log_total_counts", "log_n_genes")

    def __init__(self, covariates: Iterable[str] = ("log_total_counts",), compartment_col: str | None = None):
        self.covariates = tuple(covariates)
        bad = [c for c in self.covariates if c not in self.COVARIATES]
        if bad:
            raise ValueError(f"Unknown covariate(s) {bad}; use {self.COVARIATES}.")
        self.compartment_col = compartment_col

    # --- QC columns (the pipeline never computes depth itself) ---
    @classmethod
    def require_qc_metrics(cls, adata: AnnData, cols=("total_counts", "n_genes_by_counts")) -> None:
        missing = [c for c in cols if c not in adata.obs.columns]
        if missing:
            raise KeyError(f"adata.obs is missing {missing}. {cls.QC_HINT}")

    @classmethod
    def total_counts(cls, adata: AnnData) -> np.ndarray:
        cls.require_qc_metrics(adata, ("total_counts",))
        return np.asarray(adata.obs["total_counts"].values, dtype=float)

    @classmethod
    def n_genes(cls, adata: AnnData) -> np.ndarray:
        cls.require_qc_metrics(adata, ("n_genes_by_counts",))
        return np.asarray(adata.obs["n_genes_by_counts"].values, dtype=float)

    @property
    def label(self) -> str:
        return "+".join(self.covariates) + (f"+{self.compartment_col}" if self.compartment_col else "")

    def design_matrix(self, adata: AnnData) -> np.ndarray:
        cols = [np.ones(adata.n_obs)]
        for c in self.covariates:
            cols.append(np.log1p(self.total_counts(adata) if c == "log_total_counts" else self.n_genes(adata)))
        X = np.column_stack(cols)
        if self.compartment_col is not None:
            dummies = pd.get_dummies(adata.obs[self.compartment_col].astype(str), drop_first=True).values.astype(float)
            if dummies.size:
                X = np.column_stack([X, dummies])
        return X

    def residualizer(self, adata: AnnData) -> Callable[[np.ndarray], np.ndarray]:
        """f(y) -> residuals of the OLS fit y ~ X (one QR decomposition, reused for every y)."""
        Q, _ = np.linalg.qr(self.design_matrix(adata))

        def resid(y):
            y = np.asarray(y, dtype=float)
            return y - Q @ (Q.T @ y)

        return resid

    def residualize(self, adata: AnnData, cols: Iterable[str], suffix: str = "_resid") -> pd.DataFrame:
        """
        Regress each obs column in ``cols`` on depth, store residuals as f"{col}{suffix}", and return
        the R^2 of each fit (share of the column's variance explained by depth).
        """
        resid = self.residualizer(adata)
        rows = []
        for col in cols:
            y = np.asarray(adata.obs[col].values, dtype=float)
            r = resid(y)
            adata.obs[f"{col}{suffix}"] = r
            tss = ((y - y.mean()) ** 2).sum()
            r2 = 1 - (r ** 2).sum() / tss if tss > 0 else np.nan
            rows.append({"column": col, "residual_column": f"{col}{suffix}", "depth_R2": float(r2),
                         "covariates": self.label})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Matched random gene sets
# ---------------------------------------------------------------------------
class MatchedGeneSetSampler:
    """
    Random gene sets matched to ``marker_present`` in size and expression-bin composition
    (genes binned by mean expression into ``n_bins`` quantiles, as Seurat's AddModuleScore),
    sampled without replacement and never containing a marker gene or any gene in
    ``exclude_genes``. Without matching, highly expressed (less noisy) genes would make the
    null too easy to beat.

    Call :meth:`draw` for the mean-expression score of one random gene set.
    """

    def __init__(self, adata: AnnData, marker_present: list[str], layer: str = "lognorm", n_bins: int = 10,
                 exclude_genes: Iterable[str] = (), seed: int = 0):
        self.rng = np.random.default_rng(seed)
        _, gene_bins = self.expression_bins(adata, layer=layer, n_bins=n_bins)
        self.marker_bin_counts = gene_bins.loc[marker_present].value_counts()
        blocked = set(marker_present) | set(exclude_genes)
        self.genes_by_bin = {b: np.array(gene_bins.index[gene_bins == b].difference(blocked).tolist())
                             for b in gene_bins.unique()}
        self.fallback_pool = np.array(sorted(set(adata.var_names) - blocked))
        matrix = adata.layers[layer]
        self.is_sparse = sp.issparse(matrix)
        self.matrix = sp.csc_matrix(matrix) if self.is_sparse else matrix     # fast column slicing
        self.gene_index = {g: i for i, g in enumerate(adata.var_names)}

    @staticmethod
    def expression_bins(adata: AnnData, layer: str = "lognorm", n_bins: int = 10):
        """(mean expression per gene, expression-bin per gene)."""
        matrix = adata.layers[layer]
        mean_expr = np.asarray(matrix.mean(axis=0)).ravel() if sp.issparse(matrix) else np.asarray(matrix).mean(axis=0)
        mean_expr = pd.Series(mean_expr, index=adata.var_names)
        bins = pd.qcut(mean_expr.rank(method="first"), q=n_bins, labels=False)
        return mean_expr, bins

    def draw(self) -> np.ndarray:
        random_genes = []
        for b, count in self.marker_bin_counts.items():
            pool = self.genes_by_bin[b]
            if len(pool) >= count:
                random_genes.extend(self.rng.choice(pool, size=count, replace=False))
            else:
                remaining = np.setdiff1d(self.fallback_pool, np.array(random_genes + list(pool)))
                random_genes.extend(list(pool) + list(self.rng.choice(remaining, size=count - len(pool), replace=False)))
        idx = [self.gene_index[g] for g in random_genes]
        if self.is_sparse:
            return np.asarray(self.matrix[:, idx].mean(axis=1)).ravel()
        return np.asarray(self.matrix[:, idx]).mean(axis=1)

    __call__ = draw
