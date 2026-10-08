"""
Optional gene-pair follow-up: "are these two SPECIFIC genes spatially co-localized, and if
so, what defines the region where they are co-expressed?" Runs on the analysis h5ad saved
by the main pipeline.

  GenePairAnalysis        runs the three analyses below for one gene pair (or small set)
    .colocalization()     bivariate Moran's I (esda Moran_BV) on depth-residualized
                          expression, both directions, permutation null
    .correlation()        Spearman + Pearson, bin by bin, no space
  CoexpressionNicheDE     bins where all genes are detected -> depth-checked,
                          depth-matched DE vs. the rest of the tissue (+ top-20 dot plot)

Caveat: single genes are sparse, and the bivariate Moran's I null shuffles locations (it
asks "any spatial association at all?"). That is much weaker than the matched random-gene-set
null of the cell-type test (SignatureColocalization). Treat gene-pair results as supporting
evidence, not as the main co-localization claim.

Outputs per pair (name = "_".join(genes), e.g. GNLY_ATP8A2):
  {name}_spatial_colocalization.csv, {name}_coexpression_correlation.csv
  {name}_coexpr_depth_check.csv (before / after matching, one row per stage)
  {name}_coexpr_de_genes.csv, {name}_coexpr_de_top20.csv, {name}_coexpr_de_top20_dotplot.pdf
  spatial_{name}_coexpr_mask.pdf
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

from ._logging import get_logger
from .depth import DepthModel
from .differential import RegionDE, RegionSpec, RegionTooSmall

log = get_logger(__name__)


def genes_present(adata: AnnData, genes: Sequence[str], what: str) -> list[str]:
    """Genes of ``genes`` found in adata.var_names; at least 2 are required."""
    if len(genes) < 2:
        raise ValueError(f"Provide 2 or more genes for a {what} (got {len(genes)}).")
    present = [g for g in genes if g in adata.var_names]
    missing = sorted(set(genes) - set(present))
    if missing:
        log.warning(f"{missing} not found in adata.var_names and will be skipped.")
    if len(present) < 2:
        raise ValueError(f"Need >=2 genes present in adata.var_names; only found {present}.")
    return present


def default_name(present: Sequence[str]) -> str:
    return "_".join(present) if len(present) <= 4 else f"{len(present)}_gene"


def fdr(results_df: pd.DataFrame, p_col: str, alpha: float, q_col: str, sig_col: str) -> None:
    """BH-FDR across the rows of results_df (a single test is left as is)."""
    if len(results_df) > 1:
        reject, qval, _, _ = multipletests(results_df[p_col].values, alpha=alpha, method="fdr_bh")
        results_df[q_col] = qval
        results_df[sig_col] = reject
    else:
        results_df[q_col] = results_df[p_col]
        results_df[sig_col] = results_df[p_col] < alpha


class CoexpressionNicheDE(RegionDE):
    """
    Co-expression niche: bins where at least ``min_genes`` (default ALL - an AND-gate) of
    ``genes`` are detected above ``expr_threshold`` -> f"{name}_niche"; the rest is
    "Rest_of_Tissue". ``min_frac`` is the alternative to min_genes. Describes the gene
    combination's hotspot transcriptionally; does NOT test whether co-expression is spatially
    meaningful (see :meth:`GenePairAnalysis.colocalization`).
    """

    BACKGROUND = "Rest_of_Tissue"

    def __init__(self, outdir, genes: Sequence[str], niche_name: str | None = None, expr_threshold: float = 0.0,
                 min_genes: int | None = None, min_frac: float | None = None, **kwargs):
        super().__init__(outdir, **kwargs)
        self.genes = list(genes)
        self.niche_name = niche_name
        self.expr_threshold = expr_threshold
        self.min_genes = min_genes
        self.min_frac = min_frac

    def define_region(self, adata):
        present = genes_present(adata, self.genes, "co-expression niche")
        name = self.niche_name or default_name(present)
        n_genes = len(present)
        k = self._min_active(n_genes, self.min_genes, self.min_frac, default=n_genes)
        niche_label, mask_col = f"{name}_niche", f"{name}_coexpr_mask"
        is_niche = self._n_active(adata, present, self.expr_threshold) >= k
        adata.obs[mask_col] = pd.Categorical(np.where(is_niche, niche_label, self.BACKGROUND))
        n_niche = int(is_niche.sum())
        log.debug(f"--- Co-expression niche: >= {k}/{n_genes} of [{', '.join(present)}] "
                 f"active (> {self.expr_threshold} in {self.layer!r}) ---")
        log.debug(f"Isolated {n_niche} / {adata.n_obs} bins as {niche_label}.")
        if n_niche == 0:
            raise RegionTooSmall(f"no bins have >= {k}/{n_genes} of {present} above {self.expr_threshold} in layer "
                             f"{self.layer!r} - try a lower expr_threshold, a smaller min_genes/min_frac, "
                             "or a different gene combination.")
        return RegionSpec(mask_col, niche_label, self.BACKGROUND, f"{name}_coexpr", f"{name}_coexpr_degs",
                          palette={niche_label: self.REGION_COLOR, self.BACKGROUND: self.BACKGROUND_COLOR},
                          title=f"Co-expression niche: {', '.join(present)}", save=f"_{name}_coexpr_mask")


class GenePairAnalysis:
    """
    The gene-pair analyses for one output folder.

    permutations : conditional permutations of Moran_BV (p_sim).
    residualize, covariates, compartment_col : depth handling of the bivariate test (DepthModel).
    weight_key : the SPATIAL graph in obsp (never obsp['connectivities'], the expression k-NN graph).
    """

    def __init__(self, outdir: str | Path, layer: str = "lognorm", weight_key: str = "spatial_connectivities",
                 permutations: int = 999, seed: int = 0, alpha: float = 0.05, residualize: bool = True,
                 covariates: Sequence[str] = ("log_total_counts",), compartment_col: str | None = None,
                 plotter=None, plot: bool = True):
        self.outdir = Path(outdir)
        self.layer = layer
        self.weight_key = weight_key
        self.permutations = permutations
        self.seed = seed
        self.alpha = alpha
        self.residualize = residualize
        self.depth = DepthModel(covariates, compartment_col)
        self.plotter = plotter
        self.plot = plot

    def _vector(self, adata: AnnData, g: str) -> np.ndarray:
        vec = adata[:, g].layers[self.layer]
        return np.asarray(vec.todense()).ravel() if sp.issparse(vec) else np.asarray(vec).ravel()

    def _weights(self, adata: AnnData):
        """libpysal W with an entry for EVERY bin (W.from_sparse drops isolated bins)."""
        import libpysal
        if self.weight_key not in adata.obsp:
            raise ValueError(f"{self.weight_key!r} not found in adata.obsp - build the spatial graph first "
                             "(SpatialGrid.build_graph). Do not substitute obsp['connectivities']: that is the "
                             "transcriptomic k-NN graph, not a spatial graph.")
        A = sp.csr_matrix(adata.obsp[self.weight_key])
        n_isolated = int((np.diff(A.indptr) == 0).sum())
        if n_isolated:
            log.debug(f"Note: {n_isolated} bins have no spatial neighbors; kept with an empty neighbor list.")
        neighbors = {i: A.indices[A.indptr[i]:A.indptr[i + 1]].tolist() for i in range(A.shape[0])}
        weights = {i: A.data[A.indptr[i]:A.indptr[i + 1]].astype(float).tolist() for i in range(A.shape[0])}
        w = libpysal.weights.W(neighbors, weights, silence_warnings=True)
        if w.n != adata.n_obs:
            raise ValueError(f"Spatial weights have {w.n} bins but adata has {adata.n_obs}.")
        return w

    # ------------------------------------------------------------------ 1. bivariate Moran's I
    def colocalization(self, adata: AnnData, genes: Sequence[str], niche_name: str | None = None) -> pd.DataFrame:
        """
        Bivariate Moran's I for every pair in ``genes``, both directions (I_xy != I_yx), on
        depth-residualized expression; BH-FDR across all directions tested.
        -> {name}_spatial_colocalization.csv
        """
        from esda.moran import Moran_BV
        present = genes_present(adata, genes, "co-localization test")
        name = niche_name or default_name(present)
        w = self._weights(adata)
        expr = {g: self._vector(adata, g) for g in present}
        if self.residualize:
            resid = self.depth.residualizer(adata)
            expr = {g: resid(v) for g, v in expr.items()}

        log.debug(f"--- Pairwise spatial co-localization (bivariate Moran's I): {', '.join(present)} ---")
        np.random.seed(self.seed)
        rows = []
        for g1, g2 in itertools.combinations(present, 2):
            for ref, lag in [(g1, g2), (g2, g1)]:
                try:
                    mbv = Moran_BV(expr[ref], expr[lag], w, permutations=self.permutations, seed=self.seed)
                except TypeError:     # esda versions without a seed argument; np.random.seed keeps it reproducible
                    mbv = Moran_BV(expr[ref], expr[lag], w, permutations=self.permutations)
                rows.append({"reference_gene": ref, "lag_gene": lag, "bv_moran_I": float(mbv.I),
                             "p_sim": float(mbv.p_sim), "permutations": self.permutations,
                             "depth_residualized": bool(self.residualize)})
        df = pd.DataFrame(rows)
        fdr(df, "p_sim", self.alpha, "fdr_qvalue", "significant_at_fdr")
        df = df.sort_values("p_sim").reset_index(drop=True)
        path = self.outdir / f"{name}_spatial_colocalization.csv"
        df.to_csv(path, index=False)
        log.debug(f"Wrote pairwise bivariate Moran's I ({len(df)} reference/lag direction(s)) to {path}\n"
                 + df.to_string(index=False))
        return df

    # ------------------------------------------------------------------ 2. co-expression niche DE
    def coexpression_de(self, adata: AnnData, genes: Sequence[str], **kwargs):
        """Depth-matched DE of the co-expression niche (see :class:`CoexpressionNicheDE`)."""
        kwargs.setdefault("plot", self.plot)
        kwargs.setdefault("layer", self.layer)
        kwargs.setdefault("seed", self.seed)
        if self.plotter is not None:
            kwargs.setdefault("plotter", self.plotter)
        return CoexpressionNicheDE(self.outdir, genes, **kwargs).run(adata)

    # ------------------------------------------------------------------ 3. correlation
    def correlation(self, adata: AnnData, genes: Sequence[str], niche_name: str | None = None) -> pd.DataFrame:
        """
        Spearman (primary) and Pearson, bin by bin across all bins; not depth-corrected, and with
        ~10^5 bins tiny correlations are "significant" - read by magnitude.
        -> {name}_coexpression_correlation.csv
        """
        present = genes_present(adata, genes, "co-expression correlation test")
        name = niche_name or default_name(present)
        expr = {g: self._vector(adata, g) for g in present}
        log.debug(f"--- Pairwise non-spatial co-expression (Spearman + Pearson): {', '.join(present)} ---")
        rows = []
        for g1, g2 in itertools.combinations(present, 2):
            rho, p_s = spearmanr(expr[g1], expr[g2])
            r, p_p = pearsonr(expr[g1], expr[g2])
            rows.append({"gene_1": g1, "gene_2": g2, "spearman_rho": float(rho), "spearman_p": float(p_s),
                         "pearson_r": float(r), "pearson_p": float(p_p), "n": int(len(expr[g1]))})
        df = pd.DataFrame(rows)
        for prefix in ("spearman", "pearson"):
            fdr(df, f"{prefix}_p", self.alpha, f"{prefix}_fdr_qvalue", f"{prefix}_significant_at_fdr")
        df = df.sort_values("spearman_p").reset_index(drop=True)
        path = self.outdir / f"{name}_coexpression_correlation.csv"
        df.to_csv(path, index=False)
        log.debug(f"Wrote Spearman + Pearson co-expression correlation ({len(df)} pair(s)) to {path}\n"
                 + df.to_string(index=False))
        return df

    # ------------------------------------------------------------------ all three
    def run(self, adata: AnnData, genes: Sequence[str]) -> dict | None:
        """All three analyses for one pair; None if a gene is not in the data."""
        missing = [g for g in genes if g not in adata.var_names]
        if missing:
            log.warning(f"Skipping {list(genes)}: not in the data: {missing}")
            return None
        log.info(f"===== Gene pair: {' + '.join(genes)} =====")
        self.outdir.mkdir(parents=True, exist_ok=True)
        name = default_name(list(genes))
        try:
            coloc = self.colocalization(adata, genes)
            degs, top = self.coexpression_de(adata, genes)
            corr = self.correlation(adata, genes)
        except RegionTooSmall as e:               # extreme case: stop this pair, keep going with the next
            return self.stop(name, genes, str(e))
        (self.outdir / f"{name}{self.STATUS_SUFFIX}").unlink(missing_ok=True)   # from an earlier STOP
        return {"colocalization": coloc, "coexpr_de": degs, "coexpr_de_top": top, "correlation": corr}

    STATUS_SUFFIX = "_gene_pair_status.csv"

    def stop(self, name: str, genes: Sequence[str], reason: str) -> dict:
        """Mark a pair STOP: its partial outputs are removed (no half results in the report) and
        {name}_gene_pair_status.csv records why; the report shows the pair with a STOP badge."""
        for pattern in (f"{name}_spatial_colocalization.*", f"{name}_coexpression_correlation.*",
                        f"{name}_coexpr_*", f"spatial_{name}_coexpr_mask.*"):     # this pair's files only
            for f in self.outdir.glob(pattern):
                f.unlink()
        pd.DataFrame([{"pair": name, "genes": " + ".join(genes), "status": "STOP", "reason": reason}]).to_csv(
            self.outdir / f"{name}{self.STATUS_SUFFIX}", index=False)
        log.warning(f"{' + '.join(genes)}: STOP - {reason} Pair skipped; the next pair continues.")
        return {"status": "STOP", "reason": reason}
