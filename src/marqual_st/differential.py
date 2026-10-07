"""
Niche / region definition and depth-matched differential expression.

  DepthMatcher     depth check of region vs background (medians, depth AUROC) and coarsened
                   exact matching on log total counts
  RegionDE         base class: define a region -> map -> depth check / matching -> Wilcoxon DE
                   -> full + top-N tables -> dot plot of the top up-regulated genes
    NicheDE        strategy 1, high-confidence niche: top quantile of the smoothed signature score
    MarkerPanelDE  strategy 2, marker-panel enriched region: >= min_genes markers detected
  (CoexpressionNicheDE, bins where a gene pair is co-detected, lives in gene_pair.py)

Outputs per region ``name`` (e.g. "immune_niche", "immune_enriched_region"):
  {name}_depth_check.csv (depth before / after matching, one row per stage)
  {name}_de_genes.csv          all genes tested
  {name}_de_top20.csv          top_n up- and top_n down-regulated significant genes
  {name}_de_top20_dotplot.pdf
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.stats import mannwhitneyu

from ._logging import get_logger
from .depth import DepthModel
from .plot_style import save_pdf
from .spatial import SpatialPlotter

log = get_logger(__name__)


class DepthMatcher:
    """
    Keeps DE differences from being explained by sequencing depth.

    Marker-defined regions in sparse data are biased toward deep bins (more reads -> more markers
    detected); region-vs-background DE would then partly compare deep vs. shallow bins.
    """

    EXCLUDED_LABEL = "Excluded_depth_unmatched"

    def __init__(self, outdir: str | Path, n_depth_bins: int = 10, seed: int = 0, warn_auroc: float = 0.65):
        self.outdir = Path(outdir)
        self.n_depth_bins = n_depth_bins
        self.seed = seed
        self.warn_auroc = warn_auroc

    STAGES = ("before_matching", "after_matching")

    def depth_check(self, adata: AnnData, mask_col: str, region_label: str, background_label: str,
                    name: str, stage: str = "before_matching") -> dict:
        """
        Median total counts / genes of region vs background and the depth AUROC (P(a random region bin
        has more counts than a random background bin); 0.5 = same depth). Returns one row; :meth:`prepare`
        writes the rows of all stages to {name}_depth_check.csv.
        """
        tc = DepthModel.total_counts(adata)
        ng = DepthModel.n_genes(adata)
        labels = adata.obs[mask_col].astype(str).values
        in_r, in_b = labels == region_label, labels == background_label
        if in_r.sum() == 0 or in_b.sum() == 0:
            raise ValueError(f"region_depth_check: empty group ({region_label}={in_r.sum()}, "
                             f"{background_label}={in_b.sum()}). If the region covers every bin, the region "
                             "definition is too permissive for this data (e.g. raise min_genes / min_frac, "
                             "or expr_threshold) - there is no background to compare against.")
        u = mannwhitneyu(tc[in_r], tc[in_b], alternative="two-sided")
        auroc = float(u.statistic / (in_r.sum() * in_b.sum()))
        res = {
            "name": name,
            "stage": stage,
            "n_region": int(in_r.sum()),
            "n_background": int(in_b.sum()),
            "median_total_counts_region": float(np.median(tc[in_r])),
            "median_total_counts_background": float(np.median(tc[in_b])),
            "median_n_genes_region": float(np.median(ng[in_r])),
            "median_n_genes_background": float(np.median(ng[in_b])),
            "depth_auroc": auroc,
            "depth_mwu_p": float(u.pvalue),
            "depth_biased": bool(auroc > self.warn_auroc or auroc < 1 - self.warn_auroc),
        }
        log.debug(f"Depth check ({name}, {stage}): median total counts region {res['median_total_counts_region']:.0f} "
                 f"vs background {res['median_total_counts_background']:.0f}; depth AUROC = {auroc:.2f}")
        return res

    def match(self, adata: AnnData, mask_col: str, region_label: str, background_label: str) -> str:
        """
        Coarsened exact matching: split bins into n_depth_bins quantile bins of log(total counts); inside
        each keep the same number of region and background bins (min of the two, sampled at random).
        Unkept bins are relabeled EXCLUDED_LABEL. Returns the new obs column f"{mask_col}_depth_matched".
        """
        rng = np.random.default_rng(self.seed)
        tc = np.log1p(DepthModel.total_counts(adata))
        depth_bin = pd.qcut(pd.Series(tc).rank(method="first"), q=self.n_depth_bins, labels=False).values
        labels = adata.obs[mask_col].astype(str).values
        out = np.full(adata.n_obs, self.EXCLUDED_LABEL, dtype=object)
        for b in np.unique(depth_bin):
            r_idx = np.where((depth_bin == b) & (labels == region_label))[0]
            b_idx = np.where((depth_bin == b) & (labels == background_label))[0]
            k = min(len(r_idx), len(b_idx))
            if k == 0:
                continue
            out[rng.choice(r_idx, size=k, replace=False)] = region_label
            out[rng.choice(b_idx, size=k, replace=False)] = background_label
        new_col = f"{mask_col}_depth_matched"
        adata.obs[new_col] = pd.Categorical(out)
        n_kept = int((out == region_label).sum())
        log.debug(f"Depth matching: kept {n_kept} region + {n_kept} background bins "
                 f"(of {int((labels == region_label).sum())} region / {int((labels == background_label).sum())} background).")
        if n_kept == 0:
            raise ValueError("Depth matching left no bins - region and background do not overlap in depth at all.")
        if n_kept < 50:
            log.warning(f"only {n_kept} matched bins per group. The region covers almost all bins of comparable "
                        "depth (typical for min_genes=1), so a depth-fair comparison is barely possible - "
                        "the region definition is too permissive for DE; raise min_genes/min_frac.")
        return new_col

    def prepare(self, adata: AnnData, mask_col: str, region_label: str, background_label: str, name: str,
                depth_matched: bool = True) -> str:
        """
        Depth check, optional matching (+ check after matching); returns the obs column to use for DE.
        Writes ONE table, {name}_depth_check.csv: a row per stage ("before_matching", "after_matching").
        """
        rows = [self.depth_check(adata, mask_col, region_label, background_label, name, stage=self.STAGES[0])]
        de_col = mask_col
        if depth_matched:
            de_col = self.match(adata, mask_col, region_label, background_label)
            rows.append(self.depth_check(adata, de_col, region_label, background_label, name, stage=self.STAGES[1]))
        pd.DataFrame(rows).to_csv(self.outdir / f"{name}_depth_check.csv", index=False)
        final = rows[-1]                     # warn only if the bins the DE uses still differ in depth
        if final["depth_biased"]:
            log.warning(f"{name}: region and background still differ in depth "
                        f"({'after matching' if depth_matched else 'no depth matching'}; AUROC "
                        f"{final['depth_auroc']:.2f}) - its DE genes partly reflect sequencing depth.")
        return de_col


@dataclass
class RegionSpec:
    """What a RegionDE subclass defined: the mask column, labels, file-name stem and mask-map settings."""
    mask_col: str
    region_label: str
    background_label: str
    name: str
    key_added: str
    palette: dict = field(default_factory=dict)
    title: str = ""
    save: str = ""


class RegionDE(ABC):
    """
    Template for region-vs-background DE. Subclasses implement :meth:`define_region`
    (label the bins, return a :class:`RegionSpec`); :meth:`run` does the rest.

    DE: sc.tl.rank_genes_groups (Wilcoxon by default, BH-adjusted); a gene is significant when
    pvals_adj < pval_cutoff and |logFC| > logfc_cutoff. ``top_n`` genes per direction go to the
    top table and the dot plot. ``depth_matched`` (default True) runs DE on depth-matched bins.
    """

    REGION_COLOR, BACKGROUND_COLOR = "#e34948", "#c3c2b7"

    def __init__(self, outdir: str | Path, plotter: SpatialPlotter | None = None, layer: str = "lognorm",
                 method: str = "wilcoxon", pval_cutoff: float = 0.05, logfc_cutoff: float = 0.5, top_n: int = 20,
                 plot: bool = True, gene_symbols: str | None = None, depth_matched: bool = True,
                 n_depth_bins: int = 10, seed: int = 0):
        self.outdir = Path(outdir)
        self.plotter = plotter or SpatialPlotter(self.outdir)
        self.layer = layer
        self.method = method
        self.pval_cutoff = pval_cutoff
        self.logfc_cutoff = logfc_cutoff
        self.top_n = top_n
        self.plot = plot
        self.gene_symbols = gene_symbols
        self.depth_matched = depth_matched
        self.matcher = DepthMatcher(self.outdir, n_depth_bins=n_depth_bins, seed=seed)

    @abstractmethod
    def define_region(self, adata: AnnData) -> RegionSpec:
        """Label region / background bins in adata.obs and describe them."""

    def run(self, adata: AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Region -> mask map -> depth check / matching -> DE tables -> dot plot. Returns (degs_df, top_table)."""
        spec = self.define_region(adata)
        if self.plot:
            self.plotter.plot(adata, color=[spec.mask_col], palette=spec.palette, frameon=False,
                              title=spec.title, save=spec.save)
        de_col = self.matcher.prepare(adata, spec.mask_col, spec.region_label, spec.background_label, spec.name,
                                      depth_matched=self.depth_matched)
        return self.differential_expression(adata, de_col, spec)

    # ------------------------------------------------------------------ DE and outputs
    def differential_expression(self, adata: AnnData, de_col: str, spec: RegionSpec):
        region_label, background_label, name = spec.region_label, spec.background_label, spec.name
        log.debug(f"Running {self.method} test: {region_label} vs. {background_label} (layer={self.layer!r})...")
        sc.tl.rank_genes_groups(adata, groupby=de_col, groups=[region_label], layer=self.layer, method=self.method,
                                reference=background_label, key_added=spec.key_added, use_raw=False)
        degs_df = sc.get.rank_genes_groups_df(adata, group=region_label, key=spec.key_added,
                                              gene_symbols=self.gene_symbols)
        degs_df["significant"] = (degs_df["pvals_adj"] < self.pval_cutoff) & \
                                 (degs_df["logfoldchanges"].abs() > self.logfc_cutoff)
        degs_df = degs_df.sort_values("logfoldchanges", ascending=False).reset_index(drop=True)
        degs_path = self.outdir / f"{name}_de_genes.csv"
        degs_df.to_csv(degs_path, index=False)
        log.debug(f"Wrote full DE table ({len(degs_df)} genes, {int(degs_df['significant'].sum())} "
                 f"significant at padj<{self.pval_cutoff} & |logFC|>{self.logfc_cutoff}) to {degs_path}")

        sig_df = degs_df[degs_df["significant"]]
        up = sig_df[sig_df["logfoldchanges"] > 0].sort_values("logfoldchanges", ascending=False).head(self.top_n).copy()
        up["direction"] = "up"
        down = sig_df[sig_df["logfoldchanges"] < 0].sort_values("logfoldchanges", ascending=True).head(self.top_n).copy()
        down["direction"] = "down"
        top_table = pd.concat([up, down], ignore_index=True)
        top_path = self.outdir / f"{name}_de_top20.csv"
        top_table.to_csv(top_path, index=False)
        log.debug(f"Wrote top {len(up)} up- and {len(down)} down-regulated genes "
                 f"(of requested top_n={self.top_n} each) to {top_path}")
        if self.plot:
            self.plot_top_dotplot(adata, top_table, de_col, region_label, background_label, name)
        return degs_df, top_table

    def plot_top_dotplot(self, adata: AnnData, top_table: pd.DataFrame, group_col: str, region_label: str,
                         background_label: str, name: str, title: str | None = None) -> list[str]:
        """
        Dot plot of the top ``top_n`` up-regulated genes (all of them if fewer), region vs background, on the
        bins the DE used. Dot size = fraction of bins detected, color = mean expression. Saves
        {name}_de_top20_dotplot.pdf (publication quality); returns the genes plotted (none if nothing is up).
        """
        up = top_table[top_table["direction"] == "up"] if "direction" in top_table else top_table
        genes = up.sort_values("logfoldchanges", ascending=False)["names"].astype(str).head(self.top_n).tolist()
        if not genes:
            log.info(f"Dot plot ({name}): no significantly up-regulated genes - skipped.")
            return []
        # A small AnnData with only what the plot needs. Never subset adata itself (adata[rows, genes].copy()):
        # that also subsets the bin x bin graphs in obsp, which on ~10^5 bins exceeds scipy's 32-bit index range.
        rows = np.flatnonzero(adata.obs[group_col].astype(str).isin([region_label, background_label]).values)
        if self.gene_symbols is None:
            cols = np.flatnonzero(adata.var_names.isin(genes))
        else:
            cols = np.flatnonzero(adata.var[self.gene_symbols].astype(str).isin(genes).values)
        matrix = adata.layers[self.layer] if self.layer is not None else adata.X
        expr = matrix[:, cols][rows]
        var = (adata.var.iloc[cols][[self.gene_symbols]] if self.gene_symbols is not None
               else pd.DataFrame(index=adata.var_names[cols]))
        groups = pd.Categorical(adata.obs[group_col].astype(str).values[rows], categories=[region_label, background_label])
        sub = sc.AnnData(X=expr, obs=pd.DataFrame({group_col: groups}, index=adata.obs_names[rows]), var=var)
        dp = sc.pl.dotplot(sub, var_names=genes, groupby=group_col, use_raw=False, gene_symbols=self.gene_symbols,
                           cmap="Reds", title=title or f"{name}: top {len(genes)} up-regulated DE genes",
                           show=False, return_fig=True)
        dp.make_figure()
        path = save_pdf(dp.fig, self.outdir / f"{name}_de_top20_dotplot.pdf")
        plt.close("all")
        log.debug(f"Saved dot plot of {len(genes)} up-regulated genes to {path}")
        return genes

    # ------------------------------------------------------------------ helpers for subclasses
    @staticmethod
    def _min_active(n_genes: int, min_genes, min_frac, default: int) -> int:
        """Genes that must be active per bin: min_genes, else ceil(min_frac * n), else default."""
        if min_genes is not None:
            k = int(min_genes)
        elif min_frac is not None:
            k = int(np.ceil(min_frac * n_genes))
        else:
            k = default
        if not (1 <= k <= n_genes):
            raise ValueError(f"min_genes/min_frac resolved to k={k}, which must be between 1 and "
                             f"{n_genes} (the number of genes present).")
        return k

    def _n_active(self, adata: AnnData, genes: list[str], expr_threshold: float) -> np.ndarray:
        return np.column_stack([adata.obs_vector(g, layer=self.layer) > expr_threshold for g in genes]).sum(axis=1)


class NicheDE(RegionDE):
    """
    Strategy 1, high-confidence niche: bins whose smoothed signature score (default
    f"{celltype}_score_smoothed_1.0", written by SignatureAnalysis) is strictly above the
    ``quantile`` (default 0.95 = top 5%; config niche.quantile) -> f"{celltype}_niche"; the rest is
    "Background". Writes {celltype}_niche_definition.csv (top %, score threshold, bins).
    Answers "does this bin show coordinated, panel-wide enrichment?"
    """

    def __init__(self, outdir, celltype: str, score_col: str | None = None, quantile: float = 0.95, **kwargs):
        super().__init__(outdir, **kwargs)
        self.celltype = celltype
        self.score_col = score_col or f"{celltype}_score_smoothed_1.0"
        self.quantile = quantile

    def define_region(self, adata):
        ct, score_col, quantile = self.celltype, self.score_col, self.quantile
        if score_col not in adata.obs.columns:
            raise ValueError(f"{score_col} not found in adata.obs - run SignatureAnalysis (score + Gaussian "
                             "smoothing) for this celltype first.")
        niche_label, mask_col = f"{ct}_niche", f"{ct}_niche_mask"
        threshold = float(adata.obs[score_col].quantile(quantile))
        adata.obs[mask_col] = pd.Categorical(np.where(adata.obs[score_col] > threshold, niche_label, "Background"))
        n_niche = int((adata.obs[mask_col] == niche_label).sum())
        log.debug(f"--- {ct.capitalize()} spatial niche segregation ---")
        log.debug(f"Isolated {n_niche} / {adata.n_obs} bins ({quantile:.0%} quantile threshold on {score_col} = "
                 f"{threshold:.4f}) as the high-confidence {niche_label}.")
        if n_niche == 0:
            raise ValueError(f"No bins exceeded the {quantile:.0%} quantile threshold on {score_col} - "
                             "try a lower quantile, or check that the score column has variation.")
        pd.DataFrame([{"celltype": ct, "score_col": score_col, "top_percent": round(100 * (1 - quantile), 4),
                       "quantile": quantile, "score_threshold": threshold, "n_niche_bins": n_niche,
                       "n_bins": int(adata.n_obs)}]).to_csv(self.outdir / f"{ct}_niche_definition.csv", index=False)
        return RegionSpec(mask_col, niche_label, "Background", f"{ct}_niche", f"{ct}_niche_degs",
                          palette={niche_label: self.REGION_COLOR, "Background": self.BACKGROUND_COLOR},
                          title=(f"High-confidence {ct} niche: top {100 * (1 - quantile):.0f}% of bins\n"
                                 f"by {score_col} (threshold = {threshold:.4f}; {n_niche} bins)"),
                          save=f"_{ct}_niche_mask")


class MarkerPanelDE(RegionDE):
    """
    Strategy 2, marker-panel enriched region: bins where at least ``min_genes`` (default 1 - ANY
    single marker) of the panel are individually detected above ``expr_threshold``
    -> f"{celltype}_enriched_region". ``min_frac`` (fraction of the present panel, rounded up) is the
    alternative to min_genes. The most permissive definition - a high-sensitivity screen for sparse
    data, a much weaker claim per bin than NicheDE; the two select different bins.
    """

    def __init__(self, outdir, celltype: str, marker_list: list[str], expr_threshold: float = 0.0,
                 min_genes: int | None = None, min_frac: float | None = None, **kwargs):
        super().__init__(outdir, **kwargs)
        if len(marker_list) < 1:
            raise ValueError("marker_list must contain at least 1 gene.")
        self.celltype = celltype
        self.marker_list = list(marker_list)
        self.expr_threshold = expr_threshold
        self.min_genes = min_genes
        self.min_frac = min_frac

    def define_region(self, adata):
        ct = self.celltype
        genes = [g for g in self.marker_list if g in adata.var_names]
        missing = sorted(set(self.marker_list) - set(genes))
        if missing:
            log.warning(f"{ct}: {len(missing)} of {len(self.marker_list)} markers not in the data, skipped: "
                        f"{', '.join(missing)}")
        if not genes:
            raise ValueError(f"None of {self.marker_list} were found in adata.var_names.")
        n_genes = len(genes)
        k = self._min_active(n_genes, self.min_genes, self.min_frac, default=1)
        region_label, mask_col = f"{ct}_enriched_region", f"{ct}_enriched_mask"
        is_enriched = self._n_active(adata, genes, self.expr_threshold) >= k
        adata.obs[mask_col] = pd.Categorical(np.where(is_enriched, region_label, "Background"))
        n_region = int(is_enriched.sum())
        log.debug(f"--- {ct.capitalize()} marker panel enrichment ({n_genes} markers present) ---")
        log.debug(f">= {k}/{n_genes} marker(s) detected (> {self.expr_threshold} in {self.layer!r}) required per bin.")
        log.debug(f"Isolated {n_region} / {adata.n_obs} bins as {region_label}.")
        if n_region == 0:
            raise ValueError(f"No bins have >= {k}/{n_genes} of the {ct} panel above {self.expr_threshold} in "
                             f"layer {self.layer!r} - try a lower expr_threshold or a smaller min_genes/min_frac.")
        return RegionSpec(mask_col, region_label, "Background", f"{ct}_enriched_region", f"{ct}_enriched_degs",
                          palette={region_label: self.REGION_COLOR, "Background": self.BACKGROUND_COLOR},
                          title=f"{ct.capitalize()} marker panel enrichment (>= {k}/{n_genes} detected)",
                          save=f"_{ct}_enriched_region_mask")
