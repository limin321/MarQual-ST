"""
Cell-type co-localization from marker gene sets.

"Are cell types A and B spatially co-localized - more than expected for arbitrary genes of
similar expression, and not just because of sequencing depth or a shared compartment?"
This is co-localization, NOT interaction: two cell types can share a niche because they take
part in the same process, without direct ligand-receptor contact.

  SignatureColocalization.test         I_AB over each bin + its neighbours vs. two matched
                                       random-gene-set nulls (swap A, swap B)
  SignatureColocalization.by_distance  the same statistic in distance rings (spatial scale)
  ColocalizationTable                  FDR across the pairs of the run (the family = n_tested)

Statistic: for standardized, depth-residualized scores z_a, z_b and a symmetric binary
neighbour matrix K (bin itself + neighbours, or one ring):  I_AB = z_a^T K z_b / sum(K).
Bivariate Moran's I with global normalization; K symmetric -> I_AB = I_BA; K = identity
gives the same-bin Pearson correlation.

Why two nulls: swapping only one side can mislead (a strongly expressed panel A depresses
every other gene where A is high after normalize_total, so any panel B would look
"segregated" against a swap-A null alone). The reported p-value is the LARGER of the two.
p_colocalized / p_segregated are the primary evidence; z_swapA / z_swapB are descriptive
(a skewed null inflates the SD and understates z; the p-value is unaffected).
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.signal import fftconvolve
from statsmodels.stats.multitest import multipletests

from ._logging import get_logger
from .depth import DepthModel, MatchedGeneSetSampler
from .plot_style import save_pdf
from .spatial import SpatialGrid

log = get_logger(__name__)

COLOCALIZED, SEGREGATED, NOT_RANDOM = "CO-LOCALIZED", "SEGREGATED", "NOT_DISTINGUISHABLE_FROM_RANDOM"


def _style(ax):
    ax.tick_params(colors="#898781")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


class SignatureColocalization:
    """
    Shared settings for the co-localization tests of one run.

    residualize / covariates / compartment_col : depth (and optional compartment) removal from both scores.
        compartment_col (e.g. "tissue_section") asks for co-localization WITHIN compartments; do not use
        compartments defined from these cell types' expression.
    include_self : K includes the bin itself (at bin50 one bin holds several cells, so same-bin
        co-occurrence is the closest proximity there is).
    tag_suffix : appended to the pair name and file names (e.g. "_within_section") so control runs do
        not overwrite the main result.
    """

    DEFAULT_RINGS = ((0, 0), (0, 1.5), (1.5, 3), (3, 6), (6, 12))

    def __init__(self, outdir: str | Path, n_null: int = 1000, residualize: bool = True,
                 covariates: Iterable[str] = ("log_total_counts",), compartment_col: str | None = None,
                 include_self: bool = True, layer: str = "lognorm", n_bins: int = 10,
                 weight_key: str = SpatialGrid.WEIGHT_KEY, alpha: float = 0.05, seed: int = 0, plot: bool = True,
                 tag_suffix: str = ""):
        self.outdir = Path(outdir)
        self.n_null = n_null
        self.residualize = residualize
        self.covariates = tuple(covariates)
        self.compartment_col = compartment_col
        self.include_self = include_self
        self.layer = layer
        self.n_bins = n_bins
        self.weight_key = weight_key
        self.alpha = alpha
        self.seed = seed
        self.plot = plot
        self.tag_suffix = tag_suffix

    # ------------------------------------------------------------------ helpers
    @property
    def depth(self) -> DepthModel:
        return DepthModel(self.covariates if self.residualize else (), self.compartment_col if self.residualize else None)

    @property
    def residualized_on(self) -> str:
        return self.depth.label if self.residualize else "none"

    def _score(self, adata, genes):
        m = adata[:, genes].layers[self.layer]
        return np.asarray(m.mean(axis=1)).ravel() if sp.issparse(m) else np.asarray(m).mean(axis=1)

    @staticmethod
    def _standardize(v):
        v = np.asarray(v, dtype=float)
        v = v - v.mean()
        sd = v.std()
        return v / sd if sd > 0 else v

    @staticmethod
    def prepare_panels(adata, marker_list_a, marker_list_b, name_a, name_b):
        """Markers present in the data; genes shared by both panels are removed (they would make the
        two scores correlated by construction)."""
        a = [g for g in dict.fromkeys(marker_list_a) if g in adata.var_names]
        b = [g for g in dict.fromkeys(marker_list_b) if g in adata.var_names]
        shared = sorted(set(a) & set(b))
        if shared:
            log.warning(f"{shared} are in both the {name_a} and {name_b} panels - removed from both, "
                        "otherwise the two scores are correlated by construction.")
            a = [g for g in a if g not in shared]
            b = [g for g in b if g not in shared]
        if not a or not b:
            raise ValueError(f"After filtering, {name_a} has {len(a)} and {name_b} has {len(b)} markers present.")
        log.debug(f"{name_a}: {len(a)}/{len(marker_list_a)} markers present; "
                 f"{name_b}: {len(b)}/{len(marker_list_b)} markers present.")
        return a, b

    @staticmethod
    def p_values(obs, null):
        n = len(null)
        return (float((np.sum(null >= obs) + 1) / (n + 1)),   # co-localized (greater)
                float((np.sum(null <= obs) + 1) / (n + 1)))   # segregated (less)

    @staticmethod
    def z(obs, null):
        sd = null.std()
        return float((obs - null.mean()) / sd) if sd > 0 else np.nan

    @staticmethod
    def verdict(p_coloc, p_seg, alpha):
        if p_coloc < alpha:
            return COLOCALIZED
        if p_seg < alpha:
            return SEGREGATED
        return NOT_RANDOM

    def _samplers(self, adata, genes_a, genes_b):
        exclude = set(genes_a) | set(genes_b)
        return (MatchedGeneSetSampler(adata, genes_a, self.layer, self.n_bins, exclude_genes=exclude, seed=self.seed),
                MatchedGeneSetSampler(adata, genes_b, self.layer, self.n_bins, exclude_genes=exclude, seed=self.seed + 1))

    # ------------------------------------------------------------------ main test
    def test(self, adata: AnnData, marker_list_a, marker_list_b, name_a: str, name_b: str) -> dict:
        """
        I_AB of the two depth-residualized signature scores over each bin + its neighbours, against the
        swap-A and swap-B nulls (n_null draws each). Writes {tag}_colocalization_summary.csv,
        {tag}_colocalization_null.csv and (plot) {tag}_colocalization_null_plot.pdf.
        """
        genes_a, genes_b = self.prepare_panels(adata, marker_list_a, marker_list_b, name_a, name_b)
        if self.weight_key not in adata.obsp:
            raise ValueError(f"{self.weight_key!r} not in adata.obsp - run SpatialGrid.build_graph(adata) first.")
        K = (adata.obsp[self.weight_key] > 0).astype(float)          # symmetric binary neighbour matrix (+ self)
        K = K.maximum(K.T)
        if self.include_self:
            K = K + sp.identity(adata.n_obs, format="csr")
        K = sp.csr_matrix(K)
        S0 = float(K.sum())

        resid = self.depth.residualizer(adata)
        raw_a, raw_b = self._score(adata, genes_a), self._score(adata, genes_b)
        za, zb = self._standardize(resid(raw_a)), self._standardize(resid(raw_b))
        lag_a, lag_b = K @ za, K @ zb
        I_obs = float(za @ lag_b / S0)
        I_raw = float(self._standardize(raw_a) @ (K @ self._standardize(raw_b)) / S0)   # no depth removal
        same_bin_r = float(za @ zb / len(za))

        draw_a, draw_b = self._samplers(adata, genes_a, genes_b)
        n_null = self.n_null
        null_swap_a, null_swap_b = np.empty(n_null), np.empty(n_null)
        for i in range(n_null):
            null_swap_a[i] = self._standardize(resid(draw_a.draw())) @ lag_b / S0   # K symmetric
            null_swap_b[i] = self._standardize(resid(draw_b.draw())) @ lag_a / S0

        pc_a, ps_a = self.p_values(I_obs, null_swap_a)
        pc_b, ps_b = self.p_values(I_obs, null_swap_b)
        p_coloc, p_seg = max(pc_a, pc_b), max(ps_a, ps_b)
        tag = f"{name_a}_vs_{name_b}{self.tag_suffix}"
        result = {
            "pair": tag, "celltype_a": name_a, "celltype_b": name_b,
            "n_markers_a": len(genes_a), "n_markers_b": len(genes_b),
            "I_AB": I_obs, "I_AB_no_depth_correction": I_raw, "same_bin_corr": same_bin_r,
            "null_swapA_mean": float(null_swap_a.mean()), "null_swapA_sd": float(null_swap_a.std()),
            "null_swapB_mean": float(null_swap_b.mean()), "null_swapB_sd": float(null_swap_b.std()),
            "z_swapA": self.z(I_obs, null_swap_a), "z_swapB": self.z(I_obs, null_swap_b),
            "p_colocalized": p_coloc, "p_segregated": p_seg,
            "verdict": self.verdict(p_coloc, p_seg, self.alpha),
            "residualized_on": self.residualized_on, "include_self": self.include_self, "n_null": n_null,
        }
        pd.DataFrame([result]).to_csv(self.outdir / f"{tag}_colocalization_summary.csv", index=False)
        pd.DataFrame({"null_swapA": null_swap_a, "null_swapB": null_swap_b}).to_csv(
            self.outdir / f"{tag}_colocalization_null.csv", index=False)
        log.debug(f"{tag}: I_AB = {I_obs:.4f} (no depth correction {I_raw:.4f}); "
                 f"z vs swap-{name_a} = {result['z_swapA']:.1f}, vs swap-{name_b} = {result['z_swapB']:.1f}; "
                 f"p_colocalized = {p_coloc:.3g}, p_segregated = {p_seg:.3g} -> {result['verdict']}")
        if self.plot:
            self._plot_nulls(I_obs, null_swap_a, null_swap_b, name_a, name_b, result["verdict"], tag)
        return result

    def _plot_nulls(self, I_obs, null_a, null_b, name_a, name_b, verdict, tag):
        fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.3))
        for ax, null, who in [(axes[0], null_a, name_a), (axes[1], null_b, name_b)]:
            ax.hist(null, bins=40, color="#2a78d6", edgecolor="none", zorder=2)
            ax.axvline(I_obs, color="#e34948", linestyle="--", linewidth=1.0, zorder=3,
                       label=f"Observed $I_{{AB}}$ = {I_obs:.3f}")
            ax.set_xlabel(f"$I_{{AB}}$ with {who} replaced by random gene sets", color="#52514e")
            ax.set_ylabel(f"Count (n = {len(null)})", color="#52514e")
            ax.text(0.02, 0.95, f"z = {self.z(I_obs, null):.1f}", transform=ax.transAxes, fontsize=6, va="top",
                    color="#52514e")
            ax.legend(frameon=False, loc="upper right", fontsize=5.5)
            _style(ax)
        fig.suptitle(f"{name_a} vs {name_b}: {verdict}", fontsize=7.5, color="#0b0b0b")
        fig.tight_layout()
        save_pdf(fig, self.outdir / f"{tag}_colocalization_null_plot.pdf")
        plt.close(fig)

    # ------------------------------------------------------------------ spatial scale
    def by_distance(self, adata: AnnData, marker_list_a, marker_list_b, name_a: str, name_b: str,
                    n_null: int = 200, rings=DEFAULT_RINGS, um_per_bin: float | None = None) -> pd.DataFrame:
        """
        I_AB in distance rings, each with its own swap-A / swap-B null. rings: (inner, outer] distances in
        bins; (0, 0) = same bin only; annuli so each point measures ONE distance. um_per_bin only labels
        the x-axis / distance_um_mid. Strongest at short range and fading -> shared local niche; flat -> a
        shared broad region; negative at short range -> local exclusion. Ring sums are FFT convolutions on
        the grid; random gene sets are drawn once and scored against all rings.
        Writes {tag}_colocalization_by_distance.csv (+ plot).
        """
        genes_a, genes_b = self.prepare_panels(adata, marker_list_a, marker_list_b, name_a, name_b)
        xi, yi = SpatialGrid.positions(adata)
        shape = (xi.max() + 1, yi.max() + 1)
        occ = np.zeros(shape)
        np.add.at(occ, (xi, yi), 1.0)

        def _ring_kernel(lo, hi):
            if hi == 0:
                return np.ones((1, 1))
            r = int(np.ceil(hi))
            dx, dy = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
            d = np.sqrt(dx ** 2 + dy ** 2)
            return ((d > lo + 1e-9) & (d <= hi + 1e-9)).astype(float)

        def _lag(zv, kern):
            g = np.zeros(shape)
            np.add.at(g, (xi, yi), zv)
            return fftconvolve(g, kern, mode="same")[xi, yi]

        resid = self.depth.residualizer(adata)
        za = self._standardize(resid(self._score(adata, genes_a)))
        zb = self._standardize(resid(self._score(adata, genes_b)))
        kernels = [_ring_kernel(lo, hi) for lo, hi in rings]
        S0 = np.array([np.round(fftconvolve(occ, k, mode="same")[xi, yi]).sum() for k in kernels])
        LagA = np.column_stack([_lag(za, k) for k in kernels])   # n x R
        LagB = np.column_stack([_lag(zb, k) for k in kernels])
        I_obs = (za @ LagB) / S0

        draw_a, draw_b = self._samplers(adata, genes_a, genes_b)
        null_a, null_b = np.empty((n_null, len(rings))), np.empty((n_null, len(rings)))
        for i in range(n_null):
            null_a[i] = (self._standardize(resid(draw_a.draw())) @ LagB) / S0
            null_b[i] = (self._standardize(resid(draw_b.draw())) @ LagA) / S0

        rows = []
        for j, (lo, hi) in enumerate(rings):
            pc_a, ps_a = self.p_values(I_obs[j], null_a[:, j])
            pc_b, ps_b = self.p_values(I_obs[j], null_b[:, j])
            p_coloc, p_seg = max(pc_a, pc_b), max(ps_a, ps_b)
            row = {
                "ring_inner_bins": lo, "ring_outer_bins": hi,
                "ring_label": "same bin" if hi == 0 else f"({lo}, {hi}] bins",
                "n_pairs": int(S0[j]), "I_AB": float(I_obs[j]),
                "null_swapA_lo": float(np.quantile(null_a[:, j], 0.025)),
                "null_swapA_hi": float(np.quantile(null_a[:, j], 0.975)),
                "null_swapB_lo": float(np.quantile(null_b[:, j], 0.025)),
                "null_swapB_hi": float(np.quantile(null_b[:, j], 0.975)),
                "z_swapA": self.z(I_obs[j], null_a[:, j]), "z_swapB": self.z(I_obs[j], null_b[:, j]),
                "p_colocalized": p_coloc, "p_segregated": p_seg,
                "verdict": self.verdict(p_coloc, p_seg, self.alpha),
            }
            if um_per_bin is not None:
                row["distance_um_mid"] = (lo + hi) / 2 * um_per_bin
            rows.append(row)
        df = pd.DataFrame(rows)
        tag = f"{name_a}_vs_{name_b}{self.tag_suffix}"
        df.to_csv(self.outdir / f"{tag}_colocalization_by_distance.csv", index=False)
        log.debug("\n" + df[["ring_label", "I_AB", "z_swapA", "z_swapB", "p_colocalized", "verdict"]].to_string(index=False))
        if self.plot:
            self._plot_distance(df, rings, um_per_bin, name_a, name_b, tag)
        return df

    def _plot_distance(self, df, rings, um_per_bin, name_a, name_b, tag):
        mids = np.array([(lo + hi) / 2 for lo, hi in rings]) * (um_per_bin or 1)
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        ax.fill_between(mids, df["null_swapA_lo"], df["null_swapA_hi"], color="#2a78d6", alpha=0.18,
                        label=f"95% null, {name_a} swapped", zorder=1)
        ax.fill_between(mids, df["null_swapB_lo"], df["null_swapB_hi"], color="#898781", alpha=0.18,
                        label=f"95% null, {name_b} swapped", zorder=1)
        ax.plot(mids, df["I_AB"], color="#e34948", marker="o", markersize=3.5, linewidth=1.0,
                label="Observed $I_{AB}$", zorder=3)
        sig = df["p_colocalized"] < self.alpha
        ax.scatter(mids[sig], df["I_AB"][sig], s=28, facecolors="none", edgecolors="#0b0b0b", linewidths=0.6,
                   zorder=4, label=f"p_colocalized < {self.alpha}")
        ax.axhline(0, color="#c3c2b7", linewidth=0.6)
        ax.set_xlabel("Distance (\u00b5m, ring midpoint)" if um_per_bin else "Distance (bins, ring midpoint)",
                      color="#52514e")
        ax.set_ylabel("Co-localization $I_{AB}$", color="#52514e")
        ax.set_title(f"{name_a} vs {name_b}: co-localization by distance", fontsize=7.5, color="#0b0b0b")
        ax.legend(frameon=False, fontsize=5.5)
        _style(ax)
        fig.tight_layout()
        save_pdf(fig, self.outdir / f"{tag}_colocalization_by_distance.pdf")
        plt.close(fig)


class ColocalizationTable:
    """
    FDR-correct co-localization p-values ACROSS all pairs of the run (co-localized and segregated
    each as their own family) and assign verdict_fdr. The family is exactly the pairs passed in
    (column n_tested). Writes all_pairs_colocalization_table.csv.
    """

    FILE = "all_pairs_colocalization_table.csv"

    def __init__(self, outdir: str | Path, alpha: float = 0.05, method: str = "fdr_bh"):
        self.outdir = Path(outdir)
        self.alpha = alpha
        self.method = method

    def aggregate(self, results: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(results)
        df["n_tested"] = len(df)
        for col in ["p_colocalized", "p_segregated"]:
            _, q, _, _ = multipletests(df[col].values, alpha=self.alpha, method=self.method)
            df[col.replace("p_", "q_")] = q
        df["verdict_fdr"] = [COLOCALIZED if qc < self.alpha else (SEGREGATED if qs < self.alpha else NOT_RANDOM)
                             for qc, qs in zip(df["q_colocalized"], df["q_segregated"])]
        df = df.sort_values("p_colocalized").reset_index(drop=True)
        df.to_csv(self.outdir / self.FILE, index=False)
        return df
