"""
Statistical controls that turn a Moran's I number into defensible evidence.

  RandomGeneSetNullTest   observed Moran's I vs. many size- and expression-matched random
                          gene sets, on raw AND depth-residualized scores (z, empirical p, depth R^2)
  MoranIndexTable         FDR across the marker sets of the run (the family = n_tested) and
                          qc_verdict: PASS / DEPTH_DRIVEN / MASKED_BY_DEPTH / NO_SPATIAL_SIGNAL
  DepthConfoundChecks     Moran's I of total counts (baseline) and score-vs-depth Spearman
  NullDistributionPlotter histogram of the null with the observed value marked
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from ._logging import get_logger
from .depth import DepthModel, MatchedGeneSetSampler, moran_i, row_normalize
from .plot_style import save_pdf
from .spatial import SpatialGrid

log = get_logger(__name__)


class RandomGeneSetNullTest:
    """
    Compare the observed marker-set Moran's I against random gene sets of the same size,
    matched to the markers' expression bins.

    * residualize=True: the test is ALSO run on depth-residualized scores (observed score and
      every random-set score residualized the same way). Raw and residualized results together
      are the QC readout: significant on both -> structure beyond depth; raw only -> depth-driven.
    * z = (observed_I - null_mean) / null_sd: an effect size comparable across marker sets.
    * depth_R2: fraction of the score's variance explained by depth.

    Writes {celltype}_null_moran_distribution.csv and {celltype}_null_test_summary.csv.
    """

    def __init__(self, outdir: str | Path, n_null: int = 1000, n_bins: int = 10, layer: str = "lognorm",
                 seed: int = 0, weight_key: str = SpatialGrid.WEIGHT_KEY, residualize: bool = True,
                 covariates: Iterable[str] = ("log_total_counts",), compartment_col: str | None = None):
        self.outdir = Path(outdir)
        self.n_null = n_null
        self.n_bins = n_bins
        self.layer = layer
        self.seed = seed
        self.weight_key = weight_key
        self.residualize = residualize
        self.depth = DepthModel(covariates, compartment_col)

    @staticmethod
    def _summary(obs: float, null: np.ndarray, n_null: int, suffix: str = "") -> dict:
        sd = float(null.std())
        return {
            f"observed_I{suffix}": float(obs),
            f"null_mean_I{suffix}": float(null.mean()),
            f"null_sd_I{suffix}": sd,
            f"null_p95_I{suffix}": float(np.quantile(null, 0.95)),
            f"z_score{suffix}": float((obs - null.mean()) / sd) if sd > 0 else np.nan,
            f"empirical_p_value{suffix}": float((np.sum(null >= obs) + 1) / (n_null + 1)),
        }

    def run(self, adata: AnnData, marker_list: list[str], celltype: str) -> dict:
        marker_present = [g for g in marker_list if g in adata.var_names]
        if len(marker_present) == 0:
            raise ValueError(f"No {celltype} markers found in adata.var_names.")
        W = row_normalize(adata.obsp[self.weight_key])          # normalize once; reused 1 + n_null times
        score_col = f"{celltype}_score"
        if score_col not in adata.obs.columns:
            raise ValueError(f"{score_col} not found in adata.obs - run SignatureAnalysis for this celltype first.")
        score = np.asarray(adata.obs[score_col].values, dtype=float)
        observed_I = moran_i(score, W, row_normalized=True)

        if self.residualize:
            resid = self.depth.residualizer(adata)
            score_r = resid(score)
            observed_I_resid = moran_i(score_r, W, row_normalized=True)
            tss = ((score - score.mean()) ** 2).sum()
            depth_r2 = float(1 - (score_r ** 2).sum() / tss) if tss > 0 else np.nan
            adata.obs[f"{score_col}_resid"] = score_r

        sampler = MatchedGeneSetSampler(adata, marker_present, layer=self.layer, n_bins=self.n_bins, seed=self.seed)
        null_I = np.empty(self.n_null)
        null_I_resid = np.empty(self.n_null) if self.residualize else None
        for i in range(self.n_null):
            null_score = sampler.draw()
            null_I[i] = moran_i(null_score, W, row_normalized=True)
            if self.residualize:
                null_I_resid[i] = moran_i(resid(null_score), W, row_normalized=True)

        result = {
            "celltype": celltype,
            "n_markers_used": len(marker_present),
            "n_markers_requested": len(marker_list),
            "marker_coverage": len(marker_present) / len(marker_list),
        }
        result.update(self._summary(observed_I, null_I, self.n_null))
        if self.residualize:
            result.update(self._summary(observed_I_resid, null_I_resid, self.n_null, "_resid"))
            result["depth_R2"] = depth_r2
            result["residualized_on"] = self.depth.label
        result["n_null"] = self.n_null

        null_df = pd.DataFrame({"null_I": null_I})
        if self.residualize:
            null_df["null_I_resid"] = null_I_resid
        null_df.to_csv(self.outdir / f"{celltype}_null_moran_distribution.csv", index=False)
        pd.DataFrame([result]).to_csv(self.outdir / f"{celltype}_null_test_summary.csv", index=False)
        return result


class MoranIndexTable:
    """
    Pool per-marker-set null-test results and FDR-correct ACROSS all marker sets of the run
    (squidpy's pval_norm_fdr_bh, computed on one signature at a time, corrects nothing).

    The FDR family is exactly the marker sets passed in (column n_tested): adding marker sets
    makes the correction stricter, removing them looser, so results near q = 0.05 can change with
    the list. Read each verdict with its z, p, q and depth_R2 (report_guide.pdf, "Reading a verdict").

      PASS               significant raw and depth-corrected -> candidate niche
      DEPTH_DRIVEN       raw only -> the "pattern" is library-size variation
      MASKED_BY_DEPTH    depth-corrected only -> signal hidden by depth variation
      NO_SPATIAL_SIGNAL  neither
    ";LOW_MARKER_COVERAGE" is appended when < min_marker_coverage of the markers are in the data.

    Writes {outdir}/all_celltypes_moran_index_table.csv.
    """

    FILE = "all_celltypes_moran_index_table.csv"
    VERDICTS = {(True, True): "PASS", (True, False): "DEPTH_DRIVEN",
                (False, True): "MASKED_BY_DEPTH", (False, False): "NO_SPATIAL_SIGNAL"}

    def __init__(self, outdir: str | Path, alpha: float = 0.05, method: str = "fdr_bh", min_marker_coverage: float = 0.5):
        self.outdir = Path(outdir)
        self.alpha = alpha
        self.method = method
        self.min_marker_coverage = min_marker_coverage

    def _verdict(self, row) -> str:
        v = self.VERDICTS[(bool(row["significant_at_fdr"]), bool(row["significant_at_fdr_resid"]))]
        if "marker_coverage" in row and row["marker_coverage"] < self.min_marker_coverage:
            v += ";LOW_MARKER_COVERAGE"
        return v

    def aggregate(self, results: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(results)
        df["n_tested"] = len(df)
        reject, qval, _, _ = multipletests(df["empirical_p_value"].values, alpha=self.alpha, method=self.method)
        df["fdr_qvalue"] = qval
        df["significant_at_fdr"] = reject
        has_resid = "empirical_p_value_resid" in df.columns
        if has_resid:
            reject_r, qval_r, _, _ = multipletests(df["empirical_p_value_resid"].values, alpha=self.alpha,
                                                   method=self.method)
            df["fdr_qvalue_resid"] = qval_r
            df["significant_at_fdr_resid"] = reject_r
            df["qc_verdict"] = df.apply(self._verdict, axis=1)
        df = df.sort_values("empirical_p_value_resid" if has_resid else "empirical_p_value").reset_index(drop=True)
        df.to_csv(self.outdir / self.FILE, index=False)
        return df


class DepthConfoundChecks:
    """Two depth sanity checks next to the null test."""

    def __init__(self, outdir: str | Path, weight_key: str = SpatialGrid.WEIGHT_KEY):
        self.outdir = Path(outdir)
        self.weight_key = weight_key

    def total_counts_baseline(self, adata: AnnData) -> dict:
        """
        Moran's I of total counts, same weights as everything else. A marker signature whose I is close
        to this baseline may just track library-size variation (edges, folds, staining gradients).
        Writes total_counts_confound_baseline.csv.
        """
        baseline_I = moran_i(DepthModel.total_counts(adata), adata.obsp[self.weight_key])
        result = {"metric": "total_counts", "moran_I": baseline_I}
        pd.DataFrame([result]).to_csv(self.outdir / "total_counts_confound_baseline.csv", index=False)
        log.debug(f"Baseline Moran's I for total counts (nuisance check): {baseline_I:.4f}")
        return result

    def score_depth_correlation(self, adata: AnnData, celltype: str) -> dict:
        """
        Spearman correlation of a signature score with total counts per bin (non-spatial robustness check;
        Spearman because both are right-skewed). Writes {celltype}_score_total_counts_correlation.csv.
        """
        score_col = f"{celltype}_score"
        if score_col not in adata.obs.columns:
            raise ValueError(f"{score_col} not found in adata.obs - run SignatureAnalysis for this celltype first.")
        total_counts = DepthModel.total_counts(adata)
        rho, p_value = spearmanr(adata.obs[score_col].values, total_counts)
        result = {"celltype": celltype, "metric": "spearman_score_vs_total_counts", "rho": float(rho),
                  "p_value": float(p_value), "n": int(len(total_counts))}
        pd.DataFrame([result]).to_csv(self.outdir / f"{celltype}_score_total_counts_correlation.csv", index=False)
        log.debug(f"{score_col} vs total_counts: Spearman rho = {rho:.4f} (p = {p_value:.3g}, n = {len(total_counts)})")
        return result


class NullDistributionPlotter:
    """
    Figures of the random-gene-set null, read back from the CSVs RandomGeneSetNullTest wrote:
    one histogram per marker set, and a combined panel. ``which``: "raw" or "resid".
    """

    def __init__(self, outdir: str | Path, bins: int = 40, dpi: int = 600, formats=("pdf",)):
        self.outdir = Path(outdir)
        self.bins = bins
        self.dpi = dpi
        self.formats = formats

    @staticmethod
    def _style(ax):
        ax.tick_params(colors="#898781")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#c3c2b7")
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)

    def _save(self, fig, stem):
        """PDF through plot_style (publication quality); any other requested format as is."""
        for fmt in self.formats:
            if fmt == "pdf":
                save_pdf(fig, self.outdir / f"{stem}.pdf", dpi=self.dpi)
            else:
                fig.savefig(self.outdir / f"{stem}.{fmt}", dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def plot(self, celltype: str, which: str = "raw", figsize=(3.2, 2.4)):
        """{celltype}_null_distribution_plot[_resid].pdf"""
        sfx = "_resid" if which == "resid" else ""
        null_df = pd.read_csv(self.outdir / f"{celltype}_null_moran_distribution.csv")
        summary = pd.read_csv(self.outdir / f"{celltype}_null_test_summary.csv").iloc[0]
        observed_I = float(summary[f"observed_I{sfx}"])
        p_value = float(summary[f"empirical_p_value{sfx}"])
        z_value = float(summary[f"z_score{sfx}"]) if f"z_score{sfx}" in summary else np.nan
        n_null = int(summary["n_null"])

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(null_df[f"null_I{sfx}"], bins=self.bins, color="#2a78d6", edgecolor="none")
        ax.axvline(observed_I, color="#e34948", linestyle="--", linewidth=1.0, label=f"Observed I = {observed_I:.3f}")
        ax.set_xlabel("Moran's I (expression- and size-matched random gene sets)", color="#52514e")
        ax.set_ylabel(f"Count (n = {n_null} permutations)", color="#52514e")
        ax.set_title(f"{celltype.capitalize()} signature vs. randomized null" + (" (depth-residualized)" if sfx else ""),
                     fontsize=7.5, color="#0b0b0b")
        self._style(ax)
        ax.legend(frameon=False, loc="upper right", fontsize=6, labelcolor="#0b0b0b")
        ax.text(0.02, 0.95, f"empirical p = {p_value:.3g}\nz = {z_value:.1f}", transform=ax.transAxes,
                fontsize=6, va="top", color="#52514e")
        fig.tight_layout()
        self._save(fig, f"{celltype}_null_distribution_plot{sfx}")
        return fig

    def plot_combined(self, celltypes: list[str], which: str = "raw", figsize=None):
        """all_celltypes_null_distribution_plot[_resid].pdf; FDR significance from the index table."""
        sfx = "_resid" if which == "resid" else ""
        n = len(celltypes)
        fig, axes = plt.subplots(1, n, figsize=figsize or (min(2.4 * n, 7.0), 2.2), sharey=False)
        if n == 1:
            axes = [axes]
        index_path = self.outdir / MoranIndexTable.FILE
        fdr_lookup = {}
        if os.path.exists(index_path):
            idx_df = pd.read_csv(index_path)
            if f"significant_at_fdr{sfx}" in idx_df.columns and "celltype" in idx_df.columns:
                fdr_lookup = dict(zip(idx_df["celltype"], idx_df[f"significant_at_fdr{sfx}"]))
        for ax, celltype in zip(axes, celltypes):
            null_df = pd.read_csv(self.outdir / f"{celltype}_null_moran_distribution.csv")
            summary = pd.read_csv(self.outdir / f"{celltype}_null_test_summary.csv").iloc[0]
            observed_I = float(summary[f"observed_I{sfx}"])
            p_value = float(summary[f"empirical_p_value{sfx}"])
            n_null = int(summary["n_null"])
            sig = bool(fdr_lookup[celltype]) if celltype in fdr_lookup else None
            ax.hist(null_df[f"null_I{sfx}"], bins=self.bins, color="#2a78d6", edgecolor="none", zorder=2)
            ax.axvline(observed_I, color="#e34948", linestyle="--", linewidth=1.0,
                       label=f"Observed I = {observed_I:.3f}", zorder=3)
            ax.set_xlabel("Null Moran's I" + (" (depth-residualized)" if sfx else ""), color="#52514e")
            ax.set_ylabel(f"Count (n = {n_null})", color="#52514e")
            title = celltype.capitalize()
            if sig is not None:
                title += "  (FDR-significant)" if sig else "  (n.s.)"
            ax.set_title(title, fontsize=7.5, color="#0b0b0b")
            self._style(ax)
            ax.legend(frameon=False, loc="upper right", fontsize=5.5, labelcolor="#0b0b0b")
            ax.text(0.02, 0.95, f"p = {p_value:.3g}", transform=ax.transAxes, fontsize=6, va="top", color="#52514e")
        fig.tight_layout()
        self._save(fig, f"all_celltypes_null_distribution_plot{sfx}")
        return fig
