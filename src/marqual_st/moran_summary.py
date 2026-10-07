"""
One table and one chart of Moran's I for every marker set of a run.

The pipeline writes, per marker set, {celltype}_moran_stats.csv (squidpy's analytic test)
and {celltype}_null_test_summary.csv (random-gene-set test, raw and depth-residualized).
MoranSummary collects them from the figures folder - no hand-typed numbers.

Tests (``test=``):
  "random_geneset_resid" (default)  observed I and empirical p after removing depth, vs.
                                    expression-matched random gene sets. The defensible choice.
  "random_geneset"                  the same test without depth removal.
  "analytic"                        squidpy's normal-approximation p (~0 for almost any I > 0
                                    with ~10^5 bins). Reference only.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from statsmodels.stats.multitest import multipletests

from ._logging import get_logger
from .null_test import MoranIndexTable
from .plot_style import save_pdf

log = get_logger(__name__)


class MoranSummary:
    """
    Collect every marker set's Moran's I results in ``figdir`` -> morans_i_summary_table.csv
    and a bar chart -> morans_i_summary_chart.pdf (in ``outdir``, default figdir).
    """

    TESTS = {
        # key: (I column, p column, label used on the chart)
        "random_geneset_resid": ("observed_I_resid", "empirical_p_value_resid", "depth-corrected, vs. random gene sets"),
        "random_geneset": ("observed_I", "empirical_p_value", "vs. random gene sets"),
        "analytic": ("analytic_I", "analytic_p", "analytic, squidpy normal approximation"),
    }
    TABLE_FILE = "morans_i_summary_table.csv"

    def __init__(self, figdir: str | Path, outdir: str | Path | None = None, test: str = "random_geneset_resid",
                 alpha: float = 0.05, use_fdr: bool = True, label_map: dict[str, str] | None = None,
                 fdr_method: str = "fdr_bh", dpi: int = 600):
        if test not in self.TESTS:
            raise ValueError(f"test must be one of {list(self.TESTS)}")
        self.figdir = Path(figdir)
        self.outdir = Path(outdir or figdir)
        self.test = test
        self.alpha = alpha
        self.use_fdr = use_fdr
        self.label_map = label_map or {}
        self.fdr_method = fdr_method
        self.dpi = dpi

    @staticmethod
    def default_label(celltype: str) -> str:
        """Replace underscores and capitalize the first letter only (gene-like names stay intact)."""
        s = celltype.replace("_", " ")
        return s[:1].upper() + s[1:]

    @staticmethod
    def format_p(p, n_null=None, prefix="p") -> str:
        """Readable p/q: empirical p-values cannot be below 1/(n_null+1); analytic p = 0 is underflow."""
        if p is None or not np.isfinite(p):
            return f"{prefix} = n/a"
        if n_null is not None and np.isfinite(n_null) and p <= 1.0 / (n_null + 1) + 1e-12:
            return f"{prefix} <= {1.0 / (n_null + 1):.1g}"
        if p == 0:
            return f"{prefix} ~ 0 (underflow)"
        return f"{prefix} = {p:.2g}" if p >= 0.001 else f"{prefix} = {p:.1e}"

    def collect(self) -> pd.DataFrame:
        """
        One row per marker set, all tests side by side, with q_<test> (BH-FDR) and, from the index
        table, qc_verdict and n_tested. When the index table exists only its marker sets (the current
        run) are used, so the FDR family - and the q-values - match it; result files of other marker
        sets left in the folder from earlier runs are ignored.
        """
        rows: dict[str, dict] = {}
        for f in sorted(glob.glob(os.path.join(self.figdir, "*_moran_stats.csv"))):
            ct = os.path.basename(f)[: -len("_moran_stats.csv")]
            d = pd.read_csv(f, index_col=0)
            if d.empty or "I" not in d:
                continue
            r = d.iloc[0]
            rows.setdefault(ct, {"celltype": ct})
            rows[ct]["analytic_I"] = float(r["I"])
            rows[ct]["analytic_p"] = float(r.get("pval_norm", np.nan))

        keep = ["n_markers_used", "n_markers_requested", "observed_I", "empirical_p_value", "z_score",
                "observed_I_resid", "empirical_p_value_resid", "z_score_resid", "depth_R2", "n_null"]
        for f in sorted(glob.glob(os.path.join(self.figdir, "*_null_test_summary.csv"))):
            ct = os.path.basename(f)[: -len("_null_test_summary.csv")]
            d = pd.read_csv(f)
            if d.empty:
                continue
            r = d.iloc[0]
            rows.setdefault(ct, {"celltype": ct})
            for k in keep:
                if k in r:
                    rows[ct][k] = r[k]
        if not rows:
            raise FileNotFoundError(f"No *_moran_stats.csv or *_null_test_summary.csv found in {self.figdir}.")

        table = pd.DataFrame(list(rows.values()))
        idx = self.figdir / MoranIndexTable.FILE
        verdicts = pd.read_csv(idx) if idx.exists() else None
        if verdicts is not None and "celltype" in verdicts:
            extra = sorted(set(table["celltype"]) - set(verdicts["celltype"]))
            if extra:
                log.debug(f"Ignoring result files of {extra} - not in this run's index table ({idx}).")
            table = table[table["celltype"].isin(verdicts["celltype"])].reset_index(drop=True)

        table["label"] = [self.label_map.get(c, self.default_label(c)) for c in table["celltype"]]
        for test, (_, pcol, _) in self.TESTS.items():
            if pcol in table:
                p = table[pcol].astype(float)
                q = np.full(len(p), np.nan)
                ok = p.notna().values
                if ok.sum() > 0:
                    q[ok] = multipletests(p[ok].values, alpha=self.alpha, method=self.fdr_method)[1]
                table[f"q_{test}"] = q
        if verdicts is not None and {"celltype", "qc_verdict"} <= set(verdicts.columns):
            cols = ["celltype", "qc_verdict"] + [c for c in ("n_tested",) if c in verdicts.columns]
            table = table.merge(verdicts[cols], on="celltype", how="left")
        front = ["celltype", "label"]
        return table[front + [c for c in table.columns if c not in front]]

    def plot(self, table: pd.DataFrame, test: str | None = None, fname: str = "morans_i_summary_chart",
             title: str = "Spatial autocorrelation across cell-type signatures"):
        """
        Horizontal bar chart (largest on top), colored by significance (q, or p with use_fdr=False),
        each bar labeled with its I and q. Bars are drawn one row at a time, so every label belongs to
        its own bar. Saves {outdir}/{fname}.pdf (publication-quality, plot_style).
        """
        test = test or self.test
        icol, pcol, test_label = self.TESTS[test]
        if icol not in table or pcol not in table:
            raise ValueError(f"Columns {icol}/{pcol} not in the table - were those files in the folder?")
        df = table.dropna(subset=[icol]).copy()
        df["_p"] = df[pcol].astype(float)
        df["_q"] = df[f"q_{test}"].astype(float) if f"q_{test}" in df else df["_p"]
        df["_sig"] = df["_q" if self.use_fdr else "_p"] < self.alpha
        df = df.sort_values(icol, ascending=True)          # barh draws bottom-up -> largest on top

        sig_name = f"{'FDR q' if self.use_fdr else 'p'} < {self.alpha}"
        colors = np.where(df["_sig"], "#2a78d6", "#b4b3ad")
        n_null = df["n_null"] if (test != "analytic" and "n_null" in df) else pd.Series(np.nan, index=df.index)

        fig, ax = plt.subplots(figsize=(4.6, 0.24 * len(df) + 1.0))
        y = np.arange(len(df))
        ax.barh(y, df[icol], color=colors, height=0.7, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(df["label"], fontsize=6.5)
        ax.axvline(0, color="#52514e", linewidth=0.6, zorder=3)
        vmax, vmin = float(df[icol].max()), float(min(df[icol].min(), 0))
        span = vmax - vmin if vmax > vmin else abs(vmax) or 1.0
        ax.set_xlim(vmin - 0.02 * span, vmax + 0.45 * span)   # room for the labels
        for yy, (_, r) in zip(y, df.iterrows()):
            txt = f"I = {r[icol]:.4f}   " + self.format_p(r["_q"] if self.use_fdr else r["_p"],
                                                            n_null.loc[r.name], "q" if self.use_fdr else "p")
            ax.text(max(r[icol], 0) + 0.015 * span, yy, txt, va="center", ha="left", fontsize=5.5, color="#0b0b0b")
        ax.set_xlabel(f"Moran's I ({test_label})", fontsize=7, color="#52514e")
        ax.set_ylabel("Cell-type signature", fontsize=7, color="#52514e")
        ax.set_title(title, fontsize=8, fontweight="bold", pad=6)
        ax.legend(handles=[Patch(color="#2a78d6", label=f"Significant ({sig_name})"),
                           Patch(color="#b4b3ad", label=f"Not significant ({sig_name.replace('<', '>=')})")],
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=6)
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.4, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        save_pdf(fig, self.outdir / f"{fname}.pdf", dpi=self.dpi)
        plt.close(fig)
        log.debug(f"Saved {self.outdir / fname}.pdf ({len(df)} cell types; test = {test}; "
                 f"significance by {'FDR q' if self.use_fdr else 'raw p'})")
        return fig

    def run(self, plot: bool = True) -> pd.DataFrame:
        """Collect, write the table, draw the chart (analytic test as fallback); returns the table."""
        table = self.collect()
        path = self.outdir / self.TABLE_FILE
        table.to_csv(path, index=False)
        log.debug(f"Collected Moran's I for {len(table)} cell types -> {path}")
        test = self.test
        icol, pcol, _ = self.TESTS[test]
        if plot and (icol not in table or pcol not in table or table[pcol].isna().all()):
            log.info(f"No {test} results in {self.figdir} - falling back to the analytic squidpy test.")
            test = "analytic"
        if plot:
            self.plot(table, test=test,
                      fname="morans_i_summary_chart" + ("" if test == "random_geneset_resid" else f"_{test}"))
        return table
