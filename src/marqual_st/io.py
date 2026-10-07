"""
Data input / output: platform-specific loading, platform-neutral filtering, and the
analysis h5ad hand-off to the optional gene-pair step.

  DataLoader (abstract)   -> StereoSeqLoader, VisiumHDLoader   (unfiltered AnnData)
  Preprocessor            filter bins / genes, keep raw counts, log-normalize
  AnalysisStore           save / load the analysis h5ad

Every loader returns the same thing: raw counts in .X, gene symbols as var_names,
var["mt"] / var["ribo"], obsm["spatial"], and scanpy's QC columns in obs
(total_counts, n_genes_by_counts, pct_counts_mt, log1p_..., computed ONCE here on
the unfiltered counts - the rest of the pipeline never computes depth itself).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from ._logging import get_logger

log = get_logger(__name__)


class DataLoader(ABC):
    """Base class: read one platform's output into an unfiltered AnnData with QC metrics."""

    MT_PREFIX = ("MT-", "MTCO", "mt-")
    RIBO_PREFIX = ("RPS", "RPL", "rps", "rpl")

    #: registry of platform name -> loader class (filled by subclasses)
    registry: dict[str, type[DataLoader]] = {}

    def __init_subclass__(cls, platform: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if platform:
            DataLoader.registry[platform] = cls

    @classmethod
    def for_platform(cls, platform: str) -> DataLoader:
        """Loader instance for a platform name ("stereo", "visiumhd")."""
        try:
            return cls.registry[platform]()
        except KeyError:
            raise ValueError(f"Unknown platform {platform!r}; available: {sorted(cls.registry)}.") from None

    @abstractmethod
    def read(self, path: str | Path) -> AnnData:
        """Platform-specific reading (raw counts in .X, obsm['spatial'])."""

    def load(self, path: str | Path) -> AnnData:
        """Read ``path`` and add gene flags and scanpy QC metrics."""
        return self.add_qc_metrics(self.read(path))

    def add_qc_metrics(self, adata: AnnData) -> AnnData:
        """var["mt"] / var["ribo"] from gene symbols (var_names), then scanpy's QC metrics on raw counts in .X."""
        names = adata.var_names.to_series().astype(str)
        adata.var["mt"] = names.str.startswith(self.MT_PREFIX).values
        adata.var["ribo"] = names.str.startswith(self.RIBO_PREFIX).values
        top = [t for t in (50, 100, 200, 500) if t < adata.n_vars]      # scanpy default, capped for small panels
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], percent_top=top or None, inplace=True)
        log.info(f"Loaded {adata.n_obs} bins x {adata.n_vars} genes (unfiltered); "
                 f"mt genes: {int(adata.var['mt'].sum())}, ribo genes: {int(adata.var['ribo'].sum())}; "
                 f"median total counts {np.median(adata.obs['total_counts']):.0f}.")
        return adata


class StereoSeqLoader(DataLoader, platform="stereo"):
    """
    Stereo-seq: the tissue-cut, UNFILTERED h5ad (raw counts in .X, obsm["spatial"]).
    Gene symbols from var["real_gene_name"] become var_names; the original IDs are
    kept in var["ensembl_id"].
    """

    def read(self, path):
        adata = sc.read_h5ad(path)
        adata.var["ensembl_id"] = adata.var_names
        adata.var.index = adata.var["real_gene_name"].astype(str)
        adata.var_names_make_unique()
        # The index is now named "real_gene_name", but after var_names_make_unique its values
        # differ from that column (GENE-1, ...), and write_h5ad refuses such a DataFrame.
        adata.var.index.name = None
        return adata


class VisiumHDLoader(DataLoader, platform="visiumhd"):
    """
    Visium HD: one Space Ranger bin size, e.g. ``<spaceranger out>/binned_outputs/square_016um``.
    Uses filtered_feature_bc_matrix.h5 (bins under tissue = tissue-cut, not count-filtered)
    and spatial/tissue_positions.parquet:
        obs["array_row"], obs["array_col"]  -> grid positions (SpatialGrid)
        obsm["spatial"]                     -> full-resolution pixel (x, y)
    Not yet tested on real Space Ranger output.
    """

    def read(self, path):
        adata = sc.read_10x_h5(os.path.join(path, "filtered_feature_bc_matrix.h5"))
        adata.var_names_make_unique()               # gene symbols; Ensembl IDs are in var["gene_ids"]
        pos_file = os.path.join(path, "spatial", "tissue_positions.parquet")
        pos = pd.read_parquet(pos_file).set_index("barcode")
        missing = adata.obs_names.difference(pos.index)
        if len(missing):
            raise ValueError(f"{len(missing)} barcodes have no position in {pos_file}.")
        pos = pos.loc[adata.obs_names]
        adata.obs["array_row"] = pos["array_row"].astype(int).values
        adata.obs["array_col"] = pos["array_col"].astype(int).values
        adata.obsm["spatial"] = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)
        return adata


class Preprocessor:
    """
    Both platforms. Remove bins with total counts < ``min_counts`` or mito % >= ``pct_mt``,
    and genes detected in < ``min_cells`` bins. Raw counts -> layers["counts"];
    normalize_total + log1p -> .X and layers["lognorm"].
    """

    def __init__(self, min_counts: int = 600, min_cells: int = 6, pct_mt: float = 20):
        self.min_counts = min_counts
        self.min_cells = min_cells
        self.pct_mt = pct_mt

    def filter_stats(self, adata: AnnData, keep: np.ndarray) -> dict:
        """How many bins the filter removes, and why (kept in uns["filter_summary"])."""
        tc = np.asarray(adata.obs["total_counts"].values, dtype=float)
        mt = np.asarray(adata.obs["pct_counts_mt"].values, dtype=float)
        fail_depth = tc < self.min_counts
        fail_mt = ~(mt < self.pct_mt)                 # NaN mito % (empty bins) counts as failing
        n, n_kept = int(adata.n_obs), int(keep.sum())
        return {
            "n_bins_unfiltered": n,
            "n_bins_kept": n_kept,
            "n_bins_filtered_out": n - n_kept,
            "pct_bins_filtered_out": 100.0 * (n - n_kept) / n if n else float("nan"),
            "n_fail_min_counts": int(fail_depth.sum()),
            "n_fail_pct_mt": int(fail_mt.sum()),
            "n_fail_both": int((fail_depth & fail_mt).sum()),
            "median_total_counts_unfiltered": float(np.median(tc)) if n else float("nan"),
            "n_genes_unfiltered": int(adata.n_vars),
            "min_counts": self.min_counts,
            "pct_mt": self.pct_mt,
            "min_cells": self.min_cells,
        }

    def check_kept_bins(self, adata: AnnData, keep: np.ndarray) -> None:
        """Stop with a diagnosis when the filter would remove every bin (scanpy would otherwise fail
        later with "Found array with 0 sample(s)")."""
        if keep.any():
            return
        tc = np.asarray(adata.obs["total_counts"].values, dtype=float)
        mt = np.asarray(adata.obs["pct_counts_mt"].values, dtype=float)
        x = adata.X[: min(adata.n_obs, 2000)]
        vals = np.asarray(x.data if hasattr(x, "data") else x).ravel()
        integer = bool(vals.size == 0 or np.allclose(vals, np.round(vals)))
        q = np.nanpercentile(tc, [50, 95, 100]) if len(tc) else [np.nan] * 3
        hints = []
        if not integer:
            hints.append("adata.X is not integer counts - the input looks normalized/log-transformed; "
                         "use the RAW count h5ad")
        if (tc >= self.min_counts).sum() == 0:
            hints.append(f"no bin reaches filters.min_counts = {self.min_counts} (median total counts {q[0]:.0f}): data "
                         f"quality too low to continue - do not lower the cutoff; if this is not bin50 data, check the input")
        if np.isnan(mt).all() or (mt >= self.pct_mt).all():
            hints.append(f"no bin has mito % < filters.pct_mt = {self.pct_mt}: data quality too low to continue "
                         f"(degraded tissue?)")
        raise DataQualityStop(
            f"DATA QUALITY STOP - the bin filter removes ALL {adata.n_obs} bins (total counts >= {self.min_counts}: "
            f"{int((tc >= self.min_counts).sum())} bins; mito % < {self.pct_mt}: {int((mt < self.pct_mt).sum())} bins). "
            f"total_counts median / 95th pct / max = {q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f}; "
            f"X integer counts: {integer}. " + "; ".join(hints) + ". (The data-quality gate reports this as STOP, "
            "with technical QC and a report, before this point.)")

    def filter_and_normalize(self, adata: AnnData) -> AnnData:
        """Returns a NEW AnnData; obs labels (e.g. tissue_section, stray) and the spatial
        graph in obsp are carried over for the bins that stay."""
        n0, g0 = adata.n_obs, adata.n_vars
        keep = (adata.obs["total_counts"].values >= self.min_counts) & (adata.obs["pct_counts_mt"].values < self.pct_mt)
        stats = self.filter_stats(adata, keep)
        self.check_kept_bins(adata, keep)
        # Take the bin x bin graphs (obsp) out before subsetting and slice them rows-then-columns:
        # anndata indexes rows and columns in ONE step, which on ~10^5 unfiltered bins exceeds
        # scipy's 32-bit index range ("ValueError: could not convert integer scalar").
        graphs = {k: adata.obsp[k] for k in list(adata.obsp.keys())}
        for k in graphs:
            del adata.obsp[k]
        adata = adata[keep].copy()
        idx = np.flatnonzero(keep)
        for k, g in graphs.items():
            adata.obsp[k] = g.tocsr()[idx][:, idx]
        sc.pp.filter_genes(adata, min_cells=self.min_cells)
        if adata.n_vars == 0:
            raise ValueError(f"No gene is detected in >= {self.min_cells} of the {adata.n_obs} kept bins - "
                             "lower filters.min_cells.")
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        adata.layers["lognorm"] = adata.X.copy()
        stats["n_genes_kept"] = int(adata.n_vars)
        # + the verdict of DataQualityGate.assess, if it ran; written to filter_summary.csv by TechnicalQC.run
        adata.uns["filter_summary"] = {**adata.uns.get("filter_summary", {}), **stats}
        log.debug(f"Bins filtered out: {stats['n_bins_filtered_out']}/{stats['n_bins_unfiltered']} "
                 f"({stats['pct_bins_filtered_out']:.1f}%) - failing total counts: {stats['n_fail_min_counts']}, "
                 f"failing mito %: {stats['n_fail_pct_mt']} (both: {stats['n_fail_both']}).")
        log.info(f"Filtered (total counts >= {self.min_counts}, mito % < {self.pct_mt}, genes in >= "
                 f"{self.min_cells} bins): {adata.n_obs}/{n0} bins, {adata.n_vars}/{g0} genes kept.")
        return adata


class DataQualityStop(Exception):
    """Raised when the data-quality verdict is STOP (and force is off). It is not a code error: the
    pipeline stops on purpose, after technical QC and the report. Raising stops the rest of the
    script / notebook ("Run All"); in Jupyter it is shown as a short message, not a traceback."""

    def _render_traceback_(self):          # IPython / Jupyter: show this instead of a traceback
        import textwrap
        bar = "=" * 78
        lines = [bar, "PIPELINE STOPPED: DATA QUALITY TOO LOW (planned stop, not a code error)", bar]
        for part in str(self).split(" - ", 1):
            lines += textwrap.wrap(part, 78) or [""]
        return lines + [bar]


class DataQualityGate:
    """
    Data-quality gate on the UNFILTERED bins, before filtering: the share of bins the standard
    filter (``Preprocessor``; Stereo-seq bin50: total counts >= 600, mito % < 20) would remove.

      STOP     >= stop_pct_filtered % filtered out, or fewer than min_bins_kept bins kept - the
               bins left are the best-captured patches, and results from them would not describe
               the sample
      CAUTION  >= caution_pct_filtered % - continue; results may cover only the best-captured regions
      PASS     otherwise

    ``force`` continues after STOP (exploration only); the verdict stays STOP in every output.
    """

    PASS, CAUTION, STOP = "PASS", "CAUTION", "STOP"

    def __init__(self, preprocessor: Preprocessor, caution_pct_filtered: float = 60, stop_pct_filtered: float = 80,
                 min_bins_kept: int = 2000, force: bool = False):
        self.preprocessor = preprocessor
        self.caution_pct_filtered = caution_pct_filtered
        self.stop_pct_filtered = stop_pct_filtered
        self.min_bins_kept = min_bins_kept
        self.force = force

    def verdict(self, pct_filtered: float, n_kept: int) -> tuple[str, str]:
        if pct_filtered >= self.stop_pct_filtered or n_kept < self.min_bins_kept:
            why = [f"{pct_filtered:.1f}% of bins filtered out (STOP at >= {self.stop_pct_filtered:g}%)"
                   if pct_filtered >= self.stop_pct_filtered else "",
                   f"only {n_kept} bins kept (STOP below {self.min_bins_kept})" if n_kept < self.min_bins_kept else ""]
            return self.STOP, "; ".join(w for w in why if w) + " - data quality too low to continue the analysis."
        if pct_filtered >= self.caution_pct_filtered:
            return self.CAUTION, (f"{pct_filtered:.1f}% of bins filtered out (CAUTION at >= {self.caution_pct_filtered:g}%) "
                                  f"- results may describe only the best-captured regions of the tissue.")
        return self.PASS, f"{pct_filtered:.1f}% of bins filtered out (CAUTION at >= {self.caution_pct_filtered:g}%)."

    def assess(self, adata: AnnData) -> dict:
        """Numbers + verdict -> adata.uns["filter_summary"] (-> filter_summary.csv); returns them."""
        pp = self.preprocessor
        keep = (adata.obs["total_counts"].values >= pp.min_counts) & (adata.obs["pct_counts_mt"].values < pp.pct_mt)
        stats = pp.filter_stats(adata, keep)
        verdict, reason = self.verdict(stats["pct_bins_filtered_out"], stats["n_bins_kept"])
        if stats["n_bins_filtered_out"]:
            share = 100.0 * stats["n_fail_min_counts"] / stats["n_bins_filtered_out"]
            reason += f" {share:.0f}% of the removed bins fail total counts >= {pp.min_counts}"
            reason += (" (low capture / shallow sequencing)." if share >= 50
                       else f"; most fail mito % < {pp.pct_mt} (degraded tissue?).")
        stats.update({"data_quality": verdict, "data_quality_reason": reason,
                      "forced": bool(self.force and verdict == self.STOP),
                      "caution_pct_filtered": self.caution_pct_filtered, "stop_pct_filtered": self.stop_pct_filtered,
                      "min_bins_kept": self.min_bins_kept})
        adata.uns["filter_summary"] = stats
        msg = f"DATA QUALITY: {verdict} - {reason}" + (" Continuing anyway (force=True)." if stats["forced"] else "")
        (log.warning if verdict != self.PASS else log.info)(msg)
        return stats

    def should_stop(self, stats: dict) -> bool:
        return stats["data_quality"] == self.STOP and not self.force


class AnalysisStore:
    """
    Save / load the pipeline's working AnnData (the hand-off to the gene-pair step):
    layers (counts, lognorm), obsm["spatial"], the spatial graph (obsp), grid positions,
    all score / mask / section / annotation columns in obs, and uns (DE results).
    """

    REQUIRED_LAYERS = ("counts", "lognorm")

    def __init__(self, path: str | Path, compression: str = "gzip"):
        self.path = Path(path)
        self.compression = compression

    def save(self, adata: AnnData) -> Path:
        """Make the object writable (obs columns of mixed Python objects become strings) and write it."""
        for c in adata.obs.columns:
            if adata.obs[c].dtype == object:
                adata.obs[c] = adata.obs[c].astype(str)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(self.path, compression=self.compression)
        log.info(f"Saved analysis h5ad: {self.path} ({os.path.getsize(self.path) / 1e6:.0f} MB; "
                 f"{adata.n_obs} bins x {adata.n_vars} genes; obs columns: {adata.obs.shape[1]})")
        return self.path

    def load(self) -> AnnData:
        """Load and check the layers; rebuild the spatial graph if it is missing."""
        if not self.path.is_file():
            raise FileNotFoundError(f"Analysis h5ad not found: {self.path} - run the main pipeline first.")
        ad = sc.read_h5ad(self.path)
        missing = [layer for layer in self.REQUIRED_LAYERS if layer not in ad.layers]
        if missing:
            raise ValueError(f"{self.path} is missing layers {missing}.")
        if "spatial_connectivities" not in ad.obsp:
            from .spatial import SpatialGrid
            log.debug("Spatial graph not found in the h5ad - rebuilding it.")
            SpatialGrid().build_graph(ad)
        log.info(f"Loaded {self.path}: {ad.n_obs} bins x {ad.n_vars} genes.")
        return ad
