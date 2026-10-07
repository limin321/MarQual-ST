"""
Technical (marker-independent) spatial QC for binned spatial data.

Standard QC summarizes each bin on its own (total counts, genes, mito %). TechnicalQC
adds the SPATIAL view, independently of any marker panel:

  - depth maps (total counts, genes per bin) and a high-pass depth map, for judging
    technical patterns by eye - e.g. imaging field-of-view (FOV) rectangles;
  - tissue sections (num_tissue > 1) and strays, found on the UNFILTERED bins so filter
    holes cannot split a section; strays are LABELED only (obs["stray"]);
  - tissue section as a technical source: with num_tissue > 1, section depth is compared
    and the pipeline runs a "within section" co-localization control (tissue_qc).

FOV artifacts are not modelled separately: they change total counts per bin, and every
downstream test adjusts for sequencing depth. The module never computes depth itself;
it reads scanpy's QC columns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse.csgraph import connected_components

from ._logging import get_logger
from .depth import DepthModel
from .spatial import SpatialGrid, SpatialPlotter

log = get_logger(__name__)


class TechnicalQC:
    """
    Two calls, in this order:
      detect_sections(adata)  UNFILTERED data: strays + sections (labels carried to filtered bins)
      run(adata)              FILTERED data: section depth summary, depth maps, technical_qc_summary.csv
    """

    QC_COLUMNS = ("total_counts", "n_genes_by_counts", "log1p_total_counts", "log1p_n_genes_by_counts")
    SECTION_KEY = "tissue_section"
    STRAY_KEY = "stray"

    def __init__(self, outdir: str | Path, num_tissue: int = 1, tissue_qc: bool | None = None,
                 sigma_trend: float = 10.0, plotter: SpatialPlotter | None = None, plot: bool = True,
                 weight_key: str = SpatialGrid.WEIGHT_KEY):
        if int(num_tissue) < 1:
            raise ValueError("num_tissue must be >= 1.")
        self.outdir = Path(outdir)
        self.num_tissue = int(num_tissue)
        self.tissue_qc = tissue_qc
        self.sigma_trend = sigma_trend
        self.plotter = plotter or SpatialPlotter(self.outdir)
        self.plot = plot
        self.weight_key = weight_key

    # ------------------------------------------------------------------ sections
    def _tissue_pieces(self, adata):
        """Connected pieces of tissue in the spatial graph: (component per bin, size per component, ids largest first)."""
        if self.weight_key not in adata.obsp:
            raise ValueError(f"{self.weight_key!r} not in adata.obsp - run SpatialGrid.build_graph(adata) first.")
        _, comp = connected_components(adata.obsp[self.weight_key], directed=False)
        sizes = np.bincount(comp)
        return comp, sizes, np.argsort(-sizes)

    def label_strays(self, adata: AnnData) -> dict:
        """
        LABEL ONLY: obs["stray"] = True for every bin outside the num_tissue largest connected pieces.
        Strays are pieces already separate before filtering (debris, small detached tissue); they are
        not removed and not used by any test. In simulation, removing them changed Moran's I by <= 3%
        with no change of any verdict. Writes stray_pieces.csv.
        """
        comp, sizes, order = self._tissue_pieces(adata)
        stray = order[self.num_tissue:]
        adata.obs[self.STRAY_KEY] = ~np.isin(comp, order[:self.num_tissue])
        n_bins = int(adata.obs[self.STRAY_KEY].sum())
        pd.DataFrame({"piece_rank": np.arange(self.num_tissue + 1, self.num_tissue + 1 + len(stray)),
                      "n_bins": sizes[stray].astype(int)}).to_csv(self.outdir / "stray_pieces.csv", index=False)
        log.info(f"Strays (label only, obs[{self.STRAY_KEY!r}]): {len(stray)} piece(s) outside the {self.num_tissue} "
                 f"largest, {n_bins} bins ({n_bins / adata.n_obs:.1%}).")
        if n_bins > 0.2 * adata.n_obs:
            log.warning(f"{n_bins / adata.n_obs:.0%} of bins are strays - the chip probably carries more than "
                        f"num_tissue = {self.num_tissue} section(s); check num_tissue.")
        return {"n_stray_pieces": int(len(stray)), "n_stray_bins": n_bins}

    def label_sections(self, adata: AnnData) -> str:
        """
        Only for chips carrying MORE THAN ONE section: the num_tissue largest connected pieces are the
        sections (S1 = largest) -> obs["tissue_section"]; stray pieces get no label. Used as a technical
        variable only with tissue_qc (sections may be biologically different).
        """
        comp, sizes, order = self._tissue_pieces(adata)
        n_comp, n_sections = len(sizes), self.num_tissue
        if n_comp < n_sections:
            log.warning(f"num_tissue = {n_sections} but only {n_comp} separate piece(s) of tissue were found - "
                        "sections touch each other on the chip, so they cannot be separated by position.")
        names = {c: f"S{i + 1}" for i, c in enumerate(order[:n_sections])}
        labels = pd.Series([names.get(c) for c in comp], index=adata.obs_names, dtype="object")
        adata.obs[self.SECTION_KEY] = pd.Categorical(labels, categories=[f"S{i + 1}" for i in range(min(n_sections, n_comp))])
        kept = [int(sizes[order[i]]) for i in range(min(n_sections, n_comp))]
        log.info(f"Tissue sections: the {len(kept)} largest piece(s) labeled as sections "
                 f"({', '.join(f'S{i + 1}={k}' for i, k in enumerate(kept))} bins); stray pieces left unlabeled.")
        if len(kept) > 1 and kept[-1] < 0.05 * kept[0]:
            log.warning(f"section S{len(kept)} has only {kept[-1]} bins (< 5% of S1) - it is probably a stray "
                        f"piece, not a section; check num_tissue (= {n_sections}).")
        elif n_comp > n_sections and kept:
            nxt = int(sizes[order[n_sections]])
            if nxt > 0.2 * min(kept):
                log.warning(f"the largest stray piece ({nxt} bins) is not much smaller than the smallest "
                            f"section ({min(kept)} bins) - check num_tissue (a section may be split into pieces).")
        return self.SECTION_KEY

    def detect_sections(self, adata: AnnData) -> dict:
        """
        UNFILTERED data, after the grid and spatial graph and BEFORE filtering (bin filtering punches
        holes into the tissue and would split sections). Returns {"n_stray_pieces", "n_stray_bins"}.
        """
        stray = self.label_strays(adata)
        if self.num_tissue > 1:
            self.label_sections(adata)
        elif self.SECTION_KEY in adata.obs:
            del adata.obs[self.SECTION_KEY]                    # stale label from an earlier run
        adata.uns["technical_qc"] = {"num_tissue": self.num_tissue, **stray}
        return stray

    def section_depth_summary(self, adata: AnnData, score_cols=None) -> pd.DataFrame:
        """
        Depth per section (strays excluded): bins, median / IQR total counts, median genes, median depth
        relative to the whole chip (+ optional mean of score columns). A much shallower section makes every
        marker look weaker there - a technical difference, not biology. Writes section_depth_summary.csv.
        """
        DepthModel.require_qc_metrics(adata, ("total_counts", "n_genes_by_counts"))
        overall = float(np.median(adata.obs["total_counts"]))
        rows = []
        for sec, df in adata.obs.groupby(self.SECTION_KEY, observed=True):
            tc = df["total_counts"].astype(float)
            row = {
                "section": sec,
                "n_bins": len(df),
                "median_total_counts": float(tc.median()),
                "q25_total_counts": float(tc.quantile(0.25)),
                "q75_total_counts": float(tc.quantile(0.75)),
                "median_n_genes": float(df["n_genes_by_counts"].median()),
                "median_depth_vs_chip": float(tc.median() / overall) if overall > 0 else np.nan,
            }
            for c in (score_cols or []):
                if c in adata.obs:
                    row[f"mean_{c}"] = float(adata.obs.loc[df.index, c].mean())
            rows.append(row)
        out = pd.DataFrame(rows).sort_values("n_bins", ascending=False)
        out.to_csv(self.outdir / "section_depth_summary.csv", index=False)
        log.debug(f"\n{out.to_string(index=False)}")
        return out

    # ------------------------------------------------------------------ maps
    def plot_depth_maps(self, adata: AnnData) -> None:
        """Maps of log1p total counts and genes per bin (1st-99th percentile) -> spatial_qc_depth_maps.*"""
        DepthModel.require_qc_metrics(adata, ("log1p_total_counts", "log1p_n_genes_by_counts"))
        self.plotter.plot(adata, ["log1p_total_counts", "log1p_n_genes_by_counts"], cmap="viridis",
                          vmin="p1", vmax="p99", frameon=False, save="_qc_depth_maps")

    def highpass_depth_map(self, adata: AnnData, key: str = "log1p_total_counts_highpass") -> str:
        """
        log1p(total counts) minus its Gaussian-smoothed regional trend (sigma_trend bins): local depth
        changes, e.g. the straight edges of an FOV rectangle, stand out. Stored in obs[key].
        -> spatial_qc_depth_highpass.*
        """
        DepthModel.require_qc_metrics(adata, ("log1p_total_counts",))
        trend = SpatialGrid.smooth(adata, "log1p_total_counts", sigma=self.sigma_trend)
        adata.obs[key] = adata.obs["log1p_total_counts"].values - trend
        if self.plot:
            lim = float(np.quantile(np.abs(adata.obs[key]), 0.99))
            self.plotter.plot(adata, [key], cmap="RdBu_r", vmin=-lim, vmax=lim, frameon=False,
                              title=f"log1p total counts minus regional trend (sigma = {self.sigma_trend:g} bins)",
                              save="_qc_depth_highpass")
        return key

    # ------------------------------------------------------------------ filtered data
    def run(self, adata: AnnData, score_cols=None) -> dict:
        """
        FILTERED data (after Preprocessor), using the labels from detect_sections. Writes
        technical_qc_summary.csv (+ filter_summary.csv) and returns {"summary", "sections"}.
        """
        DepthModel.require_qc_metrics(adata, self.QC_COLUMNS)
        info = adata.uns.get("technical_qc")
        if info is None or self.STRAY_KEY not in adata.obs:
            raise ValueError("No section labels - run TechnicalQC.detect_sections on the UNFILTERED data first.")
        num_tissue = int(info["num_tissue"])
        tissue_qc = self.tissue_qc
        if tissue_qc is None:
            tissue_qc = num_tissue > 1
        elif tissue_qc and num_tissue == 1:
            log.info("tissue_qc=True has no effect with num_tissue=1 (one section - nothing to compare).")
        tissue_qc = bool(tissue_qc and num_tissue > 1)

        sections = None
        if num_tissue == 1:
            log.info("num_tissue = 1: single tissue section - section step skipped.")
        elif tissue_qc:
            sections = self.section_depth_summary(adata, score_cols=score_cols)
        else:
            log.info(f"num_tissue = {num_tissue}, tissue_qc = False: obs['tissue_section'] kept for reference only.")

        if self.plot:
            self.plot_depth_maps(adata)
        self.highpass_depth_map(adata)

        adata.uns["technical_qc"] = {**info, "tissue_qc": tissue_qc}
        fs = dict(adata.uns.get("filter_summary", {}))
        if fs:
            pd.DataFrame([fs]).to_csv(self.outdir / "filter_summary.csv", index=False)
        summary = {
            "n_bins": adata.n_obs,
            **({"n_bins_unfiltered": fs["n_bins_unfiltered"],
                "pct_bins_filtered_out": fs["pct_bins_filtered_out"]} if fs else {}),
            **({"data_quality": fs["data_quality"]} if "data_quality" in fs else {}),
            "median_total_counts": float(np.median(adata.obs["total_counts"])),
            "median_n_genes": float(np.median(adata.obs["n_genes_by_counts"])),
            "num_tissue": num_tissue,
            "tissue_qc": tissue_qc,
            "n_stray_pieces_unfiltered": info["n_stray_pieces"],
            "n_stray_bins_unfiltered": info["n_stray_bins"],
            "n_stray_bins_after_filter": int(adata.obs[self.STRAY_KEY].sum()),
        }
        if sections is not None and len(sections) > 1:
            summary["section_depth_max_min_ratio"] = float(sections["median_total_counts"].max() /
                                                           sections["median_total_counts"].min())
        pd.DataFrame([summary]).to_csv(self.outdir / "technical_qc_summary.csv", index=False)
        log.debug(f"\n{pd.Series(summary).to_string()}")
        return {"summary": summary, "sections": sections}
