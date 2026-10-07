"""
Marker-set signature score and its spatial autocorrelation (analytic Moran's I).

  SignatureAnalysis   score -> Moran's I -> smoothed scores -> map, for one marker set

Signature score: mean log-normalized expression of the markers present, per bin.
Moran's I: whether that score is spatially clustered. Its analytic p-value is ~0 for
almost any I > 0 with ~10^5 bins, so significance comes from the random-gene-set null
test (null_test.RandomGeneSetNullTest). Gaussian smoothing (sigma 1 and 2 bins) is for
viewing and for defining the high-confidence niche.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import squidpy as sq
from anndata import AnnData

from ._logging import get_logger
from .spatial import SpatialGrid, SpatialPlotter

log = get_logger(__name__)


class SignatureAnalysis:
    """
    Spatial analysis of one curated marker-gene signature at a time.

    Writes, per cell type:
      obs[f"{celltype}_score"], obs[f"{celltype}_score_smoothed_{sigma}"] (sigma 1.0 and 2.0)
      {outdir}/{celltype}_moran_stats.csv
      spatial{celltype}_smooth.pdf       (raw and sigma-1 smoothed score)
    """

    SIGMAS = (1.0, 2.0)

    def __init__(self, outdir: str | Path, grid: SpatialGrid | None = None, plotter: SpatialPlotter | None = None,
                 layer: str = "lognorm"):
        self.outdir = Path(outdir)
        self.grid = grid or SpatialGrid()
        self.plotter = plotter or SpatialPlotter(self.outdir)
        self.layer = layer

    def score(self, adata: AnnData, genes: list[str]) -> np.ndarray:
        """Mean log-normalized expression of ``genes`` per bin (never scaled data)."""
        if len(genes) == 0:
            raise ValueError("No genes from this signature are present in adata.")
        log.debug(f"Signature uses {len(genes)} genes present in adata.")
        matrix = adata[:, genes].layers[self.layer]
        if sp.issparse(matrix):
            return np.asarray(matrix.mean(axis=1)).ravel()
        return np.asarray(matrix).mean(axis=1)

    @staticmethod
    def morans_i(adata: AnnData, score_name: str) -> pd.DataFrame:
        """Analytic Moran's I of obs[score_name] (squidpy; columns I, pval_norm, var_norm, pval_norm_fdr_bh)."""
        score_adata = sc.AnnData(X=adata.obs[[score_name]].values, obs=adata.obs.copy())
        score_adata.var_names = [score_name]
        score_adata.obsm["spatial"] = adata.obsm["spatial"].copy()
        score_adata.obsp["spatial_connectivities"] = adata.obsp["spatial_connectivities"].copy()
        score_adata.obsp["spatial_distances"] = adata.obsp["spatial_distances"].copy()
        sq.gr.spatial_autocorr(score_adata, mode="moran", genes=[score_name])
        return score_adata.uns["moranI"]

    def run(self, adata: AnnData, marker_list: list[str], celltype: str) -> pd.DataFrame:
        """Score, analytic Moran's I, smoothed scores and map for one marker set; returns the Moran's I table."""
        marker_present = [g for g in marker_list if g in adata.var_names]
        score_col = f"{celltype}_score"
        adata.obs[score_col] = self.score(adata, marker_present)

        if SpatialGrid.WEIGHT_KEY not in adata.obsp:         # built once by the pipeline; here only standalone
            self.grid.build_graph(adata)

        # squidpy row-normalizes the weights (transformation=True by default), and so does
        # null_test's Moran's I, so the I values match
        moran = self.morans_i(adata, score_col)
        moran.to_csv(self.outdir / f"{celltype}_moran_stats.csv", index=True)
        log.debug(f"\n{moran}")

        for sigma in self.SIGMAS:
            adata.obs[f"{score_col}_smoothed_{sigma}"] = SpatialGrid.smooth(adata, score_col, sigma=sigma)

        self.plotter.plot(adata, color=[score_col, f"{score_col}_smoothed_1.0"], cmap="magma", vmax="p98",
                          save=f"{celltype}_smooth")
        return moran
