"""
Spatial basics shared by every step.

  SpatialGrid     integer grid position of every bin, the 8-neighbour spatial graph,
                  and normalized-convolution Gaussian smoothing on the grid
  SpatialPlotter  spatial maps with the y-axis flipped at draw time, square markers,
                  PDF + PNG output
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import squidpy as sq
from anndata import AnnData
from scipy.ndimage import gaussian_filter

from ._logging import get_logger
from .plot_style import save_pdf

log = get_logger(__name__)


class SpatialGrid:
    """
    Bin grid of a binned spatial dataset.

    ``bin_size``: distance between neighbouring bins in ``obsm["spatial"]`` units
    (Stereo-seq bin50: 50); not needed when obs has ``array_row`` / ``array_col``
    (Visium HD). ``radius``: neighbour radius in bins (1.5 = the 8 surrounding bins).
    """

    ROW_COL = ("array_row", "array_col")
    WEIGHT_KEY = "spatial_connectivities"

    def __init__(self, bin_size: float | None = None, radius: float = 1.5):
        self.bin_size = bin_size
        self.radius = radius

    def add_positions(self, adata: AnnData) -> None:
        """
        Store every bin's integer grid position ONCE; all modules read it from here.
          adata.obs["grid_col"], adata.obs["grid_row"]   (0-based column / row)
          adata.obsm["grid"]                              (same, as an n x 2 array)
        From obs[array_row/array_col] when present, else round((coordinate - min) / bin_size).
        """
        r, c = self.ROW_COL
        if r in adata.obs and c in adata.obs:
            col = np.asarray(adata.obs[c], dtype=int)
            row = np.asarray(adata.obs[r], dtype=int)
            source = f"obs[{c!r}], obs[{r!r}]"
        else:
            if self.bin_size is None:
                raise ValueError(f"No {r}/{c} in adata.obs - set bin_size, the distance between neighboring bins in "
                                 "obsm['spatial'] units (Stereo-seq bin50: 50).")
            xy = np.asarray(adata.obsm["spatial"], dtype=float)
            col = np.round((xy[:, 0] - xy[:, 0].min()) / self.bin_size).astype(int)
            row = np.round((xy[:, 1] - xy[:, 1].min()) / self.bin_size).astype(int)
            source = f"obsm['spatial'] / bin_size {self.bin_size}"
        col, row = col - col.min(), row - row.min()
        adata.obs["grid_col"], adata.obs["grid_row"] = col, row
        adata.obsm["grid"] = np.c_[col, row].astype(float)
        n_shared = len(col) - len(set(zip(col, row)))
        log.debug(f"Grid positions from {source}: {col.max() + 1} columns x {row.max() + 1} rows"
                 + (f"; WARNING: {n_shared} bins share a grid position - check bin_size" if n_shared else ""))

    @staticmethod
    def positions(adata: AnnData) -> tuple[np.ndarray, np.ndarray]:
        """(column, row) integer arrays stored by :meth:`add_positions`."""
        if "grid_col" not in adata.obs or "grid_row" not in adata.obs:
            raise ValueError("Grid positions missing - run SpatialGrid(bin_size=...).add_positions(adata) first.")
        return np.asarray(adata.obs["grid_col"], dtype=int), np.asarray(adata.obs["grid_row"], dtype=int)

    def build_graph(self, adata: AnnData) -> None:
        """
        The spatial neighbor graph, built ONCE for the whole pipeline (Moran's I, null test,
        bivariate Moran's I all reuse adata.obsp["spatial_connectivities"]): grid positions
        with a radius of 1.5 bins = the 8 surrounding bins, the same on every platform.
        """
        self.positions(adata)                     # clear error if add_positions was not run
        sq.gr.spatial_neighbors(adata, spatial_key="grid", coord_type="generic", radius=self.radius)
        n_neigh = np.asarray(adata.obsp[self.WEIGHT_KEY].sum(axis=1)).ravel()
        log.debug(f"Spatial graph: radius {self.radius} bins; neighbors per bin median = {np.median(n_neigh):.0f}, "
                 f"bins with 0 neighbors = {int((n_neigh == 0).sum())}")

    @classmethod
    def smooth(cls, adata: AnnData, column: str, sigma: float = 2.0) -> np.ndarray:
        """
        Smooth obs[column] on the bin grid (visualization / niche definition, NOT significance).
        sigma is in bins. Normalized convolution (smooth(score) / smooth(occupancy)): empty grid
        cells outside the tissue are not treated as zeros, so edge bins are not pulled down.
        """
        x_indices, y_indices = cls.positions(adata)
        grid_shape = (x_indices.max() + 1, y_indices.max() + 1)
        values = np.asarray(adata.obs[column].values, dtype=float)

        value_grid = np.zeros(grid_shape, dtype=float)
        count_grid = np.zeros(grid_shape, dtype=float)
        np.add.at(value_grid, (x_indices, y_indices), values)
        np.add.at(count_grid, (x_indices, y_indices), 1.0)
        occupied = (count_grid > 0).astype(float)
        mean_grid = np.divide(value_grid, count_grid, out=np.zeros_like(value_grid), where=count_grid > 0)

        smoothed_values = gaussian_filter(mean_grid, sigma=sigma)
        smoothed_weight = gaussian_filter(occupied, sigma=sigma)
        smoothed_grid = smoothed_values / np.maximum(smoothed_weight, 1e-12)
        return smoothed_grid[x_indices, y_indices]


class SpatialPlotter:
    """
    Spatial maps (replacement for ``sc.pl.embedding(adata, basis="spatial", ...)``).

    * The y-axis is flipped at DRAW time only (image convention; adata.obsm["spatial"] is never
      changed), so graphs, smoothing and statistics keep the original coordinates.
    * Square markers by default: round markers on a regular bin grid leave gaps that render as a
      false dotted grid. ``sort_order=False``: scanpy otherwise draws high values on top, which
      makes a sparse score look far brighter than it is on a dense bin grid.
    * Equal x/y scale (square bins); unless ``size`` is given, markers are sized to tile the bin
      grid exactly at the printed size.
    * Saved to ``figdir`` as ``spatial{save}.pdf`` only - a publication-quality PDF
      (plot_style.PublicationStyle: bin layer rasterized at 600 dpi, text and axes vector).
    """

    def __init__(self, figdir: str | os.PathLike | None = None, flip_y: bool = True, show: bool = False):
        self.figdir = figdir
        self.flip_y = flip_y
        self.show = show

    def plot(self, adata: AnnData, color, save: str | None = None, flip_y: bool | None = None, **kwargs):
        kwargs.setdefault("marker", "s")
        kwargs.setdefault("sort_order", False)
        flip = self.flip_y if flip_y is None else flip_y
        auto_size = "size" not in kwargs
        axes = sc.pl.embedding(adata, basis="spatial", color=color, show=False, **kwargs)
        ax_list = axes if isinstance(axes, (list, tuple)) else [axes]
        for ax in ax_list:
            if flip:
                ax.invert_yaxis()
            self.square_bins(ax, adata, auto_size)
        fig = ax_list[0].figure
        if save:
            figdir = self.figdir if self.figdir is not None else sc.settings.figdir
            name = f"spatial{save}"
            for ext in (".pdf", ".png", ".svg", ".jpg"):
                if name.endswith(ext):
                    name = name[: -len(ext)]
            path = save_pdf(fig, f"{figdir}/{name}.pdf")
            log.debug(f"Saved spatial plot to {path}")
        if self.show:
            plt.show()
        plt.close(fig)
        return fig

    @staticmethod
    def square_bins(ax, adata: AnnData, resize_markers: bool = True) -> None:
        """Equal x/y scale (bins are square); unless a size was given, markers sized to tile the grid."""
        ax.set_aspect("equal", adjustable="datalim")           # keeps the axes box (colorbar matches)
        if not resize_markers or "grid_col" not in adata.obs:
            return
        xy = np.asarray(adata.obsm["spatial"], dtype=float)
        n_steps = float(adata.obs["grid_col"].max())
        if n_steps <= 0:
            return
        step = (xy[:, 0].max() - xy[:, 0].min()) / n_steps     # bin spacing in coordinate units
        fig = ax.figure
        fig.canvas.draw()                                      # apply the aspect before measuring
        width_pt = ax.get_window_extent().width * 72.0 / fig.dpi
        x0, x1 = ax.get_xlim()
        side = step * width_pt / abs(x1 - x0)                  # bin side length in points
        for coll in ax.collections:
            try:
                n = len(coll.get_offsets())
            except Exception:
                continue
            if n == adata.n_obs:
                coll.set_sizes([(side * 1.04) ** 2])
