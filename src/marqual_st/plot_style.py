"""
Publication-quality figure output for every MarQual-ST plot.

Every figure is saved as ONE vector PDF (no PNG files):
  - text stays editable: TrueType fonts embedded (pdf.fonttype 42), Arial / Helvetica
    (Liberation Sans or DejaVu Sans when Arial is not installed)
  - journal sizes: 7 pt text, 8 pt titles, 0.6 pt axes; figures sized for a single
    (~89 mm) or double (~183 mm) journal column, so they can be placed at 100%
  - dense layers (spatial bin maps, ~10^5 squares) are rasterized at ``raster_dpi`` inside
    the vector PDF; axes, labels, legends and colorbars stay vector
The HTML report makes its own previews from these PDFs (report.PdfPreviewer).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib


class PublicationStyle:
    """Matplotlib parameters + PDF writer used by every plot of the package."""

    FONT_FAMILY = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]

    def __init__(self, raster_dpi: int = 600, dense_points: int = 2000):
        self.raster_dpi = raster_dpi
        self.dense_points = dense_points

    @property
    def rc(self) -> dict:
        return {
            "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",      # editable, embedded fonts
            "font.family": "sans-serif", "font.sans-serif": self.FONT_FAMILY,
            "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "legend.title_fontsize": 7, "figure.titlesize": 8,
            "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
            "lines.linewidth": 1.0, "patch.linewidth": 0.5,
            "figure.figsize": (3.4, 3.0), "figure.dpi": 100,
            "savefig.dpi": self.raster_dpi, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
            "savefig.facecolor": "white", "savefig.format": "pdf",
            "image.interpolation": "none",
        }

    def apply(self) -> None:
        matplotlib.rcParams.update(self.rc)

    def rasterize_dense(self, fig) -> None:
        """Rasterize scatter collections with many points (keeps PDFs small and fast to open)."""
        for ax in fig.axes:
            for coll in ax.collections:
                try:
                    n = len(coll.get_offsets())
                except Exception:
                    continue
                if n > self.dense_points:
                    coll.set_rasterized(True)

    def save_pdf(self, fig, path: str | os.PathLike, dpi: int | None = None) -> Path:
        """Save ``fig`` as a publication-quality PDF (the extension is forced to .pdf)."""
        path = Path(path).with_suffix(".pdf")
        self.rasterize_dense(fig)
        fig.savefig(path, format="pdf", dpi=dpi or self.raster_dpi, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white", metadata={"Creator": "MarQual-ST"})
        return path


STYLE = PublicationStyle()
STYLE.apply()
save_pdf = STYLE.save_pdf
