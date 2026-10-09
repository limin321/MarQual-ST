"""
Save Scanpy figures that meet GigaScience (OUP) figure requirements.

GigaScience "Preparing Figures" spec (Instructions to Authors):
  - Width: 85 mm (half page) or 170 mm (full page); max height 225 mm
  - Resolution: ~300 dpi at final size
  - Text: no smaller than 7 pt
  - Line width: 0.25-1 pt
  - Formats: vector -> PDF/EPS/SVG with embedded fonts (PDF preferred);
             raster -> uncompressed or LZW-compressed TIFF
  - Colour: RGB recommended
  - Each figure file < 10 MB; multi-panel figures in ONE composite file
  - Name files by order of citation: fig1.tif, fig2.pdf, ...
  - Titles/legends go in the manuscript, not inside the image
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
from PIL import Image

MM_PER_INCH = 25.4
GIGA_WIDTH_MM = {"single": 85, "double": 170}
GIGA_MAX_HEIGHT_MM = 225
GIGA_DPI = 300  # 600 is also fine and gives sharper line art
MAX_FILE_MB = 10


def mm2in(mm):
    return mm / MM_PER_INCH


def set_gigascience_style(base_fontsize=7):
    """Global matplotlib/scanpy style for GigaScience figures."""
    sc.set_figure_params(
        dpi=100,               # on-screen preview only
        dpi_save=GIGA_DPI,
        frameon=False,
        vector_friendly=True,  # rasterise scatter dots, keep text/axes as vectors
        fontsize=base_fontsize,
        format="pdf",
        transparent=False,
    )
    mpl.rcParams.update({
        # Fonts: sans-serif, embedded as TrueType (Type 42) so text stays editable
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # Text >= 7 pt everywhere
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 1,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize,
        "legend.title_fontsize": base_fontsize,
        "figure.titlesize": base_fontsize + 1,
        # Line widths within 0.25-1 pt
        "axes.linewidth": 0.5,
        "lines.linewidth": 0.75,
        "patch.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.25,
        "ytick.minor.width": 0.25,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.linewidth": 0.25,
        # Output
        "figure.dpi": 100,
        "savefig.dpi": GIGA_DPI,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,  # trim white border, keep >= 2 px
        "axes.grid": False,
    })


def new_figure(width="double", height_mm=100, nrows=1, ncols=1,
               wspace=0.45, hspace=0.55, **kwargs):
    """
    Create a figure at the exact final print size.

    Uses fixed spacing instead of constrained_layout, because Scanpy's
    dotplot/matrixplot/heatmap add their own sub-axes inside `ax`, which
    makes constrained_layout collapse. Tune wspace/hspace if labels overlap.
    """
    if height_mm > GIGA_MAX_HEIGHT_MM:
        raise ValueError(f"Height {height_mm} mm exceeds GigaScience max of {GIGA_MAX_HEIGHT_MM} mm")
    w_mm = GIGA_WIDTH_MM[width] if isinstance(width, str) else width
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(mm2in(w_mm), mm2in(height_mm)),
        gridspec_kw={"wspace": wspace, "hspace": hspace},
        **kwargs,
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.10, top=0.94)
    return fig, axes


def add_panel_labels(axes, labels="abcdefghijklmnopqrstuvwxyz", fontsize=9, uppercase=False):
    """Add bold panel letters (a, b, c ...) to the top-left of each axis."""
    for ax, lab in zip(list(axes), labels):
        lab = lab.upper() if uppercase else lab
        ax.text(-0.08, 1.04, lab, transform=ax.transAxes,
                fontsize=fontsize, fontweight="bold", va="bottom", ha="right")


def rasterize_points(fig):
    """Rasterise dense scatter layers only (keeps PDFs small, text stays vector)."""
    for ax in fig.axes:
        for coll in ax.collections:
            if isinstance(coll, mpl.collections.PathCollection):
                coll.set_rasterized(True)


def save_gigascience(fig, name, outdir="figures_gigascience", formats=("pdf", "tif"), dpi=GIGA_DPI):
    """
    Save one composite figure in GigaScience-ready formats.

    name    : e.g. "fig1" (match order of citation in the text)
    formats : any of "pdf", "eps", "svg", "tif", "png"
    """
    os.makedirs(outdir, exist_ok=True)
    rasterize_points(fig)

    w_in, h_in = fig.get_size_inches()
    print(f"[{name}] size = {w_in * MM_PER_INCH:.0f} x {h_in * MM_PER_INCH:.0f} mm @ {dpi} dpi")

    paths = []
    for fmt in formats:
        path = os.path.join(outdir, f"{name}.{fmt}")
        if fmt in ("tif", "tiff"):
            # Save, then flatten to RGB and re-save with lossless LZW compression
            fig.savefig(path, dpi=dpi, format="tiff")
            with Image.open(path) as im:
                im.convert("RGB").save(path, compression="tiff_lzw", dpi=(dpi, dpi))
        elif fmt == "eps":
            # EPS has no transparency; rasterised layers are embedded at `dpi`
            fig.savefig(path, dpi=dpi, format="eps")
        else:
            fig.savefig(path, dpi=dpi, format=fmt)

        size_mb = os.path.getsize(path) / 1e6
        flag = "  <-- exceeds 10 MB, reduce dpi or rasterise more" if size_mb > MAX_FILE_MB else ""
        print(f"   saved {path} ({size_mb:.2f} MB){flag}")
        paths.append(path)
    return paths


# ----------------------------------------------------------------------------
# Example: a 4-panel composite figure (full page width, 170 x 150 mm)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    set_gigascience_style(base_fontsize=7)

    adata = sc.datasets.pbmc3k_processed()  # replace with your own AnnData

    fig, axes = new_figure(width="double", height_mm=150, nrows=2, ncols=2)
    axes = axes.ravel()

    # Always pass ax=... and show=False so Scanpy draws into YOUR sized figure
    sc.pl.umap(adata, color="louvain", ax=axes[0], show=False,
               legend_loc="on data", legend_fontsize=7, legend_fontoutline=1,
               title="Cell types", size=4)
    sc.pl.umap(adata, color="CST3", ax=axes[1], show=False,
               color_map="viridis", title="CST3", size=4)
    sc.pl.violin(adata, keys="NKG7", groupby="louvain", ax=axes[2], show=False,
                 rotation=90, stripplot=False)
    axes[2].set_title("NKG7")
    sc.pl.dotplot(adata, var_names=["CST3", "NKG7", "PPBP", "MS4A1", "CD3E"],
                  groupby="louvain", ax=axes[3], show=False)

    add_panel_labels(axes[:3])  # dotplot creates its own sub-axes; label it by hand if needed
    axes[3].text(-0.02, 1.04, "d", transform=axes[3].transAxes,
                 fontsize=9, fontweight="bold", va="bottom", ha="right")

    save_gigascience(fig, "fig1", formats=("pdf", "tif"))
    plt.close(fig)

    # --- Single-column (85 mm) figure ------------------------------------------
    fig, ax = new_figure(width="single", height_mm=70)
    sc.pl.umap(adata, color="louvain", ax=ax, show=False, size=3,
               legend_fontsize=7, frameon=False, title="")
    save_gigascience(fig, "fig2", formats=("pdf", "tif"))
    plt.close(fig)
