"""
Cell-type annotation of every bin from the high-confidence niches.

A bin belongs to cell type X when it is in X's high-confidence niche - the top 5% of bins
(config niche.quantile = 0.95; 0.90 = top 10%) by
X's Gaussian-smoothed signature score, obs[f"{X}_niche_mask"] == f"{X}_niche", written by
:class:`marqual_st.differential.NicheDE`.

Binned data holds several cells per bin, so one bin can be in several niches:

  one cell type                 -> its name, e.g. "NK"
  several cell types            -> the names joined with "_" in marker_sets order, e.g. "NK_Neuron"
  combination with < min_bins   -> "Mix"
  no niche                      -> "Structural Base"

NOTE: bins annotated with several cell types whose combination has fewer than min_bins
(default 100) bins are labeled as the "Mix" cell type. A single cell type keeps its own name
however few bins it has.

Plots show at most max_plot_labels (default 15) colors, "Mix" included: the largest labels by
number of bins keep their color and all smaller labels are plotted as "Mix". The bin count of
the smallest plotted label (the plotting cutoff) is logged, shown in the plots and saved in uns.
Every label is kept in obs["celltype_annotation"].

Outputs
  obs["celltype_annotation"]       final label (categorical), all labels
  obs["celltype_annotation_plot"]  the same, with labels beyond the plotted ones as "Mix"
  obs["celltype_combination"]      full combination before "Mix" collapsing
  uns["celltype_annotation"]       settings, note, plotting cutoff
  celltype_annotation_counts.csv            bins per label, rank, plotted, plotted_as
  spatial_celltype_annotation.pdf           map of the plotted labels
  celltype_annotation_bin_counts.pdf        bins per plotted label
"""
from __future__ import annotations

import textwrap
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

from ._logging import get_logger
from .plot_style import save_pdf
from .spatial import SpatialPlotter

log = get_logger(__name__)


class Palette:
    """Distinct colors for annotation labels (user colors first, then a curated list, then
    farthest-point colors, so there is always a distinct color for every label)."""

    MIX_LABEL = "Mix"
    BASE_LABEL = "Structural Base"
    MIX_COLOR = "#9467bd"          # purple
    BASE_COLOR = "#E5E5E5"         # soft grey background
    # High-contrast colors in an order where neighbouring colors stay distinguishable
    # with color-vision deficiency.
    AUTO_COLORS = ["#1f77b4", "#ff7f0e", "#008080", "#d62728", "#17becf", "#8c564b", "#FFD700",
                   "#e377c2", "#2ca02c", "#bcbd22", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
                   "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5", "#393b89", "#637939", "#843c39", "#7b4173"]
    MIN_DIST = 45.0                # minimum RGB (0-255) distance between two auto colors

    def __init__(self, user_colors: dict[str, str] | None = None):
        self.user_colors = dict(user_colors or {})

    @staticmethod
    def to_hex(c) -> str:
        return c if isinstance(c, str) else "#%02x%02x%02x" % tuple(int(round(255 * v)) for v in c[:3])

    @staticmethod
    def to_rgb(c: str) -> np.ndarray:
        c = c.lstrip("#")
        return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)

    def build(self, labels: Sequence[str]) -> dict[str, str]:
        """Color per label; labels earlier in the list get the more distinct colors."""
        palette = self.user_colors
        used = {self.to_hex(c).lower() for c in palette.values()} | {self.MIX_COLOR.lower(), self.BASE_COLOR.lower()}
        need = [lab for lab in labels if lab not in palette and lab not in (self.MIX_LABEL, self.BASE_LABEL)]

        taken = [self.to_rgb(c) for c in used]
        pool = []
        candidates = self.AUTO_COLORS + [self.to_hex(c) for cmap in ("tab20", "tab20b", "tab20c")
                                         for c in plt.get_cmap(cmap).colors]
        for c in candidates:
            rgb = self.to_rgb(c)
            if all(np.linalg.norm(rgb - t) >= self.MIN_DIST for t in taken):
                pool.append(c)
                taken.append(rgb)
        if len(pool) < len(need):
            # more labels than listed colors: each extra color is the candidate farthest from every
            # color already in use (candidates: turbo colormap plus darker / lighter shades)
            base = plt.get_cmap("turbo")(np.linspace(0.02, 0.98, 256))[:, :3]
            cand = np.vstack([base, base * 0.6, base * 0.5 + 0.5]) * 255
            taken_arr = np.array(taken)
            for _ in range(len(need) - len(pool)):
                d = np.min(np.linalg.norm(cand[:, None, :] - taken_arr[None, :, :], axis=2), axis=1)
                best = cand[int(np.argmax(d))]
                pool.append("#%02x%02x%02x" % tuple(int(round(v)) for v in best))
                taken_arr = np.vstack([taken_arr, best])
        auto = iter(pool)
        out = {}
        for lab in labels:
            if lab in palette:
                out[lab] = palette[lab]
            elif lab == self.MIX_LABEL:
                out[lab] = self.MIX_COLOR
            elif lab == self.BASE_LABEL:
                out[lab] = self.BASE_COLOR
            else:
                out[lab] = next(auto)
        return out


class CellTypeAnnotator:
    """
    Annotate every bin from the high-confidence niches of ``celltypes``.

    celltypes : names in the order used to join combinations (the keys of marker_sets).
        Each needs obs[mask_fmt] - run NicheDE for it first.
    min_bins : combinations of several cell types with fewer bins are labeled "Mix";
        single cell types always keep their name.
    palette : optional {label: color}; missing labels get distinct automatic colors.
    max_plot_labels : plots use at most this many colors, "Mix" included ("Structural Base" not
        counted). None = all.
    """

    MIX_LABEL = Palette.MIX_LABEL
    BASE_LABEL = Palette.BASE_LABEL
    COUNTS_FILE = "celltype_annotation_counts.csv"

    def __init__(self, outdir: str | Path, celltypes: Sequence[str], min_bins: int = 100,
                 palette: dict[str, str] | None = None, max_plot_labels: int | None = 15,
                 key: str = "celltype_annotation", mask_fmt: str = "{ct}_niche_mask",
                 label_fmt: str = "{ct}_niche", plotter: SpatialPlotter | None = None, plot: bool = True):
        self.outdir = Path(outdir)
        self.celltypes = list(celltypes)
        self.min_bins = min_bins
        self.palette = Palette(palette)
        self.max_plot_labels = max_plot_labels
        self.key = key
        self.mask_fmt = mask_fmt
        self.label_fmt = label_fmt
        self.plotter = plotter or SpatialPlotter(self.outdir)
        self.plot = plot

    @classmethod
    def mix_note(cls, min_bins: int) -> str:
        return (f"Bins annotated with several cell types whose combination has fewer than {min_bins} bins "
                f"are labeled as the '{cls.MIX_LABEL}' cell type.")

    # ------------------------------------------------------------------ main
    def annotate(self, adata: AnnData) -> pd.DataFrame:
        """Annotate ``adata`` in place; returns the counts table (also written to CSV)."""
        MIX, BASE = self.MIX_LABEL, self.BASE_LABEL
        celltypes, min_bins, key = self.celltypes, self.min_bins, self.key
        missing = [ct for ct in celltypes if self.mask_fmt.format(ct=ct) not in adata.obs]
        if missing:
            raise ValueError(f"No niche mask for {missing} (obs[{self.mask_fmt!r}]) - run the niche DE "
                             "for these cell types first.")

        tags = np.column_stack([adata.obs[self.mask_fmt.format(ct=ct)].astype(str).values
                                == self.label_fmt.format(ct=ct) for ct in celltypes])
        names = np.array(celltypes, dtype=object)
        combo = np.array(["_".join(names[row]) if row.any() else BASE for row in tags], dtype=object)
        n_tags = tags.sum(axis=1)

        combo_counts = pd.Series(combo).value_counts()
        # combinations of several cell types with fewer than min_bins bins -> Mix (decided by the
        # number of tags, so a cell-type name containing "_" is still a single type)
        small = {c for c, n in combo_counts.items()
                 if c != BASE and n < min_bins and (n_tags[combo == c] > 1).all()}
        final = np.where(np.isin(combo, list(small)), MIX, combo)

        # category order: single cell types (marker_sets order), combinations (largest first), Mix, Base
        singles = [ct for ct in celltypes if (final == ct).any()]
        combos = [c for c in combo_counts.index if c in set(final) and c not in singles and c != BASE]
        order = singles + combos + ([MIX] if (final == MIX).any() else []) + ([BASE] if (final == BASE).any() else [])
        adata.obs[key] = pd.Categorical(final, categories=order)
        adata.obs["celltype_combination"] = pd.Categorical(combo)
        note = self.mix_note(min_bins)

        rows = []
        for lab in order:
            in_lab = final == lab
            n_ct = int(n_tags[in_lab].max()) if lab not in (MIX, BASE) else (
                0 if lab == BASE else int(n_tags[in_lab].min()))
            kind = ("background" if lab == BASE else "mix" if lab == MIX
                    else "single" if n_ct == 1 else "combination")
            merged = ""
            if lab == MIX:
                merged = "; ".join(f"{c} ({combo_counts[c]})" for c in combo_counts.index if c in small)
            rows.append({"label": lab, "kind": kind, "n_celltypes": n_ct, "n_bins": int(in_lab.sum()),
                         "fraction_of_bins": float(in_lab.mean()), "combinations_in_mix": merged})
        counts = pd.DataFrame(rows)

        # plots color at most max_plot_labels labels, Mix included; smaller labels are plotted as Mix
        ranked = counts[counts["label"] != BASE].sort_values("n_bins", ascending=False, kind="stable")
        counts["rank_by_bins"] = counts["label"].map({lab: i + 1 for i, lab in enumerate(ranked["label"])})
        named = [lab for lab in ranked["label"] if lab != MIX]
        mpl = self.max_plot_labels
        if mpl is None or len(named) + (MIX in order) <= mpl:
            top = named
        else:
            top = named[:max(mpl - 1, 0)]            # one slot is kept for "Mix"
        rest = [lab for lab in named if lab not in top]
        counts["plotted_as"] = [lab if (lab in top or lab in (MIX, BASE)) else MIX for lab in counts["label"]]
        counts["plotted"] = counts["plotted_as"] == counts["label"]
        plot_min_bins = int(ranked.loc[ranked["label"].isin(top), "n_bins"].min()) if top else 0
        n_other_labels = len(rest)
        n_other_bins = int(counts.loc[counts["label"].isin(rest), "n_bins"].sum())

        # colors: plotted labels get the first (most distinct) colors; all labels get one
        plot_order = [lab for lab in order if lab in top]
        colors = self.palette.build(plot_order + [lab for lab in order if lab not in top])
        adata.uns[f"{key}_colors"] = [colors[lab] for lab in order]
        plot_col = f"{key}_plot"
        plot_labels = np.where(np.isin(final, top) | (final == BASE), final, MIX)
        has_mix = bool((plot_labels == MIX).any())
        plot_cats = plot_order + ([MIX] if has_mix else []) + ([BASE] if BASE in order else [])
        adata.obs[plot_col] = pd.Categorical(plot_labels, categories=plot_cats)
        plot_colors = {**{lab: colors[lab] for lab in plot_order}, MIX: colors.get(MIX, Palette.MIX_COLOR),
                       BASE: Palette.BASE_COLOR}
        adata.uns[f"{plot_col}_colors"] = [plot_colors[lab] for lab in plot_cats]
        if n_other_labels:
            cutoff_note = (f"Plots show the {len(top)} largest labels in color (>= {plot_min_bins} bins each); the "
                           f"other {n_other_labels} labels ({n_other_bins} bins) are plotted as '{MIX}', together "
                           f"with combinations of < {min_bins} bins. All labels are in obs['{key}'].")
        else:
            cutoff_note = f"Plots show all {len(ranked)} labels."
        adata.uns[key] = {"celltypes": list(celltypes), "min_bins": int(min_bins),
                          "source": "high-confidence niches (" + self.mask_fmt + ")", "note": note,
                          "max_plot_labels": -1 if mpl is None else int(mpl),
                          "plot_min_bins": plot_min_bins, "n_labels_not_plotted": n_other_labels,
                          "plot_note": cutoff_note}
        self.outdir.mkdir(parents=True, exist_ok=True)
        counts.to_csv(self.outdir / self.COUNTS_FILE, index=False)

        log.debug(f"Cell-type annotation from high-confidence niches ({len(celltypes)} cell types):\n"
                  + counts[["label", "kind", "n_bins", "plotted"]].to_string(index=False))
        log.debug(f"NOTE: {note}")
        log.debug(f"PLOTS: {cutoff_note}")
        log.info(f"  {len(counts)} labels from {len(celltypes)} cell types; bins per label: {self.COUNTS_FILE}")

        if self.plot:
            mix_def = (f"{MIX}: combinations < {min_bins} bins + {n_other_labels} labels < {plot_min_bins} bins"
                       if n_other_labels else f"{MIX}: several cell types, combination < {min_bins} bins")
            self.plotter.plot(adata, color=[plot_col], palette=plot_colors, frameon=False,
                              title=f"Cell-type annotation (high-confidence niches)\n{mix_def}",
                              save="_celltype_annotation")
            self.plot_counts(counts, plot_colors, cutoff_note=cutoff_note)
        return counts

    __call__ = annotate

    # ------------------------------------------------------------------ plot
    def plot_counts(self, counts: pd.DataFrame, colors: dict[str, str],
                    fname: str = "celltype_annotation_bin_counts", cutoff_note: str | None = None) -> None:
        """
        Horizontal bar chart of bins per plotted label; labels plotted as "Mix" are summed into
        the Mix bar. "Structural Base" is left out of the bars; its count is in the title.
        """
        MIX, BASE = self.MIX_LABEL, self.BASE_LABEL
        col = "plotted_as" if "plotted_as" in counts else "label"
        shown = (counts[counts[col] != BASE].groupby(col, sort=False)["n_bins"].sum()
                 .rename_axis("label").reset_index())
        n_merged = int(((counts[col] == MIX) & (counts["label"] != MIX)).sum()) if col == "plotted_as" else 0
        shown = pd.concat([shown[shown["label"] != MIX].sort_values("n_bins", ascending=False, kind="stable"),
                           shown[shown["label"] == MIX]], ignore_index=True)        # Mix last
        if n_merged:
            mix_name = f"{MIX} (incl. {n_merged} smaller labels)"
            shown["label"] = shown["label"].replace(MIX, mix_name)
            colors = {**colors, mix_name: colors.get(MIX, Palette.MIX_COLOR)}
        lab = shown.iloc[::-1]                                          # barh draws bottom-up
        n_base = int(counts.loc[counts["label"] == BASE, "n_bins"].sum())
        fig, ax = plt.subplots(figsize=(3.6, 0.2 * max(len(lab), 1) + 0.9))

        names, values = lab["label"].tolist(), lab["n_bins"].to_numpy()
        y = np.arange(len(names))
        ax.barh(y, values, color=[colors[n] for n in names], height=0.72, edgecolor="white", linewidth=0.6, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6.5, color="#0b0b0b")
        vmax = max(values) if len(values) else 1
        for yy, v in zip(y, values):
            ax.text(v + 0.01 * vmax, yy, f"{int(v):,}", va="center", ha="left", fontsize=6, color="#52514e")
        ax.set_xlim(0, vmax * 1.18)
        ax.set_xlabel("Bins", color="#52514e")
        ax.set_title(f"Bins per annotation label\n({BASE}: {n_base:,} bins, not shown)",
                     fontsize=7.5, color="#0b0b0b", loc="left")
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors="#898781")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        note = "Note: " + self.mix_note(self.min_bins) + (f"\n{cutoff_note}" if cutoff_note else "")
        note = "\n".join(textwrap.fill(par, 110) for par in note.split("\n"))
        ax.annotate(note, xy=(0, 0), xycoords="axes fraction", xytext=(0, -26), textcoords="offset points",
                    fontsize=5.5, color="#52514e", ha="left", va="top", annotation_clip=False)
        path = save_pdf(fig, self.outdir / f"{fname}.pdf")
        plt.close(fig)
        log.debug(f"Saved {path}")

    # ------------------------------------------------------------------ saved h5ad
    def annotate_h5ad(self, path: str | Path, out_path: str | Path | None = None):
        """Annotate a saved analysis h5ad and write it back (to ``out_path``, default the same file) -
        re-annotate with another min_bins or palette without rerunning the pipeline."""
        from .io import AnalysisStore
        adata = AnalysisStore(path).load()
        counts = self.annotate(adata)
        AnalysisStore(out_path or path).save(adata)
        return adata, counts
