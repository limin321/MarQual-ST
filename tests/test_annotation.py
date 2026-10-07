import numpy as np
import pandas as pd
from anndata import AnnData

from marqual_st.annotation import CellTypeAnnotator, Palette


def masked_adata(tags: dict[str, np.ndarray], n: int):
    obs = pd.DataFrame(index=[f"b{i}" for i in range(n)])
    for ct, on in tags.items():
        obs[f"{ct}_niche_mask"] = np.where(on, f"{ct}_niche", "Background")
    ad = AnnData(X=np.zeros((n, 1)), obs=obs)
    ad.obsm["spatial"] = np.c_[np.arange(n) % 30, np.arange(n) // 30].astype(float)
    return ad


def test_labels_mix_and_base(tmp_path):
    n = 600
    a = np.zeros(n, bool)
    b = np.zeros(n, bool)
    a[:200] = True          # A only: 0-149, A_B: 150-199 (50 bins)
    b[150:260] = True       # B only: 200-259 (60)
    b[300:310] = True       # B only +10 -> 70
    c = np.zeros(n, bool)
    c[195:200] = True       # A_B_C: 5 bins -> Mix (< min_bins)
    ad = masked_adata({"A": a, "B": b, "C": c}, n)
    counts = CellTypeAnnotator(tmp_path, ["A", "B", "C"], min_bins=20, plot=False).annotate(ad)
    lab = ad.obs["celltype_annotation"].astype(str).values
    assert (lab[:150] == "A").all()
    assert (lab[150:195] == "A_B").all()
    assert (lab[195:200] == "Mix").all()
    assert (lab[400:] == "Structural Base").all()
    assert set(counts["label"]) == {"A", "B", "A_B", "Mix", "Structural Base"}
    assert (tmp_path / "celltype_annotation_counts.csv").exists()
    assert "A_B_C (5)" in counts.loc[counts["label"] == "Mix", "combinations_in_mix"].iloc[0]


def test_single_celltype_keeps_name_when_small(tmp_path):
    n = 300
    a = np.zeros(n, bool)
    a[:3] = True
    ad = masked_adata({"A": a}, n)
    CellTypeAnnotator(tmp_path, ["A"], min_bins=100, plot=False).annotate(ad)
    assert (ad.obs["celltype_annotation"].astype(str).values[:3] == "A").all()


def test_max_plot_labels(tmp_path):
    n = 1000
    tags = {}
    for k in range(6):                                   # six cell types, decreasing size
        m = np.zeros(n, bool)
        m[k * 150: k * 150 + 150 - 20 * k] = True
        tags[f"T{k}"] = m
    ad = masked_adata(tags, n)
    counts = CellTypeAnnotator(tmp_path, list(tags), min_bins=10, max_plot_labels=4, plot=True).annotate(ad)
    plotted = counts[counts["plotted"] & ~counts["label"].isin(["Mix", "Structural Base"])]
    assert len(plotted) == 3                              # one slot kept for Mix
    assert set(ad.obs["celltype_annotation_plot"].cat.categories) == {"T0", "T1", "T2", "Mix", "Structural Base"}
    assert ad.uns["celltype_annotation"]["n_labels_not_plotted"] == 3
    assert (tmp_path / "celltype_annotation_bin_counts.pdf").exists()
    assert (tmp_path / "spatial_celltype_annotation.pdf").exists()
    assert not list(tmp_path.glob("*.png"))                    # PDF only


def test_palette_many_labels_distinct():
    labels = [f"L{i}" for i in range(80)] + ["Mix", "Structural Base"]
    colors = Palette({"L0": "#000000"}).build(labels)
    assert colors["L0"] == "#000000" and colors["Mix"] == Palette.MIX_COLOR
    auto = [colors[f"L{i}"].lower() for i in range(1, 80)]
    assert len(set(auto)) == len(auto)
