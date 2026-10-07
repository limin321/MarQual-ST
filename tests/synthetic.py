"""Synthetic Stereo-seq-like bin50 data for the tests: two tissue sections, a depth gradient,
planted immune + neuron hotspots and a co-localized GNLY / ATP8A2 pair."""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

IMMUNE = ["CCL5", "CCR7", "CD2", "CD3D", "CD3E", "CD3G", "PTPRC", "CD8A", "CD8B", "CD14", "CD163", "CD68", "CD1C",
          "CLEC10A", "FCER1A", "CD79A", "CD79B", "JCHAIN", "MS4A1", "MZB1", "GNLY", "GZMB", "KLRD1", "NKG7", "TPSAB1",
          "TPSB2", "CPA3", "KIT", "MRC1", "IL1B", "NDST2", "MS4A2", "HPGDS", "GATA2"]
NEURON = ["GPC5", "CNTN5", "CPNE4", "RBFOX3", "TUBB3", "MAP2", "UCHL1", "NEFL", "NEFM", "NEFH", "STMN2", "SNAP25",
          "SYT1", "SYP", "ELAVL3", "ELAVL4", "PRPH", "GAP43"]


def make_synthetic(seed: int = 1, n_background_genes: int = 600) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    genes = IMMUNE + NEURON + ["ATP8A2"] + [f"MT-CO{i}" for i in range(1, 4)] + \
        [f"GENE{i}" for i in range(n_background_genes)]
    n_genes = len(genes)
    xs, ys = [], []
    for cx in (25, 85):                                  # two sections side by side
        for x in range(cx - 25, cx + 25):
            for y in range(0, 55):
                if ((x - cx) / 25) ** 2 + ((y - 27) / 28) ** 2 < 1 - 0.1 * np.sin(y / 4):
                    xs.append(x)
                    ys.append(y)
    for x, y in [(55, 5), (56, 5), (55, 50)]:           # stray bins
        xs.append(x)
        ys.append(y)
    xs, ys = np.array(xs), np.array(ys)

    base = rng.gamma(2.0, 1.0, n_genes) * 0.004 + 0.0005
    base /= base.sum()
    gi = {g: i for i, g in enumerate(genes)}
    for g in IMMUNE + NEURON + ["ATP8A2"]:
        base[gi[g]] = 1.5e-5                              # sparse markers
    depth = np.exp(rng.normal(np.log(2000) + 0.006 * ys, 0.35))
    depth[xs > 55] *= 0.8
    rate = np.outer(depth, base)
    d_imm, d_neu = np.hypot(xs - 20, ys - 20), np.hypot(xs - 26, ys - 26)
    d_imm2 = np.hypot(xs - 90, ys - 30)
    for g in IMMUNE:
        rate[:, gi[g]] *= 1 + 60 * np.exp(-(d_imm / 5) ** 2) + 40 * np.exp(-(d_imm2 / 4) ** 2)
    for g in NEURON:
        rate[:, gi[g]] *= 1 + 60 * np.exp(-(d_neu / 6) ** 2)
    rate[:, gi["ATP8A2"]] *= 1 + 120 * np.exp(-(d_imm / 5) ** 2)
    rate[:, gi["GNLY"]] *= 1 + 60 * np.exp(-(d_imm / 5) ** 2)
    for g in ["GENE1", "GENE2", "GENE3"]:
        rate[:, gi[g]] *= 1 + 6 * np.exp(-(d_imm / 5) ** 2)
    X = sp.csr_matrix(rng.poisson(rate).astype(np.float32))
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=[f"bin{i}" for i in range(len(xs))]),
                   var=pd.DataFrame({"real_gene_name": genes}, index=[f"ENSG{i:05d}" for i in range(n_genes)]))
    a.obsm["spatial"] = np.c_[xs * 50 + 1000, ys * 50 + 3000].astype(float)
    return a
