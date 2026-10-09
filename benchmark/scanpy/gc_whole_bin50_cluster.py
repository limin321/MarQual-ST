#! /bin/bash/env python
import os
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

dir1 = "/stomics_data/liminData/project/STO081/mayo/Y40283GC"
outdir="/stomics_data/liminData/project/STO081/mayo/Y40283GC/analysis/whole_tissue/bin50"
os.makedirs(outdir, exist_ok=True)

adata = sc.read_h5ad(f"{dir1}/data/Y40283GC.tissue.bin50.h5ad")

adata.var["ensembl_id"] = adata.var_names
adata.var.index = adata.var['real_gene_name'].astype(str)
adata.var_names_make_unique()


# To regress out mt-, ribo,
adata.var["ribo"] = adata.var["real_gene_name"].str.startswith(("RPS", "RPL", "rps", "rpl"))
adata.var["mt"]=adata.var["real_gene_name"].str.startswith(("MT-", "MTCO", "mt-"))

# # Calcualte QC matrix
sc.pp.calculate_qc_metrics(
    adata, 
    qc_vars=["mt", "ribo"],
    inplace=True
)


# Plot QC plots per sample
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo",],
    multi_panel=True
)


# filtering based on this histogram
sc.pp.filter_cells(adata, min_counts=600)
sc.pp.filter_genes(adata, min_cells=6)

# Remove dying cells.
adata = adata[adata.obs.pct_counts_mt < 20,:]

# # Filter out extreme Ribosomal cells 
# adata = adata[adata.obs.pct_counts_ribo < 20, :]

# Store raw data
adata.layers['counts'] = adata.X.copy()


# 1. Normalize and log-transformation the main matrix (.X)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
adata.layers["lognorm"] = adata.X.copy()

# 2. Find highly variable genes first (while variance is pure)
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=2000,
    #batch_key="sample",  # CRITICAL: Forces HVGs to be chosen per sample
    layer="counts",
    subset=False
)

# Regress them out to clean the biological signal
# (Do NOT run the genes_to_keep filtering step if you do this)
sc.pp.regress_out(adata, ['total_counts','pct_counts_mt', 'pct_counts_ribo'])


sc.pp.scale(
    adata,
    zero_center=False,
    max_value=10
)

# Downstream embedding and clustering
sc.pp.pca(adata, n_comps=30, random_state=42)
sc.pp.neighbors(
    adata, metric="cosine",
    n_neighbors=20,
    n_pcs=30
)
sc.tl.umap(adata)
sc.tl.leiden(adata, flavor='igraph', n_iterations=-1, resolution=0.3)
sc.pl.umap(adata, color="leiden")
sc.pl.spatial(
    adata, 
    color="leiden",
    img=None,
    spot_size=30,
    save="_cluster_spatial_distribution.png"
)

adata.var.index.name = None
adata.write_h5ad(f"{outdir}/Y40283GC.tissue.bin50.processed.h5ad")
