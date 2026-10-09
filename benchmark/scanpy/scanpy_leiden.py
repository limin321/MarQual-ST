# Save GitaScience publication-level figures
dir2 = "/stomics_data/liminData/project/STO081/mayo/Y40283GC/workflow/benchmark"
dir1 = "/stomics_data/liminData/project/STO081/mayo/Y40283GC/workflow/script"
sys.path.append(dir1)

import sys
import matplotlib.pyplot as plt
import scanpy as sc
from gigascience_figures import set_gigascience_style, new_figure, mm2in, save_gigascience

set_gigascience_style()   # run once, before any plotting

outdir = "/stomics_data/liminData/project/STO081/mayo/Y40283GC/analysis/whole_tissue/bin50"
sc.settings.figdir = outdir
adata = sc.read_h5ad(f"{outdir}/Y40283GC.tissue.bin50.processed.h5ad")

# ---- Figure: spatial clusters ----------------------------------------------
fig, ax = new_figure("single", height_mm=70)     # 85 x 70 mm
sc.pl.spatial(
    adata,
    img=None,
    color="leiden",
    spot_size=50,
    frameon=False, title="",
    ax=ax, show=False,                           # replaces save="_cluster_spatial.png"
)
ax.set_aspect("equal")                           # keep tissue proportions
save_gigascience(fig, f"{dir2}/sptail_dist")                    # -> fig4.pdf + fig4.tif
plt.close(fig)


# ---- Figure: marker-gene dotplot -------------------------------------------
anndata = adata.copy()
cluster_col = "leiden"
# 2. Clean up unused categories so Scanpy plotting functions don't expect them
anndata.obs[cluster_col] = anndata.obs[cluster_col].cat.remove_unused_categories()
sc.tl.dendrogram(anndata, groupby=cluster_col)

# Find markers
sc.tl.rank_genes_groups(
    anndata, 
    groupby=cluster_col, 
    method='wilcoxon',
    layer="lognorm", 
    pts=True, 
    use_raw=False
)

dp = sc.pl.rank_genes_groups_dotplot(
    anndata, n_genes=5,
    #gene_symbols="real_gene_name",
    use_raw=False,
    swap_axes=True,                              # genes as rows -> fits the page
    dendrogram=False,
    figsize=(mm2in(85), mm2in(225)),             # half-page width, max height
    return_fig=True, show=False,
)
dp.legend(width=1.0).style(dot_edge_lw=0.25).make_figure()
save_gigascience(dp.fig, f"{dir2}/dotplot")                 # -> fig3.pdf + fig3.tif
plt.close(dp.fig)


# Extract positive marker genes for cell annotation
markers_df = sc.get.rank_genes_groups_df(
    anndata, 
    key='rank_genes_groups',
    group=None,
    log2fc_min=0
)

markers_df = markers_df[markers_df['pvals_adj'] < 0.05]
markers_df.to_csv(f'{dir2}/Y40283GC.bin50_{cluster_col}_markers.csv', index=False)

