#! /usr/bin/bash

# left-uterosacral ligament
dir1=/stomics_data/liminData/project/STO081/mayo
dir3=${dir1}/analysis/bin50/SDAS/cell2location_complete
dir2=${dir1}/analysis/bin50/sdas/Y40283GC/cell2location


mkdir -p ${dir2}/ref ${dir2}/c2location

ref_h5ad=${dir1}/data/single-cell/endometriumAtlasV2_cells_complete.h5ad
spatial_data=${dir1}/Y40283GC_FFPE/Y40283GC.GC1.label_bin50.h5ad
bin_size=50

fname=$(basename "${ref_h5ad}" .h5ad)


# Run SDAS
/stomics_data/bwudata/SDAS/SDAS/sdas-1.0.0/SDAS cellAnnotation cell2location \
    -i ${spatial_data} \
    -o ${dir2} \
    --reference_csv ${dir3}/ref/${fname}_inf_aver.csv \
    --input_gene_symbol_key real_gene_name \
    --seed 59 \
    --bin_size ${bin_size} \
    --gpu_id 0
