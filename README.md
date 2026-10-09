# MarQual-ST

**Mar**ker-based **Quali**tative **Qual**ity control framework for high-resolution binned (or single-cell level) **S**patial **T**ranscriptomics
(Stereo-seq, Visium HD) data.

Low-quality spatial data (few reads per bin, strong technical depth variation) often cannot be
clustered: clusters mostly follow sequencing depth. MarQual-ST starts from known marker gene sets
and serves:

2. **Qualitative QC** – does each marker set show real spatial structure, beyond random genes of
   matched expression and beyond sequencing depth? (`qc_verdict`: PASS / DEPTH_DRIVEN /
   MASKED_BY_DEPTH / NO_SPATIAL_SIGNAL). It performs six steps: technical QC, spatial niche discovery, depth-matched DE, cell-type annotation, cell types co-localization, optional gene-pairs co-localization.
2. **Low-quality ST analysis workflow** – When ST data fail the standard analysis workflow, when to abandon the data? or still any biological signal exist and how to extract them?

Every run ends with one self-contained HTML report, `{sample}_QCreport.html`.
How to read it: [`docs/report_guide.pdf`](docs/report_guide.pdf).

---

## Install

### Conda (development, notebooks, HPC)

```bash
conda env create -f environment.yml      # or: mamba / micromamba create -f environment.yml
conda activate marqual-st
pip install --no-deps -e .               # adds the `marqual-st` command
marqual-st --version
```

`environment.yml` pins Python 3.11.13 and the analysis libraries (scanpy 1.11.5, squidpy 1.7.0,
anndata 0.12.8, numpy 2.3.5, pandas 2.2.3, scipy 1.17.0, matplotlib 3.10.8, statsmodels 0.14.6,
esda 2.9.0, libpysal 4.14.1, Pillow 12.0.0, numba 0.63.1). `setuptools<81` is required:
squidpy 1.7.0 → spatialdata → xarray_schema still imports `pkg_resources`.

The compiled libraries (numpy, scipy, h5py, ...) are installed from conda-forge, whose builds also
run on older Linux (glibc 2.17, e.g. CentOS 7); pip adds only pure-Python packages, pandas and
pypdfium2 from prebuilt wheels and never compiles anything. If an earlier attempt left a broken
environment, remove it first: `conda env remove -n marqual-st`.

### Docker

```bash
docker pull limin321/marqual-st:latest
docker run --rm --user "$(id -u):$(id -g)" -v /path/to/project:/data \
    limin321/marqual-st:latest run -p /data/params.csv -m /data/marker_sets.csv
```

The container sees only the folders you mount: `-v /path/to/project:/data` makes your folder
`/data` inside it. So `input` and `outdir` in `params.csv` must use the container paths
(e.g. `/data/SAMPLE01.tissue.bin50.h5ad`, `/data/results/SAMPLE01`) - or paths relative to
`params.csv`, which work inside and outside the container. `--user` makes the result files belong
to you. On HPC with Singularity/Apptainer:
`apptainer run docker://limin321/marqual-st:latest run -p params.csv -m marker_sets.csv`.

---

## Use

A run needs two CSV files: **`params.csv`** (settings) and **`marker_sets.csv`** (marker genes).

```bash
# marqual-st init-config -o my_run          # writes my_run/params.csv + my_run/marker_sets.csv to edit
marqual-st validate   -p my_run/params.csv -m my_run/marker_sets.csv   # check both, and the paths
marqual-st run        -p my_run/params.csv -m my_run/marker_sets.csv   # -> {outdir}/figures/{sample}_QCreport.html
marqual-st gene-pairs -p my_run/params.csv -m my_run/marker_sets.csv   # optional gene-pair step, report rebuilt
marqual-st gene-pairs -p ... -m ... --pair GNLY ATP8A2 --pair CD3E CD8A
marqual-st annotate   -p ... -m ...       # re-annotate the saved h5ad (new min_bins / palette)
marqual-st report     -p my_run/params.csv  # rebuild the report only

# Note, the missing part in outdir will be automatically created when provided in params.csv
```

From Python / a notebook:

```python
from marqual_st import MarQualPipeline, GenePairPipeline

pipe = MarQualPipeline.from_csv("params.csv", "marker_sets.csv")
adata = pipe.run()                 # or step by step: pipe.load(); pipe.detect_sections(); ...
GenePairPipeline(pipe.config).run()
```

### `marker_sets.csv`

One row per cell type: the cell-type name, then its marker genes, one gene per cell. No header
(a first row starting with `celltype` / `cell_type` is skipped). Empty cells are ignored, so rows
can have different lengths; a gene listed twice in one set is kept once. An example:

```
Neuron,GPC5,CNTN5,CPNE4,RBFOX3,TUBB3,MAP2,UCHL1,NEFL,SNAP25,SYT1
NK,GNLY,GZMB,KLRD1,NKG7
Mast,CPA3,KIT,TPSAB1,TPSB2
```

This file is the only place cell types are named; every step loops over its rows. All sets in
the file are **one multiple-testing family**: q-values and verdicts are corrected across exactly
these sets. Fix the list before looking at results and report all of them - adding sets makes the
correction stricter, removing sets looser. Sets that share genes (e.g. two T-cell sets with
CD3D/E/G) look co-localized partly because of the shared genes.

### `params.csv`

Columns `section, parameter, value, description` - one row per setting, easy to edit in Excel.
Only `sample`, `input` and `outdir` are required: a row that is missing, or has an empty value,
keeps the default. `init-config` writes every setting with its default and a description
(see [`config/params.csv`](config/params.csv)). Relative paths are relative to the folder of
`params.csv`. Unknown sections or parameters, values of the wrong type (e.g. `95%` for a number)
and settings given twice are errors that name the line.

| section | parameters |
|---|---|
| `run` | `sample`, `input` (UNFILTERED tissue-cut input: `stereo` h5ad, or `visiumhd` Space Ranger `square_XXXum` folder), `outdir`, `platform`, `analysis_h5ad`; `panel_region_celltypes` (`all` or `NK; Mast`); `celltype_pairs` (`NK:Neuron; Mast:Fibroblasts` or `all`); `force`, `display_plots`, `log_level` |
| `grid` | `bin_size`, `um_per_bin` |
| `filters` | `min_counts`, `min_cells`, `pct_mt` (keep the Stereo-seq bin50 cutoffs 600 / 6 / 20 fixed). You should change the cutoffs based-on bin-size. |
| `quality` | data-quality gate on the share of bins the filter removes: CAUTION at ≥ `caution_pct_filtered` (60 %), STOP at ≥ `stop_pct_filtered` (80 %) or < `min_bins_kept` (2,000) bins kept - the run ends after technical QC + report (exit code 3); `run,force,true` continues after STOP |
| `sections` | `num_tissue`, `tissue_qc` |
| `statistics` | random gene sets for the null tests (`n_null`, `n_null_distance`), `seed` |
| `niche` | `quantile` (0.95 = top 5 % of bins by smoothed score; 0.90 = top 10 %), `top_n` DE genes |
| `annotation` | `min_bins`, `max_plot_labels`, `palette` (`NK=#1f77b4; Mix=#9467bd`) |
| `gene_pairs` | `pairs` (`GNLY:ATP8A2; CD3E:CD8A`), `permutations` |

Lists go in one cell, separated by `;`; pairs are written `A:B`; true / false (or yes / no).

The terminal shows progress and one line per marker set, niche and pair (verdict, key numbers),
plus warnings; everything else - result tables, files written, depth checks, library warnings -
goes to `{outdir}/marqual_st.log`. `run,log_level,DEBUG` prints it all to the terminal too.

### Outputs (`{outdir}`)

```
{sample}.marker_pipeline.h5ad   analysis AnnData (layers, spatial graph, scores, masks, celltype_annotation)
params_used.csv                 every setting of the run, defaults filled in  } pass both back
marker_sets_used.csv            the marker sets as run                       } to repeat the run
marqual_st.log                  full log for troubleshooting: result tables, files written, library warnings
figures/                        every table, every figure (publication-quality PDF) + {sample}_QCreport.html
figures/gene_pairs/             gene-pair step
```

---

## Design

Source layout (`src/marqual_st/`), one class per responsibility, following the pipeline order:

| module | classes | role |
|---|---|---|
| `config.py` | `PipelineConfig` (+ section dataclasses) | params.csv + marker_sets.csv → validated, typed settings |
| `io.py` | `DataLoader` (abstract) → `StereoSeqLoader`, `VisiumHDLoader`; `Preprocessor`; `AnalysisStore` | platform loading (registry by platform name), filter + normalize, analysis h5ad |
| `spatial.py` | `SpatialGrid`, `SpatialPlotter` | grid positions, 8-neighbour graph, smoothing; spatial maps |
| `technical_qc.py` | `TechnicalQC` | sections, strays, depth maps, high-pass map |
| `signature.py` | `SignatureAnalysis` | signature score, analytic Moran's I, smoothed scores |
| `depth.py` | `DepthModel`, `MatchedGeneSetSampler`, `moran_i` | depth covariates / residuals, matched random gene sets |
| `null_test.py` | `RandomGeneSetNullTest`, `MoranIndexTable`, `DepthConfoundChecks`, `NullDistributionPlotter` | random-gene-set null test, FDR + `qc_verdict` |
| `moran_summary.py` | `MoranSummary` | one table + chart of Moran's I across marker sets |
| `differential.py` | `RegionDE` (abstract, template method) → `NicheDE`, `MarkerPanelDE`; `DepthMatcher` | region definition → depth check / matching → DE → dot plot |
| `colocalization.py` | `SignatureColocalization`, `ColocalizationTable` | I_AB vs swap-A / swap-B nulls, by distance, FDR |
| `annotation.py` | `CellTypeAnnotator`, `Palette` | bin labels from the niches, Mix / Structural Base |
| `gene_pair.py` | `GenePairAnalysis`, `CoexpressionNicheDE(RegionDE)` | bivariate Moran's I, correlation, co-expression niche DE |
| `report.py` | `ReportBuilder`, `FileIndex`, `Html`, `ImageEmbedder` | HTML report (template in `templates/report.html`) |
| `pipeline.py` | `MarQualPipeline`, `GenePairPipeline` | orchestration (each step is a method) |
| `cli.py` | – | `marqual-st` command |

Principles: every analysis class receives its settings in the constructor and exposes a `run`/
`test`/`annotate` method on an AnnData (dependency injection – plotters, grids and depth models
are passed in, so they can be swapped or mocked); new platforms are a `DataLoader` subclass,
new region definitions a `RegionDE` subclass; logging instead of `print`; no settings in code.

The package reproduces the earlier script version (`pipeline.py` + modules) exactly: on the same
data and settings all 62 result tables and the analysis h5ad are identical
(`tools/compare_runs.py OLD/figures NEW/figures`).

---

## Development

```bash
conda env create -f environment.yml && conda activate marqual-st
pip install --no-deps -e . && pip install pytest flake8 ruff
ruff check src tests          # lint
pytest -q                     # unit + end-to-end tests on synthetic data (~30 s)
pytest -q -m "not slow"       # unit tests only
```

Workflow: branch → change + tests → `ruff` / `pytest` → pull request → merge to `master`.
Bump `__version__` in `src/marqual_st/__init__.py` (the only place it is set) and add a
`CHANGELOG.md` entry for each release.

### CI/CD (`.github/workflows/ci.yml`)

* every push and pull request to `master` (and *Run workflow* in the Actions tab): the conda
  environment of `environment.yml` → lint (flake8, ruff) → tests (pytest) → smoke test of the
  `marqual-st` command
* push to `master`, after the tests pass: the Docker image is built from the same environment and
  pushed to Docker Hub as `<user>/marqual-st:latest` and `:<version>` (two names for one image).
  Bump `__version__` for a release: the previous version tag stays available unchanged - cite
  the version tag (e.g. `limin321/marqual-st:0.1.0`) for reproducibility

## Scientific Registry

* **RRID:** [RRID:SCR_029119](https://scicrunch.org)
* **bio.tools ID:** [biotools:marqual-st](https://bio.tools)
* **DOI:** https://doi.org/10.5281/zenodo.23227164

## License

MIT – see [LICENSE](LICENSE).
