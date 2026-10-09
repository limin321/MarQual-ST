# Changelog

All notable changes to MarQual-ST. Format: [Keep a Changelog](https://keepachangelog.com/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- benchmark, but ignored by Docker image creating.
- Data-quality gate (`quality`, `force` in the config; `DataQualityGate`): share of bins the
  standard filter removes -> PASS / CAUTION (>= 60 %) / STOP (>= 80 % or < 2,000 bins kept).
  STOP ends the run after technical QC on the unfiltered bins and the report (exit code 3);
  `force: true` continues. Verdict in `filter_summary.csv`, `technical_qc_summary.csv` and the
  report Overview.
- `filter_summary.csv`: bins removed by the filter and why (total counts / mito %).
- `{celltype}_niche_definition.csv`: top %, quantile, score threshold and bins of each
  high-confidence niche (size set by `niche.quantile`: 0.95 = top 5%, 0.90 = top 10%).
- Clear error when the filter removes every bin.
- Path check before anything runs (`run`, `validate`): a missing `input` or an `outdir` that
  cannot be created (no write permission) is a one-line config error (exit code 2) instead of a
  traceback.
- "Run parameters" report section and `run_parameters.csv`.
- `environment.yml` installs the compiled libraries (numpy 2.3.5, scipy 1.17.0, scanpy, squidpy
  and their dependencies) from conda-forge instead of pip, so the environment also installs on
  older Linux (glibc 2.17, CentOS 7 / RHEL 7). Before, pip found no numpy 2.3.5 wheel for glibc
  2.17 and failed while compiling it ("NumPy requires GCC >= 9.3"). pip now installs only pandas,
  anndata, esda, libpysal and pypdfium2, from wheels (`--only-binary=:all:`). Same versions as before.
  numba is pinned to 0.63.1 (llvmlite 0.46.0): the conda-forge numba 0.68.0 that conda picked
  failed in `sc.pp.calculate_qc_metrics` on CentOS 7 with numpy 2.3.5.
  The README no longer tells conda users to `pip install -r requirements-lock.txt` (Docker / CI
  only; on glibc 2.17 pip would compile numpy / imagecodecs and fail).

### Changed
- Update README.txt
- Docker Hub tags: `latest` and `<version>` only (the per-commit `sha-<commit>` tag is dropped, so
  images no longer accumulate with every push).
- Gene-pair step, extreme case: when a pair's co-expression niche leaves fewer than 10 bins per
  group for DE (after depth matching) - e.g. "only 1 matched bins per group", where scanpy failed with
  "Could not calculate statistics ... only contain one sample" and the whole step stopped - the pair
  is STOPPED (`RegionTooSmall`): its partial outputs are removed, `{pair}_gene_pair_status.csv`
  records the reason, the terminal shows a warning, the next pair runs, and the report shows the pair
  with a STOP badge and the reason. A later run that succeeds clears the STOP.
- Docker image: `docker run --user ...` failed at startup ("cannot cache function ...: no
  locator available") because the build-time check created the numba cache folder owned by the
  build user. The cache folders are now recreated empty and writable for every user; CI also runs
  the pushed image as an arbitrary user.
- CI/CD (`.github/workflows/ci.yml`, "CI/CD Pipeline"): the conda environment of
  `environment.yml` (as users install it) -> flake8 + ruff -> pytest -> smoke test of the command;
  on push to master the Docker image (`<DOCKERHUB_USERNAME>/marqual-st:latest`, `:sha-<commit>`,
  `:<version>`) is built, pushed and run. Docker image and CI now use `environment.yml` only;
  `requirements-lock.txt` is removed (it replaced conda packages with other pip versions, so the
  image differed from the tested environment). Example sample name: `SAMPLE01`.
- Shorter terminal output: progress, one line per marker set (verdict, z, q), niche (bins, depth
  AUROC, significant DE genes) and pair (verdict, I_AB, q), warnings, the report path. Result
  tables, "saved ... to ..." lines, depth checks and per-test numbers are DEBUG and go to
  `{outdir}/marqual_st.log` only, together with the libraries' warnings (deprecation notices,
  squidpy's graph message). `run,log_level,DEBUG` shows everything in the terminal. The depth
  warning now fires only when the bins used for DE still differ in depth (after matching), and
  names the region; the missing-marker warning names the cell type. Results are unchanged.
- **Run settings are two CSV files instead of the YAML config**: `params.csv` (columns section,
  parameter, value, description; only sample / input / outdir are required, a missing row or empty
  value keeps the default, relative paths are relative to params.csv) and `marker_sets.csv` (one
  row per cell type: name, then genes; no header). Commands take `-p params.csv -m marker_sets.csv`
  (`report`: `-p` only); `init-config -o DIR` writes both templates (`config/params.csv`,
  `config/marker_sets.csv`); `MarQualPipeline.from_csv(params, markers)` in Python. Errors name the
  file line (unknown parameter, wrong type, duplicate setting, duplicate cell type). Every run saves
  `params_used.csv` + `marker_sets_used.csv` in `{outdir}` (all settings, defaults filled in) - pass
  them back to repeat the run; they replace `run_config.yaml`. The YAML loader, the example YAML
  configs and the PyYAML dependency are removed. Results are unchanged.
- Figures are publication-quality PDFs only (`plot_style.PublicationStyle`): embedded,
  editable TrueType fonts (Arial/Helvetica), journal font sizes and figure widths, square bins,
  spatial bin layers rasterized at 600 dpi inside the vector PDF. No PNG files are written; the
  report embeds previews made in memory from the PDFs (`PdfPreviewer`; new dependency
  `pypdfium2`) and links each figure to its PDF.
- One depth-check table per niche, `{name}_depth_check.csv`, with a row per stage
  (`before_matching`, `after_matching`); `{name}_depth_check_after_matching.csv` is no longer
  written. The report shows one before / after table (older runs are still read).
- Report: every scrollable table box shows its header + 5 rows (the rest scrolls; rows that wrap
  stay whole). Overview tables still show in full.
- Report layout: two columns; tables sit side by side in pairs, and figures in pairs share a row
  (each as wide as its aspect ratio, so both have the same height, at most 340 px). Pairs: depth
  maps + high-pass map; raw + depth-corrected null distributions; niche mask + DE dot plot;
  annotation map + bins per label; co-localization null plot + by-distance plot (with the summary
  and by-distance tables side by side above them); gene-pair niche + DE dot plot.
- Report, Run parameters: shows only the marker sets and software versions; every other setting
  stays in `run_parameters.csv` (linked), `params_used.csv` and `uns["run_parameters"]`.

## [0.1.0] - 2026-09-28

First packaged release. Results are identical to the earlier script version on the same data
and settings (all result tables and the analysis h5ad).

### Added
- Installable Python package `marqual_st` (src layout, `pyproject.toml`) with the `marqual-st`
  command: `init-config`, `validate`, `run`, `gene-pairs`, `annotate`, `report`.
- One YAML config per run (`PipelineConfig`), validated before anything runs; unknown keys are
  rejected. The resolved config is saved as `run_config.yaml` next to the results.
- Object-oriented modules: one class per analysis step, `MarQualPipeline` and
  `GenePairPipeline` orchestrate them; each step is a public method.
- Logging to the console and `{outdir}/marqual_st.log`, with step timings.
- Tests (pytest; synthetic data, unit + end-to-end), ruff lint, `tools/compare_runs.py`
  regression check.
- `environment.yml` (pinned), `requirements-lock.txt` (all dependencies), `Dockerfile`,
  GitHub Actions CI/CD: tests on every push / pull request, Docker image pushed to
  `limin321/marqual-st` on every push to `master`.

### Changed
- Default analysis h5ad name: `{sample}.marker_pipeline.h5ad` (set `analysis_h5ad` in the
  config to keep another name).
- Settings are no longer edited in `pipeline.py` / `gene_pair_colocalization_pipeline.py`;
  the report guide refers to config keys.
