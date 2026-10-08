"""
The two runnable workflows.

MarQualPipeline (``marqual-st run``)
  1. load the unfiltered data, grid positions, spatial graph
  2. tissue sections + strays (unfiltered bins)
  3. data-quality gate (PASS / CAUTION / STOP; STOP ends the run after technical QC + report
     unless force), filter + log-normalize
  4. technical (marker-independent) spatial QC
  5. spatial structure of each marker set: signature score, Moran's I, random-gene-set null
     test (raw + depth-corrected), qc_verdict, Moran's I summary
  6. niches and DE (high-confidence niche; any-marker region), depth-matched, dot plots
  7. cell-type co-localization (+ within-section control when tissue_qc)
  8. cell-type annotation of every bin from the high-confidence niches
  9. save the analysis h5ad, build the HTML report

GenePairPipeline (``marqual-st gene-pairs``)
  Optional follow-up on the saved analysis h5ad: gene-pair co-localization for every pair in
  ``gene_pairs.pairs``; rebuilds the report so it includes the results.

The marker sets of a run are ONE multiple-testing family: q-values and verdicts are corrected
across exactly these sets. Fix the list before looking at results and report all of them.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import scanpy as sc
from anndata import AnnData

from ._logging import LOG_FILE, configure_logging, get_logger
from .annotation import CellTypeAnnotator
from .colocalization import ColocalizationTable, SignatureColocalization
from .config import USED_MARKERS_FILE, USED_PARAMS_FILE, PipelineConfig
from .differential import MarkerPanelDE, NicheDE
from .gene_pair import GenePairAnalysis
from .io import AnalysisStore, DataLoader, DataQualityGate, DataQualityStop, Preprocessor
from .moran_summary import MoranSummary
from .null_test import DepthConfoundChecks, MoranIndexTable, NullDistributionPlotter, RandomGeneSetNullTest
from .report import ReportBuilder
from .run_params import RunParameters, settings_of
from .signature import SignatureAnalysis
from .spatial import SpatialGrid, SpatialPlotter
from .technical_qc import TechnicalQC

log = get_logger(__name__)


class _Workflow:
    """Shared plumbing: config, output folders, logging, step timing."""

    LOG_FILE = LOG_FILE
    NEEDS_INPUT = False            # the main pipeline reads `input`; the follow-up steps read the h5ad

    def __init__(self, config: PipelineConfig, configure_logs: bool = True):
        self.config = config.validate().check_paths(need_input=self.NEEDS_INPUT)
        self.figdir = config.figdir
        self.figdir.mkdir(parents=True, exist_ok=True)
        if configure_logs:
            configure_logging(config.log_level, Path(config.outdir) / self.LOG_FILE)
        self.plotter = SpatialPlotter(self.figdir, show=config.display_plots)
        self.timings: dict[str, float] = {}
        self.params = RunParameters()          # -> run_parameters.csv, "Run parameters" in the report
        self.params.add_config(config)

    MERGE_PARAMS = False                       # the main run writes its full record; follow-up steps merge
    @classmethod
    def from_csv(cls, params: str | Path, markers: str | Path, **kwargs):
        """Build from params.csv + marker_sets.csv."""
        return cls(PipelineConfig.from_csv(params, markers), **kwargs)

    @contextmanager
    def step(self, name: str):
        log.info(f"========== {name} ==========")
        t0 = time.perf_counter()
        yield
        self.timings[name] = time.perf_counter() - t0
        log.debug(f"{name}: done in {self.timings[name]:.1f} s")

    def build_report(self) -> Path:
        self.params.write(self.figdir, merge=self.MERGE_PARAMS)
        return ReportBuilder(self.figdir, sample=self.config.sample).build()


class MarQualPipeline(_Workflow):
    """
    The main marker-based spatial QC workflow. ``run()`` executes every step in order; each step
    is also a public method, so a notebook can run them one at a time:

        pipe = MarQualPipeline.from_csv("params.csv", "marker_sets.csv")
        adata = pipe.run()
    """

    NEEDS_INPUT = True

    def __init__(self, config: PipelineConfig, configure_logs: bool = True):
        super().__init__(config, configure_logs)
        c = config
        self.grid = SpatialGrid(bin_size=c.grid.bin_size)
        self.technical = TechnicalQC(self.figdir, num_tissue=c.sections.num_tissue, tissue_qc=c.sections.tissue_qc,
                                     plotter=self.plotter)
        self.signature = SignatureAnalysis(self.figdir, grid=self.grid, plotter=self.plotter)
        self.null_test = RandomGeneSetNullTest(self.figdir, n_null=c.statistics.n_null, seed=c.statistics.seed,
                                               residualize=True, covariates=("log_total_counts",))
        self.adata: AnnData | None = None
        self.null_results: list[dict] = []
        self.coloc_results: list[dict] = []

    # ------------------------------------------------------------------ steps
    def load(self) -> AnnData:
        """Step 1: unfiltered data + QC metrics, grid positions, spatial graph (built once)."""
        adata = DataLoader.for_platform(self.config.platform).load(self.config.input)
        self.grid.add_positions(adata)
        self.grid.build_graph(adata)
        self.adata = adata
        return adata

    def detect_sections(self) -> None:
        """Step 2: tissue sections + strays on the UNFILTERED bins (filter holes cannot split a section)."""
        self.technical.detect_sections(self.adata)

    @property
    def preprocessor(self) -> Preprocessor:
        f = self.config.filters
        return Preprocessor(f.min_counts, f.min_cells, f.pct_mt)

    def quality_gate(self) -> dict:
        """Step 3a: share of bins the filter removes -> PASS / CAUTION / STOP. On STOP (without force):
        technical QC on the UNFILTERED bins (shows where capture failed) + report, then DataQualityStop."""
        q = self.config.quality
        gate = DataQualityGate(self.preprocessor, q.caution_pct_filtered, q.stop_pct_filtered, q.min_bins_kept,
                               force=self.config.force)
        self.params.add("DataQualityGate", settings_of(gate))
        self.params.add("Preprocessor", settings_of(gate.preprocessor))
        self.params.add("TechnicalQC", settings_of(self.technical))
        stats = gate.assess(self.adata)
        if gate.should_stop(stats):
            self.technical_qc()
            self.build_report()
            raise DataQualityStop(f"{self.config.sample}: {stats['data_quality_reason']} Technical QC and the report "
                                  f"are in {self.figdir}. Set force: true to continue anyway (exploration only).")
        return stats

    def preprocess(self) -> AnnData:
        """Step 3b: filter bins / genes, raw counts -> layers['counts'], log-normalize -> layers['lognorm']."""
        self.adata = self.preprocessor.filter_and_normalize(self.adata)
        return self.adata

    def technical_qc(self) -> dict:
        """Step 4: marker-independent QC on the filtered bins (depth maps, high-pass map, section depth)."""
        return self.technical.run(self.adata)

    def spatial_structure(self) -> pd.DataFrame:
        """Step 5: per marker set score + Moran's I + null test; baselines; index table; summary chart."""
        adata, marker_sets = self.adata, self.config.marker_sets
        self.null_results = []
        self.params.add("SignatureAnalysis", settings_of(self.signature))
        self.params.add("RandomGeneSetNullTest", settings_of(self.null_test))
        for celltype, markers in marker_sets.items():
            self.signature.run(adata, markers, celltype)
            self.null_results.append(self.null_test.run(adata, markers, celltype))
        checks = DepthConfoundChecks(self.figdir)
        checks.total_counts_baseline(adata)
        index = MoranIndexTable(self.figdir)
        self.params.add("MoranIndexTable", settings_of(index))
        index_table = index.aggregate(self.null_results)
        log.debug("\n" + index_table[["celltype", "n_tested", "marker_coverage", "observed_I", "z_score",
                                      "observed_I_resid", "z_score_resid", "depth_R2", "fdr_qvalue",
                                      "fdr_qvalue_resid", "qc_verdict"]].to_string(index=False))
        sfx = "_resid" if "z_score_resid" in index_table else ""
        for r in index_table.itertuples(index=False):            # one line per marker set
            r = r._asdict()
            log.info(f"  {r['celltype']:<16} {r['qc_verdict']:<18} z = {r['z_score' + sfx]:.1f}, "
                     f"q = {r['fdr_qvalue' + sfx]:.3g} (markers {r['marker_coverage']:.0%})")
        for celltype in marker_sets:
            checks.score_depth_correlation(adata, celltype)
        plotter = NullDistributionPlotter(self.figdir)
        for which in ("raw", "resid"):
            for celltype in marker_sets:
                plotter.plot(celltype, which=which)
            plotter.plot_combined(list(marker_sets), which=which)
        summary = MoranSummary(self.figdir)
        self.params.add("MoranSummary", settings_of(summary))
        summary.run()
        return index_table

    def niches(self) -> None:
        """Step 6: high-confidence niche DE for every marker set; any-marker region DE for panel_regions."""
        c = self.config
        common = dict(plotter=self.plotter, top_n=c.niche.top_n, seed=c.statistics.seed)
        for celltype in c.marker_sets:
            de = NicheDE(self.figdir, celltype, quantile=c.niche.quantile, **common)
            self.params.add("NicheDE", {**settings_of(de), **settings_of(de.matcher)}, call=celltype)
            degs, top = de.run(self.adata)
            self._log_region(f"{celltype}_niche", degs, top)
        for celltype in c.panel_regions:
            de = MarkerPanelDE(self.figdir, celltype, c.marker_sets[celltype], **common)
            self.params.add("MarkerPanelDE", {**settings_of(de), **settings_of(de.matcher)}, call=celltype)
            degs, top = de.run(self.adata)
            self._log_region(f"{celltype}_enriched_region", degs, top)

    def _log_region(self, name: str, degs: pd.DataFrame, top: pd.DataFrame) -> None:
        """One terminal line per niche / region; the top-gene table goes to the log file."""
        log.debug(f"{name}:\n" + top[["names", "direction", "logfoldchanges", "pvals_adj"]].to_string())
        dc = self.figdir / f"{name}_depth_check.csv"
        stages = pd.read_csv(dc).set_index("stage") if dc.exists() else pd.DataFrame()
        bits = []
        if "before_matching" in stages.index:
            bits.append(f"{int(stages.loc['before_matching', 'n_region']):,} bins")
        if "after_matching" in stages.index:
            bits.append(f"depth AUROC after matching {stages.loc['after_matching', 'depth_auroc']:.2f}")
        sig = degs[degs["significant"]] if "significant" in degs else degs.iloc[0:0]
        n_up = int((sig["logfoldchanges"] > 0).sum()) if len(sig) else 0
        bits.append(f"{len(sig)} significant DE genes ({n_up} up)")
        log.info(f"  {name:<28} " + "; ".join(bits))

    def colocalization(self) -> pd.DataFrame | None:
        """Step 7: co-localization per cell-type pair (+ by distance); within-section control; FDR table."""
        c, adata, ms = self.config, self.adata, self.config.marker_sets
        if not c.pairs:
            log.info("No celltype_pairs - co-localization skipped.")
            return None
        test = SignatureColocalization(self.figdir, n_null=c.statistics.n_null, seed=c.statistics.seed,
                                       residualize=True, include_self=True)
        self.params.add("SignatureColocalization", settings_of(test))
        self.coloc_results = []
        for a, b in c.pairs:
            self.coloc_results.append(test.test(adata, ms[a], ms[b], a, b))
            test.by_distance(adata, ms[a], ms[b], a, b, n_null=c.statistics.n_null_distance,
                             um_per_bin=c.grid.um_per_bin)

        if adata.uns.get("technical_qc", {}).get("tissue_qc", False):
            control = SignatureColocalization(self.figdir, n_null=c.statistics.n_null, seed=c.statistics.seed,
                                              residualize=True, include_self=True,
                                              compartment_col="tissue_section", tag_suffix="_within_section")
            self.params.add("SignatureColocalization", settings_of(control), call="within_section control")
            rows = []
            for a, b in c.pairs:
                r = control.test(adata, ms[a], ms[b], a, b)
                r["control"] = "tissue_section"
                rows.append(r)
            ctl = pd.DataFrame(rows)
            ctl.to_csv(self.figdir / "colocalization_technical_controls.csv", index=False)
            log.debug("\n" + ctl[["pair", "control", "I_AB", "z_swapA", "z_swapB", "p_colocalized",
                                  "verdict"]].to_string(index=False))

        table = ColocalizationTable(self.figdir).aggregate(self.coloc_results)
        log.debug("\n" + table[["pair", "n_tested", "I_AB", "I_AB_no_depth_correction", "z_swapA", "z_swapB",
                                "q_colocalized", "q_segregated", "verdict_fdr"]].to_string(index=False))
        for r in table.itertuples(index=False):                  # one line per pair
            log.info(f"  {r.pair:<28} {r.verdict_fdr:<32} I_AB = {r.I_AB:.3f}, q = {r.q_colocalized:.3g}")
        return table

    def annotate(self) -> pd.DataFrame:
        """Step 8: cell-type annotation of every bin from the high-confidence niches."""
        a = self.config.annotation
        annotator = CellTypeAnnotator(self.figdir, list(self.config.marker_sets), min_bins=a.min_bins,
                                      palette=a.palette, max_plot_labels=a.max_plot_labels, plotter=self.plotter)
        self.params.add("CellTypeAnnotator", settings_of(annotator))
        return annotator.annotate(self.adata)

    def save(self) -> Path:
        """Step 9a: the analysis h5ad (input of the gene-pair step)."""
        self.params.to_uns(self.adata)
        return AnalysisStore(self.config.analysis_h5ad_path).save(self.adata)

    # ------------------------------------------------------------------ all
    def run(self) -> AnnData:
        old_figdir = sc.settings.figdir
        sc.settings.figdir = self.figdir            # scanpy's own plots (save=...) land in figdir too
        try:
            log.info(f"MarQual-ST run: sample {self.config.sample}, platform {self.config.platform}, "
                     f"marker sets {list(self.config.marker_sets)}")
            out = Path(self.config.outdir)            # every setting of this run, defaults included: pass
            self.config.to_csv(out / USED_PARAMS_FILE, out / USED_MARKERS_FILE)   # both back to repeat it
            with self.step("1 Load data"):
                self.load()
            with self.step("2 Tissue sections"):
                self.detect_sections()
            with self.step("3 Data-quality gate + filter + normalize"):
                self.quality_gate()
                self.preprocess()
            with self.step("4 Technical QC"):
                self.technical_qc()
            with self.step("5 Spatial structure (Moran's I)"):
                self.spatial_structure()
            with self.step("6 Niches and DE"):
                self.niches()
            with self.step("7 Co-localization"):
                self.colocalization()
            with self.step("8 Cell-type annotation"):
                self.annotate()
            with self.step("9 Save + report"):
                self.save()
                self.build_report()
        finally:
            sc.settings.figdir = old_figdir
        log.info("Step times (s): " + ", ".join(f"{k}: {v:.0f}" for k, v in self.timings.items()))
        return self.adata


class GenePairPipeline(_Workflow):
    """Optional follow-up on the saved analysis h5ad: gene-pair co-localization, then the report again."""

    MERGE_PARAMS = True                        # keep the main run's parameters in run_parameters.csv

    def run(self, pairs: list[list[str]] | None = None) -> dict:
        c = self.config
        pairs = pairs if pairs is not None else c.gene_pairs.pairs
        if not pairs:
            raise ValueError("No gene pairs - set gene_pairs,pairs in params.csv (or pass --pair GENE_A GENE_B).")
        outdir = c.gene_pair_dir
        outdir.mkdir(parents=True, exist_ok=True)
        with self.step("Load analysis h5ad"):
            adata = AnalysisStore(c.analysis_h5ad_path).load()
        analysis = GenePairAnalysis(outdir, permutations=c.gene_pairs.permutations, seed=c.statistics.seed,
                                    plotter=SpatialPlotter(outdir, show=c.display_plots))
        self.params.add("GenePairAnalysis", {**settings_of(analysis), "pairs": ["_".join(p) for p in pairs]})
        results = {}
        with self.step("Gene pairs"):
            for genes in pairs:
                res = analysis.run(adata, genes)
                if res is not None:
                    results["_".join(genes)] = res
        stopped = [p for p, r in results.items() if r.get("status") == "STOP"]
        log.info(f"Done: {len(results) - len(stopped)} gene pair(s) analysed"
                 + (f", {len(stopped)} STOPPED ({', '.join(stopped)}; reason in the report)" if stopped else "")
                 + f"; outputs in {outdir}")
        with self.step("Report"):
            self.build_report()
        return results
