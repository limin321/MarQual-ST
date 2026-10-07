"""
Command-line interface. A run is set by two CSV files: params.csv and marker_sets.csv.

  marqual-st init-config  [-o DIR]                         write the two templates to edit
  marqual-st validate     -p params.csv -m marker_sets.csv  check them without running
  marqual-st run          -p params.csv -m marker_sets.csv  the main pipeline
  marqual-st gene-pairs   -p ... -m ... [--pair A B]        optional gene-pair follow-up (+ report)
  marqual-st annotate     -p ... -m ...                     re-annotate the saved h5ad (new min_bins / palette)
  marqual-st report       -p params.csv | --figdir DIR [--sample S]   rebuild the HTML report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from ._logging import LOG_FILE, configure_logging, get_logger
from .config import MARKERS_FILE, PARAMS_FILE, ConfigError, PipelineConfig, read_params
from .io import DataQualityStop

log = get_logger(__name__)


def _config(args) -> PipelineConfig:
    return PipelineConfig.from_csv(args.params, args.markers)


def _start(args, need_input: bool) -> PipelineConfig:
    """Settings, then logging - before scanpy / squidpy are imported, so their import-time warnings
    go to {outdir}/marqual_st.log instead of the terminal."""
    cfg = _config(args).check_paths(need_input=need_input)
    configure_logging(cfg.log_level, Path(cfg.outdir) / LOG_FILE)
    return cfg


def cmd_init_config(args) -> int:
    folder = Path(args.output)
    params, markers = folder / PARAMS_FILE, folder / MARKERS_FILE
    existing = [str(f) for f in (params, markers) if f.exists()]
    if existing and not args.force:
        print(f"{', '.join(existing)} exist(s) - use --force to overwrite.", file=sys.stderr)
        return 1
    PipelineConfig.example().to_csv(params, markers)
    print(f"Wrote {params} and {markers}. Edit input, outdir and the marker sets, then:\n"
          f"  marqual-st run -p {params} -m {markers}")
    return 0


def cmd_validate(args) -> int:
    cfg = _config(args).check_paths()
    print(f"OK: sample {cfg.sample}, platform {cfg.platform}, {len(cfg.marker_sets)} marker sets "
          f"({', '.join(cfg.marker_sets)}), {len(cfg.pairs)} cell-type pair(s), "
          f"{len(cfg.gene_pairs.pairs)} gene pair(s).\n    input   {cfg.input}\n    outputs {cfg.figdir}")
    return 0


def cmd_run(args) -> int:
    cfg = _start(args, need_input=True)
    from .pipeline import MarQualPipeline
    MarQualPipeline(cfg).run()
    return 0


def cmd_gene_pairs(args) -> int:
    cfg = _start(args, need_input=False)
    from .pipeline import GenePairPipeline
    GenePairPipeline(cfg).run(pairs=args.pair or None)
    return 0


def cmd_report(args) -> int:
    from .report import ReportBuilder
    if args.params:
        d = read_params(args.params)                    # only outdir and sample are needed
        if "outdir" not in d:
            raise ConfigError(f"{args.params}: missing required setting outdir (section run).")
        figdir, sample = Path(d["outdir"]) / "figures", args.sample or d.get("sample")
    elif args.figdir:
        figdir, sample = args.figdir, args.sample
    else:
        print("Give -p params.csv or --figdir DIR.", file=sys.stderr)
        return 2
    configure_logging("INFO")
    print(ReportBuilder(figdir, sample=sample).build())
    return 0


def cmd_annotate(args) -> int:
    cfg = _start(args, need_input=False)
    from .annotation import CellTypeAnnotator
    from .spatial import SpatialPlotter
    a = cfg.annotation
    CellTypeAnnotator(cfg.figdir, list(cfg.marker_sets), min_bins=a.min_bins, palette=a.palette,
                      max_plot_labels=a.max_plot_labels,
                      plotter=SpatialPlotter(cfg.figdir)).annotate_h5ad(cfg.analysis_h5ad_path)
    return 0


def _add_inputs(s: argparse.ArgumentParser) -> None:
    s.add_argument("-p", "--params", required=True, help="params.csv (settings; see marqual-st init-config)")
    s.add_argument("-m", "--markers", required=True, help="marker_sets.csv (one row per cell type: name, genes)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="marqual-st", description="MarQual-ST: marker-based quality control and "
                                "niche discovery for binned spatial transcriptomics.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init-config", help="write the templates params.csv and marker_sets.csv")
    s.add_argument("-o", "--output", default=".", help="folder for the two files (default: here)")
    s.add_argument("--force", action="store_true", help="overwrite existing files")
    s.set_defaults(func=cmd_init_config)

    for name, func, hlp in (("validate", cmd_validate, "check params.csv + marker_sets.csv"),
                            ("run", cmd_run, "run the main pipeline"),
                            ("annotate", cmd_annotate, "re-annotate the saved analysis h5ad")):
        s = sub.add_parser(name, help=hlp)
        _add_inputs(s)
        s.set_defaults(func=func)

    s = sub.add_parser("gene-pairs", help="optional gene-pair co-localization on the saved analysis h5ad")
    _add_inputs(s)
    s.add_argument("--pair", nargs="+", action="append", metavar="GENE",
                   help="gene pair (repeatable); overrides gene_pairs,pairs in params.csv")
    s.set_defaults(func=cmd_gene_pairs)

    s = sub.add_parser("report", help="rebuild the HTML report from a figures folder")
    s.add_argument("-p", "--params", help="params.csv (figures folder and sample from it)")
    s.add_argument("--figdir", help="figures folder (instead of -p)")
    s.add_argument("--sample", help="sample name (report file name)")
    s.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DataQualityStop as e:
        print(f"STOPPED (data quality): {e}", file=sys.stderr)
        return 3
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    except (FileNotFoundError, KeyError, ValueError) as e:
        log.error(str(e))
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
