"""
MarQual-ST: marker-based quality control and niche discovery for binned spatial
transcriptomics (Stereo-seq, Visium HD).

Typical use::

    from marqual_st import MarQualPipeline
    adata = MarQualPipeline.from_csv("params.csv", "marker_sets.csv").run()

or on the command line: ``marqual-st run -p params.csv -m marker_sets.csv``.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .config import ConfigError, PipelineConfig  # noqa: E402

__all__ = [
    "__version__",
    "ConfigError",
    "PipelineConfig",
    "MarQualPipeline",
    "GenePairPipeline",
    "ReportBuilder",
]


def __getattr__(name):
    # heavy imports (scanpy, squidpy) only when the workflow classes are used
    if name in ("MarQualPipeline", "GenePairPipeline"):
        from . import pipeline
        return getattr(pipeline, name)
    if name == "ReportBuilder":
        from .report import ReportBuilder
        return ReportBuilder
    raise AttributeError(f"module 'marqual_st' has no attribute {name!r}")
