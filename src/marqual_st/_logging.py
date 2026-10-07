"""
Logging setup shared by all MarQual-ST modules.

Two outputs:
* the terminal shows progress and results that need attention: step headers, one line per
  marker set / niche / pair, warnings, the report path (level ``log_level``, default INFO);
* ``{outdir}/marqual_st.log`` keeps everything for troubleshooting: the same lines plus the DEBUG
  details (result tables, files written, depth checks, per-test numbers) and the warnings of
  the libraries (deprecation notices etc.), with the module that wrote each line.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PACKAGE_LOGGER = "marqual_st"
LOG_FILE = "marqual_st.log"                      # in {outdir}
_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
# library loggers that print INFO chatter to the terminal on their own (squidpy builds its spatial
# graph through spatialdata's logger: "Creating graph using ... transform")
_QUIET_LIBRARIES = ("spatialdata._logging",)


class _ConsoleFormatter(logging.Formatter):
    """'HH:MM:SS message'; warnings and errors keep their level so they stand out."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        t = self.formatTime(record, "%H:%M:%S")
        return f"{t} {record.levelname}: {msg}" if record.levelno >= logging.WARNING else f"{t} {msg}"


def get_logger(name: str) -> logging.Logger:
    """Logger for a module of the package (a child of the ``marqual_st`` logger)."""
    return logging.getLogger(name if name.startswith(PACKAGE_LOGGER) else f"{PACKAGE_LOGGER}.{name}")


def _mark(h: logging.Handler) -> logging.Handler:
    h._marqual_st = True  # type: ignore[attr-defined]
    return h


def _drop_handlers(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        if getattr(h, "_marqual_st", False):
            logger.removeHandler(h)
            h.close()


def configure_logging(level: int | str = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """
    Terminal at ``level`` (short lines); ``log_file`` (if given) at DEBUG with library warnings.

    Safe to call more than once: handlers added by an earlier call are replaced,
    so a notebook re-running a pipeline does not print every line twice.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    _drop_handlers(logger)
    logger.setLevel(logging.DEBUG)                     # the handlers decide what goes where
    console = _mark(logging.StreamHandler(sys.stdout))
    console.setLevel(level)
    console.setFormatter(_ConsoleFormatter("%(message)s"))
    logger.addHandler(console)
    logger.propagate = False

    # Python warnings of the libraries (FutureWarning, DeprecationWarning ...) -> log file only
    logging.captureWarnings(True)
    pyw = logging.getLogger("py.warnings")
    _drop_handlers(pyw)
    pyw.propagate = False
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = _mark(logging.FileHandler(log_file, encoding="utf-8"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(fh)
        pyw.addHandler(fh)
    else:
        pyw.addHandler(_mark(logging.NullHandler()))
    for name in _QUIET_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logger


logging.getLogger(PACKAGE_LOGGER).addHandler(logging.NullHandler())
