"""
Every parameter a run used, for the "Run parameters" section of the report.

``RunParameters`` collects the resolved config (settings set in params.csv AND the defaults it
left unchanged), the settings of every analysis object the pipeline built, and the software
versions; it writes them to ``{figdir}/run_parameters.csv`` (columns step, call, parameter,
value) and to ``adata.uns["run_parameters"]``.
"""
from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import pandas as pd

FILE = "run_parameters.csv"
UNS_KEY = "run_parameters"
COLUMNS = ["step", "call", "parameter", "value"]
_SIMPLE = (str, int, float, bool, type(None), Path)


def fmt(v: Any) -> str:
    if isinstance(v, (list, tuple, set)):
        return ", ".join(fmt(x) for x in v)
    if isinstance(v, dict):
        return "; ".join(f"{k}: {fmt(x)}" for k, x in v.items()) if v else "{}"
    return str(v)


def _is_simple(v: Any) -> bool:
    if isinstance(v, _SIMPLE):
        return True
    if isinstance(v, (list, tuple, set)):
        return all(isinstance(x, _SIMPLE) for x in v)
    if isinstance(v, dict):
        return all(isinstance(x, _SIMPLE) or _is_simple(x) for x in v.values())
    return False


def settings_of(obj: Any) -> dict[str, Any]:
    """Public, simple-valued attributes of an analysis object (its settings)."""
    return {k: v for k, v in vars(obj).items() if not k.startswith("_") and _is_simple(v)}


def software_versions() -> dict[str, str]:
    from . import __version__
    out = {"marqual_st": __version__, "python": platform.python_version()}
    from importlib.metadata import PackageNotFoundError, version
    for mod in ("scanpy", "squidpy", "anndata", "numpy", "pandas", "scipy", "statsmodels", "matplotlib",
                "esda", "libpysal"):
        try:
            out[mod] = version(mod)
        except PackageNotFoundError:
            pass
    return out


class RunParameters:
    """Ordered (step, call) -> {parameter: value} record of one run."""

    def __init__(self):
        self._rows: dict[tuple[str, str], dict[str, str]] = {}

    def add(self, step: str, params: dict[str, Any] | Any, call: str = "") -> None:
        if not isinstance(params, dict):
            params = settings_of(params)
        self._rows[(step, call)] = {k: fmt(v) for k, v in params.items()}

    def add_config(self, config) -> None:
        """The resolved config: every setting, whether set in params.csv or left at its default."""
        d = config.to_dict()
        marker_sets = d.pop("marker_sets")
        sections = {k: d.pop(k) for k in list(d) if isinstance(d[k], dict)}
        self.add("config: run", d)
        for ct, genes in marker_sets.items():
            self.add("marker_sets", {"marker_list": genes}, call=ct)
        for name, values in sections.items():
            self.add(f"config: {name}", values)
        self.add("software", software_versions())

    def table(self) -> pd.DataFrame:
        rows = [{"step": s, "call": c, "parameter": p, "value": v}
                for (s, c), params in self._rows.items() for p, v in params.items()]
        return pd.DataFrame(rows, columns=COLUMNS)

    def write(self, folder: str | Path, merge: bool = True) -> Path:
        """Write {folder}/run_parameters.csv; with ``merge``, steps from an earlier write that were not
        recorded again (e.g. the main run, when the gene-pair step writes) are kept."""
        path = Path(folder) / FILE
        new = self.table()
        if merge and path.exists():
            old = pd.read_csv(path, dtype=str, keep_default_na=False)
            done = set(zip(new["step"], new["call"]))
            old = old[[(s, c) not in done for s, c in zip(old["step"], old["call"])]]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(path, index=False)
        return path

    def to_uns(self, adata) -> None:
        adata.uns[UNS_KEY] = {f"{s}|{c}": dict(p) for (s, c), p in self._rows.items()}
