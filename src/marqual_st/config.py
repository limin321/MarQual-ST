"""
Run settings for MarQual-ST: two CSV files.

``params.csv``       columns ``section, parameter, value, description``; one row per setting. Only
                     sample, input and outdir are required; a missing row or an empty value keeps
                     the default. Relative paths are relative to the folder of params.csv.
``marker_sets.csv``  no header; one row per cell type: the cell-type name, then its marker genes
                     (one gene per cell; empty cells are ignored).

``marqual-st init-config`` writes both templates. Every run saves the resolved settings as
``{outdir}/params_used.csv`` and ``{outdir}/marker_sets_used.csv`` - pass them back to repeat it.
Nothing about a run is edited in the code, so the same installed package or Docker image serves
every sample.
"""
from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, field, fields
from itertools import combinations
from pathlib import Path
from typing import Any

PLATFORMS = ("stereo", "visiumhd")


class ConfigError(ValueError):
    """Raised when a configuration file is missing a value or has an invalid one."""


@dataclass
class GridConfig:
    """Bin grid. ``bin_size``: distance between neighbouring bins in ``obsm['spatial']`` units
    (Stereo-seq bin50: 50; not used for Visium HD, which has array_row / array_col).
    ``um_per_bin``: bin width in micrometers, only used to label distances."""
    bin_size: float | None = 50
    um_per_bin: float | None = 25


@dataclass
class FilterConfig:
    """Bin / gene filter, applied after tissue sections are found."""
    min_counts: int = 600
    min_cells: int = 6
    pct_mt: float = 20


@dataclass
class SectionConfig:
    """Tissue sections on the chip. ``tissue_qc``: treat sections as a technical variable
    (None = True when num_tissue > 1)."""
    num_tissue: int = 1
    tissue_qc: bool | None = None


@dataclass
class StatisticsConfig:
    """Random-gene-set nulls. ``n_null``: Moran's I null test and co-localization test;
    ``n_null_distance``: co-localization by distance."""
    n_null: int = 1000
    n_null_distance: int = 200
    seed: int = 0


@dataclass
class NicheConfig:
    """High-confidence niche: bins above the ``quantile`` of the smoothed signature score
    (0.95 = top 5%, default; 0.90 = top 10%); ``top_n`` DE genes per direction."""
    quantile: float = 0.95
    top_n: int = 20


@dataclass
class QualityConfig:
    """Data-quality gate on the share of bins the bin filter removes (filters.* - keep the
    recommended cutoffs fixed across samples). STOP at >= ``stop_pct_filtered`` % or fewer than
    ``min_bins_kept`` bins kept; CAUTION at >= ``caution_pct_filtered`` %. Defaults from three
    samples that ran well (2.7%, 9.8%, 23.1% filtered out)."""
    caution_pct_filtered: float = 60
    stop_pct_filtered: float = 80
    min_bins_kept: int = 2000


@dataclass
class AnnotationConfig:
    """Cell-type annotation: combinations with fewer than ``min_bins`` bins -> "Mix";
    plots use at most ``max_plot_labels`` colors; optional ``palette`` {label: color}."""
    min_bins: int = 100
    max_plot_labels: int = 15
    palette: dict[str, str] = field(default_factory=dict)


@dataclass
class GenePairConfig:
    """Optional gene-pair step: ``pairs`` of genes, permutations for bivariate Moran's I."""
    pairs: list[list[str]] = field(default_factory=list)
    permutations: int = 999


@dataclass
class PipelineConfig:
    """
    Everything one run needs. Build it with :meth:`from_csv` (params.csv + marker_sets.csv),
    or with :meth:`from_dict` in Python, and call :meth:`validate`.

    Paths: tables and figures go to ``{outdir}/figures``; the analysis h5ad (input of
    the gene-pair step) to ``{outdir}/{sample}.marker_pipeline.h5ad`` unless
    ``analysis_h5ad`` is given.
    """
    sample: str
    input: str
    outdir: str
    marker_sets: dict[str, list[str]]
    platform: str = "stereo"
    analysis_h5ad: str | None = None
    grid: GridConfig = field(default_factory=GridConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    sections: SectionConfig = field(default_factory=SectionConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    niche: NicheConfig = field(default_factory=NicheConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    gene_pairs: GenePairConfig = field(default_factory=GenePairConfig)
    panel_region_celltypes: list[str] | str = "all"
    celltype_pairs: list[list[str]] | str = field(default_factory=list)
    force: bool = False
    display_plots: bool = False
    log_level: str = "INFO"

    _SECTIONS = {"grid": GridConfig, "filters": FilterConfig, "quality": QualityConfig, "sections": SectionConfig,
                 "statistics": StatisticsConfig, "niche": NicheConfig, "annotation": AnnotationConfig,
                 "gene_pairs": GenePairConfig}

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"Unknown setting(s) {unknown}. Allowed: {sorted(k for k in known if not k.startswith('_'))}.")
        for key, section_cls in cls._SECTIONS.items():
            if key in data and data[key] is not None:
                if not isinstance(data[key], dict):
                    raise ConfigError(f"'{key}' must be a mapping of settings.")
                bad = sorted(set(data[key]) - set(section_cls.__dataclass_fields__))
                if bad:
                    raise ConfigError(f"Unknown setting(s) {bad} in '{key}'. "
                                      f"Allowed: {sorted(section_cls.__dataclass_fields__)}.")
                data[key] = section_cls(**data[key])
            else:
                data.pop(key, None)
        missing = [k for k in ("sample", "input", "outdir", "marker_sets") if k not in data]
        if missing:
            raise ConfigError(f"Missing required setting(s): {missing}.")
        return cls(**data)

    @classmethod
    def from_csv(cls, params: str | Path, markers: str | Path) -> PipelineConfig:
        """Load params.csv + marker_sets.csv and validate."""
        data = read_params(params)
        missing = [k for k in ("sample", "input", "outdir") if k not in data]
        if missing:
            raise ConfigError(f"{params}: missing required setting(s) {missing} (section run).")
        data["marker_sets"] = read_marker_sets(markers)
        return cls.from_dict(data).validate()

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    def to_csv(self, params: str | Path, markers: str | Path) -> tuple[Path, Path]:
        """Write every setting (defaults included) to params.csv and the marker sets to marker_sets.csv."""
        return write_params(self, params), write_marker_sets(self.marker_sets, markers)

    @classmethod
    def example(cls) -> PipelineConfig:
        """The settings written by ``marqual-st init-config`` (edit input, outdir, marker sets)."""
        return cls(
            sample="SAMPLE01", input="/path/to/SAMPLE01.tissue.bin50.h5ad", outdir="/path/to/results/SAMPLE01",
            marker_sets={
                "immune": ["CCL5", "CCR7", "CD2", "CD3D", "CD3E", "CD3G", "PTPRC", "CD8A", "CD8B", "CD14", "CD163",
                           "CD68", "CD1C", "CLEC10A", "FCER1A", "CD79A", "CD79B", "JCHAIN", "MS4A1", "MZB1", "GNLY",
                           "GZMB", "KLRD1", "NKG7", "TPSAB1", "TPSB2", "CPA3", "KIT", "MRC1", "IL1B", "NDST2",
                           "MS4A2", "HPGDS", "GATA2"],
                "neuron": ["GPC5", "CNTN5", "CPNE4", "RBFOX3", "TUBB3", "MAP2", "UCHL1", "NEFL", "NEFM", "NEFH",
                           "STMN2", "SNAP25", "SYT1", "SYP", "ELAVL3", "ELAVL4", "PRPH", "GAP43", "ULK4", "FAM155A"],
            },
            celltype_pairs=[["immune", "neuron"]], gene_pairs=GenePairConfig(pairs=[["GNLY", "ATP8A2"]]))

    # ------------------------------------------------------------------ checks
    def validate(self) -> PipelineConfig:
        """Check values and cross-references; returns self (raises :class:`ConfigError`)."""
        if self.platform not in PLATFORMS:
            raise ConfigError(f"platform must be one of {PLATFORMS}, got {self.platform!r}.")
        if not self.marker_sets or not isinstance(self.marker_sets, dict):
            raise ConfigError("marker_sets must map each cell type to a list of marker genes.")
        for ct, genes in self.marker_sets.items():
            if not isinstance(genes, (list, tuple)) or not genes:
                raise ConfigError(f"marker_sets[{ct!r}] must be a non-empty list of genes.")
        if self.platform == "stereo" and not self.grid.bin_size:
            raise ConfigError("grid.bin_size is required for platform 'stereo'.")
        if self.sections.num_tissue < 1:
            raise ConfigError("sections.num_tissue must be >= 1.")
        if not 0 < self.niche.quantile < 1:
            raise ConfigError("niche.quantile must be between 0 and 1 (0.95 = top 5%, 0.90 = top 10%).")
        q = self.quality
        if not 0 <= q.caution_pct_filtered <= q.stop_pct_filtered <= 100:
            raise ConfigError("quality: need 0 <= caution_pct_filtered <= stop_pct_filtered <= 100.")
        if self.annotation.max_plot_labels is not None and self.annotation.max_plot_labels < 2:
            raise ConfigError("annotation.max_plot_labels must be >= 2.")
        if self.celltype_pairs != "all":
            for pair in self.celltype_pairs or []:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ConfigError(f"celltype_pairs entries must be [celltype_a, celltype_b], got {pair!r}.")
        for name, used in (("celltype_pairs", [c for pair in self.pairs for c in pair]),
                           ("panel_region_celltypes", self.panel_regions)):
            unknown = sorted(set(used) - set(self.marker_sets))
            if unknown:
                raise ConfigError(f"params.csv row run,{name} names cell type(s) {unknown} that are not in the "
                                  f"marker sets ({', '.join(self.marker_sets)}). Edit that row (names must match "
                                  f"marker_sets.csv exactly, e.g. NK:Neuron; Mast:Fibroblasts), or leave it empty.")
        for pair in self.gene_pairs.pairs:
            if len(pair) < 2:
                raise ConfigError(f"gene_pairs.pairs entries need 2 or more genes, got {pair}.")
        return self

    def check_paths(self, need_input: bool = True) -> PipelineConfig:
        """Before anything runs: the input exists and the output folder can be created."""
        hint = ("Edit it in params.csv (section run). In Docker, use the container paths of the mounted "
                "folders, e.g. /data/...")
        if need_input and not Path(self.input).exists():
            raise ConfigError(f"input not found: {self.input}. {hint}")
        out = Path(self.outdir).expanduser().absolute()
        parent = out
        while not parent.exists() and parent != parent.parent:   # nearest folder that exists
            parent = parent.parent
        if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
            raise ConfigError(f"outdir {self.outdir} cannot be created: no write permission in {parent}. {hint}")
        return self

    # ------------------------------------------------------------------ derived values
    @property
    def figdir(self) -> Path:
        return Path(self.outdir) / "figures"

    @property
    def analysis_h5ad_path(self) -> Path:
        return Path(self.analysis_h5ad) if self.analysis_h5ad else Path(self.outdir) / f"{self.sample}.marker_pipeline.h5ad"

    @property
    def gene_pair_dir(self) -> Path:
        return self.figdir / "gene_pairs"

    @property
    def panel_regions(self) -> list[str]:
        """Cell types that get the any-marker region DE."""
        if self.panel_region_celltypes == "all" or self.panel_region_celltypes is None:
            return list(self.marker_sets)
        return list(self.panel_region_celltypes)

    @property
    def pairs(self) -> list[tuple[str, str]]:
        """Cell-type pairs for co-localization."""
        if self.celltype_pairs == "all":
            return list(combinations(self.marker_sets, 2))
        return [tuple(p) for p in (self.celltype_pairs or [])]


# ---------------------------------------------------------------------------
# CSV files
# ---------------------------------------------------------------------------
PARAMS_FILE, MARKERS_FILE = "params.csv", "marker_sets.csv"            # init-config templates
USED_PARAMS_FILE, USED_MARKERS_FILE = "params_used.csv", "marker_sets_used.csv"   # saved with every run
PARAM_COLUMNS = ["section", "parameter", "value", "description"]
RUN_SECTION = "run"
LIST_SEP, PAIR_SEP = ";", ":"

# (section, parameter) -> description, in the order written to params.csv
DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("run", "sample"): "Sample name (file names, report title).",
    ("run", "input"): "EDIT. UNFILTERED tissue-cut input - stereo: tissue-cut Stereo-seq h5ad; visiumhd: Space Ranger "
                      "binned_outputs/square_XXXum folder. A relative path is relative to the folder of this file.",
    ("run", "outdir"): "EDIT. Run folder, created if missing: tables + figures -> {outdir}/figures; analysis h5ad -> "
                       "{outdir}/{sample}.marker_pipeline.h5ad.",
    ("run", "platform"): "stereo or visiumhd.",
    ("run", "analysis_h5ad"): "Optional other path for the analysis h5ad (empty = {outdir}/{sample}.marker_pipeline.h5ad).",
    ("run", "panel_region_celltypes"): "Cell types that get the any-marker region DE: all, or cell types from "
                                       "marker_sets.csv separated by ; (slow on large regions).",
    ("run", "celltype_pairs"): "Co-localization pairs A:B separated by ; (e.g. NK:Neuron; Mast:Fibroblasts), or all "
                               "for every pair; empty = none.",
    ("run", "force"): "true = continue after a data-quality STOP (exploration only; the verdict stays STOP).",
    ("run", "display_plots"): "true = also show plots interactively (notebooks).",
    ("run", "log_level"): "Terminal detail: INFO (progress + one line per result), WARNING (warnings only) or DEBUG "
                          "(everything). {outdir}/marqual_st.log always has everything.",
    ("grid", "bin_size"): "Distance between neighbouring bins in obsm['spatial'] units (Stereo-seq bin50: 50; not "
                          "used for Visium HD).",
    ("grid", "um_per_bin"): "Bin width in um, only labels distances (Stereo-seq bin50: 25; Visium HD 8 um: 8).",
    ("filters", "min_counts"): "Remove bins with total counts < min_counts. 600 / 6 / 20 is the recommended Stereo-seq "
                               "bin50 filter - keep it fixed across samples.",
    ("filters", "min_cells"): "Remove genes detected in < min_cells bins.",
    ("filters", "pct_mt"): "Remove bins with mitochondrial % >= pct_mt.",
    ("quality", "caution_pct_filtered"): "Data-quality gate: CAUTION when >= this % of bins is filtered out (the run "
                                         "continues; warning in the report). Samples that ran well lost 2.7%, 9.8%, 23.1%.",
    ("quality", "stop_pct_filtered"): "STOP when >= this % of bins is filtered out: the run ends after technical QC "
                                      "(maps of the UNFILTERED bins) + report.",
    ("quality", "min_bins_kept"): "Also STOP when fewer bins than this are left after the filter.",
    ("sections", "num_tissue"): "Tissue sections on the chip; > 1: the N largest pieces -> obs['tissue_section'] (S1, S2 ...).",
    ("sections", "tissue_qc"): "Treat sections as a technical variable: true or false; empty = true when num_tissue > 1.",
    ("statistics", "n_null"): "Random gene sets for the Moran's I null test and the co-localization test.",
    ("statistics", "n_null_distance"): "Random gene sets for co-localization by distance.",
    ("statistics", "seed"): "Random seed.",
    ("niche", "quantile"): "High-confidence niche = bins above this quantile of the smoothed score (0.95 = top 5%; "
                           "0.90 = top 10%).",
    ("niche", "top_n"): "Top DE genes per direction (tables, dot plot).",
    ("annotation", "min_bins"): "Combinations of cell types with fewer bins -> Mix.",
    ("annotation", "max_plot_labels"): "Plots use at most this many colors (Mix included).",
    ("annotation", "palette"): "Optional colors label=color separated by ; (e.g. NK=#1f77b4; Mix=#9467bd); empty = "
                               "default colors.",
    ("gene_pairs", "pairs"): "Optional gene-pair step (marqual-st gene-pairs): gene pairs A:B separated by ; (e.g. "
                             "GNLY:ATP8A2; CD3E:CD8A).",
    ("gene_pairs", "permutations"): "Permutations for the bivariate Moran's I of the gene-pair step.",
}
# settings that are lists / mappings in one CSV cell
_SPECIAL = {("run", "panel_region_celltypes"): "names", ("run", "celltype_pairs"): "pairs",
            ("gene_pairs", "pairs"): "pairs", ("annotation", "palette"): "mapping"}
_PATHS = ("input", "outdir", "analysis_h5ad")


def _param_types() -> dict[str, dict[str, str]]:
    """{section: {parameter: annotation}} of every setting (run = the top-level settings)."""
    out = {RUN_SECTION: {f.name: str(f.type) for f in fields(PipelineConfig)
                         if not f.name.startswith("_") and f.name != "marker_sets"
                         and f.name not in PipelineConfig._SECTIONS}}
    for name, section_cls in PipelineConfig._SECTIONS.items():
        out[name] = {f.name: str(f.type) for f in fields(section_cls)}
    return out


def _split(text: str) -> list[str]:
    return [t.strip() for t in text.split(LIST_SEP) if t.strip()]


def _parse(section: str, name: str, text: str, typ: str) -> Any:
    """One CSV cell -> the setting's value (raises ValueError with what was expected)."""
    kind = _SPECIAL.get((section, name))
    if kind == "names":
        return "all" if text.lower() == "all" else _split(text)
    if kind == "pairs":
        if text.lower() == "all" and section == RUN_SECTION:
            return "all"
        return [[g.strip() for g in item.split(PAIR_SEP) if g.strip()] for item in _split(text)]
    if kind == "mapping":
        out = {}
        for item in _split(text):
            if "=" not in item:
                raise ValueError(f"expected label=color, got {item!r}")
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
        return out
    t = typ.replace(" ", "")
    if text.lower() in ("none", "null") and "None" in t:
        return None
    if t.startswith("bool"):
        if text.lower() in ("true", "yes", "1"):
            return True
        if text.lower() in ("false", "no", "0"):
            return False
        raise ValueError("expected true or false")
    if t.startswith(("int", "float")):
        try:
            x = float(text)
        except ValueError:
            raise ValueError("expected a number") from None
        if t.startswith("int"):
            if x != int(x):
                raise ValueError("expected a whole number")
            return int(x)
        try:
            return int(text)                    # "20" stays 20 (as the default 20), "0.95" -> 0.95
        except ValueError:
            return x
    return text


def _format(section: str, name: str, value: Any) -> str:
    """A setting's value -> one CSV cell (the inverse of _parse)."""
    if value is None:
        return ""
    kind = _SPECIAL.get((section, name))
    if kind == "names":
        return value if isinstance(value, str) else "; ".join(value)
    if kind == "pairs":
        return value if isinstance(value, str) else "; ".join(PAIR_SEP.join(p) for p in value)
    if kind == "mapping":
        return "; ".join(f"{k}={v}" for k, v in value.items())
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _rows(path: Path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:          # utf-8-sig: files saved by Excel
        return [[c.strip() for c in row] for row in csv.reader(fh)]


def read_params(path: str | Path) -> dict[str, Any]:
    """params.csv -> settings dict for :meth:`PipelineConfig.from_dict` (without marker_sets)."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"params file not found: {path}")
    rows = _rows(path)
    if not rows or [c.lower() for c in rows[0][:3]] != PARAM_COLUMNS[:3]:
        raise ConfigError(f"{path}: the first row must be the header: {','.join(PARAM_COLUMNS)}")
    types, data, seen = _param_types(), {}, {}
    for n, row in enumerate(rows[1:], start=2):
        section, name, text = (row + ["", "", ""])[:3]
        if not section and not name or section.startswith("#"):
            continue                                                  # blank or comment row
        where = f"{path.name} line {n}"
        if section not in types:
            raise ConfigError(f"{where}: unknown section {section!r}. Sections: {list(types)}.")
        if name not in types[section]:
            raise ConfigError(f"{where}: unknown parameter {name!r} in section {section!r}. "
                              f"Allowed: {sorted(types[section])}.")
        if (section, name) in seen:
            raise ConfigError(f"{where}: {section}.{name} is already set on line {seen[section, name]}.")
        seen[section, name] = n
        if text == "":
            continue                                                  # empty value = default
        try:
            value = _parse(section, name, text, types[section][name])
        except ValueError as e:
            raise ConfigError(f"{where}: {section}.{name} = {text!r}: {e}.") from None
        if section == RUN_SECTION:
            data[name] = value
        else:
            data.setdefault(section, {})[name] = value
    for key in _PATHS:                                                # relative to the folder of params.csv
        if isinstance(data.get(key), str):
            p = Path(data[key]).expanduser()
            data[key] = str(p) if p.is_absolute() else os.path.normpath(path.parent.absolute() / p)
    return data


def read_marker_sets(path: str | Path) -> dict[str, list[str]]:
    """marker_sets.csv -> {cell type: [genes]}. One row per cell type: name, then genes; no header
    (a first row starting with celltype / cell_type is skipped); empty cells are ignored; a gene listed
    twice in one set is kept once."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"marker sets file not found: {path}")
    sets: dict[str, list[str]] = {}
    first: dict[str, int] = {}
    for n, row in enumerate(_rows(path), start=1):
        cells = [c for c in row if c]
        if not cells or cells[0].startswith("#"):
            continue
        if n == 1 and cells[0].lower().replace(" ", "_") in ("celltype", "cell_type"):
            continue                                                  # optional header row
        ct, genes = cells[0], list(dict.fromkeys(cells[1:]))
        if ct in sets:
            raise ConfigError(f"{path.name} line {n}: cell type {ct!r} is already defined on line {first[ct]}.")
        if not genes:
            raise ConfigError(f"{path.name} line {n}: cell type {ct!r} has no marker genes.")
        sets[ct], first[ct] = genes, n
    if not sets:
        raise ConfigError(f"{path}: no marker sets (one row per cell type: name, gene, gene, ...).")
    return sets


def write_params(config: PipelineConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = config.to_dict()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(PARAM_COLUMNS)
        for (section, name), desc in DESCRIPTIONS.items():
            value = d[name] if section == RUN_SECTION else d[section][name]
            if name in _PATHS and section == RUN_SECTION and value:
                value = os.path.abspath(value)                        # usable from any folder
            w.writerow([section, name, _format(section, name, value), desc])
    return path


def write_marker_sets(marker_sets: dict[str, list[str]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for ct, genes in marker_sets.items():
            w.writerow([ct, *genes])
    return path
