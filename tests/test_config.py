from pathlib import Path

import pytest

from conftest import base_config
from marqual_st.config import ConfigError, PipelineConfig


def test_defaults_and_derived_paths(tmp_path):
    cfg = PipelineConfig.from_dict(base_config("in.h5ad", tmp_path)).validate()
    assert cfg.figdir == tmp_path / "figures"
    assert cfg.analysis_h5ad_path == tmp_path / "SYN.marker_pipeline.h5ad"
    assert cfg.gene_pair_dir == tmp_path / "figures" / "gene_pairs"
    assert cfg.pairs == [("immune", "neuron"), ("immune", "epi"), ("neuron", "epi")]
    assert cfg.panel_regions == ["immune", "neuron"]
    assert cfg.filters.min_counts == 600 and cfg.grid.bin_size == 50


def test_panel_regions_all(tmp_path):
    cfg = PipelineConfig.from_dict(base_config("in.h5ad", tmp_path, panel_region_celltypes="all"))
    assert cfg.panel_regions == ["immune", "neuron", "epi"]


@pytest.mark.parametrize("override, match", [
    ({"platform": "xenium"}, "platform"),
    ({"celltype_pairs": [["immune", "tcell"]]}, r"run,celltype_pairs names cell type\(s\) \['tcell'\]"),
    ({"celltype_pairs": [["immune", "neuron", "epi"]]}, "celltype_pairs entries"),
    ({"panel_region_celltypes": ["bcell"]}, r"run,panel_region_celltypes names cell type\(s\) \['bcell'\]"),
    ({"marker_sets": {"immune": []}}, "non-empty"),
    ({"niche": {"quantile": 1.5}}, "quantile"),
    ({"quality": {"caution_pct_filtered": 90, "stop_pct_filtered": 80}}, "caution_pct_filtered"),
    ({"gene_pairs": {"pairs": [["GNLY"]]}}, "2 or more genes"),
])
def test_validate_rejects(tmp_path, override, match):
    cfg = PipelineConfig.from_dict(base_config("in.h5ad", tmp_path, **override))
    with pytest.raises(ConfigError, match=match):
        cfg.validate()


def test_unknown_and_missing_keys(tmp_path):
    with pytest.raises(ConfigError, match="Unknown setting"):
        PipelineConfig.from_dict({**base_config("in", tmp_path), "n_nul": 5})
    with pytest.raises(ConfigError, match="Unknown setting"):
        PipelineConfig.from_dict(base_config("in", tmp_path, filters={"min_count": 5}))
    d = base_config("in", tmp_path)
    d.pop("marker_sets")
    with pytest.raises(ConfigError, match="Missing"):
        PipelineConfig.from_dict(d)


def test_csv_roundtrip(tmp_path):
    cfg = PipelineConfig.from_dict(base_config(tmp_path / "in.h5ad", tmp_path, annotation={"palette": {"immune": "#1f77b4"}},
                                               sections={"num_tissue": 2, "tissue_qc": False})).validate()
    p, m = cfg.to_csv(tmp_path / "params.csv", tmp_path / "marker_sets.csv")
    again = PipelineConfig.from_csv(p, m)
    assert again.to_dict() == cfg.to_dict()
    rel = PipelineConfig.from_dict(base_config("in.h5ad", "out"))      # relative paths are saved absolute
    p2, _ = rel.to_csv(tmp_path / "sub" / "params.csv", tmp_path / "sub" / "m.csv")
    assert PipelineConfig.from_csv(p2, tmp_path / "sub" / "m.csv").input == str(Path("in.h5ad").absolute())


def test_every_setting_has_a_description():
    from marqual_st.config import DESCRIPTIONS, _param_types
    types = _param_types()
    assert {(s, n) for s, names in types.items() for n in names} == set(DESCRIPTIONS)


def test_example_files_match_init_config(tmp_path):
    """config/params.csv + config/marker_sets.csv = what marqual-st init-config writes, and they load."""
    from marqual_st.cli import main
    root = Path(__file__).resolve().parents[1] / "config"
    assert main(["init-config", "-o", str(tmp_path)]) == 0
    for f in ("params.csv", "marker_sets.csv"):
        assert (root / f).read_text() == (tmp_path / f).read_text(), f"config/{f} is out of date"
    cfg = PipelineConfig.from_csv(root / "params.csv", root / "marker_sets.csv")
    assert cfg.pairs == [("immune", "neuron")] and cfg.gene_pairs.pairs == [["GNLY", "ATP8A2"]]


def test_params_csv_partial_and_relative(tmp_path):
    """Only the rows that differ from the defaults; relative paths are relative to params.csv."""
    (tmp_path / "markers.csv").write_text("NK,GNLY,GZMB,,\nMast,CPA3,KIT,KIT\n\nNeuron,SNAP25\n")
    (tmp_path / "p.csv").write_text(
        "section,parameter,value,description\n"
        "run,sample,S1,\nrun,input,data/in.h5ad,\nrun,outdir,out,\n"
        "run,celltype_pairs,NK:Neuron; Mast:Neuron,\nrun,panel_region_celltypes,NK;Mast,\n"
        "niche,quantile,0.9,top 10%\nsections,tissue_qc,,\nrun,force,yes,\n")
    cfg = PipelineConfig.from_csv(tmp_path / "p.csv", tmp_path / "markers.csv")
    assert cfg.marker_sets == {"NK": ["GNLY", "GZMB"], "Mast": ["CPA3", "KIT"], "Neuron": ["SNAP25"]}
    assert cfg.pairs == [("NK", "Neuron"), ("Mast", "Neuron")] and cfg.panel_regions == ["NK", "Mast"]
    assert cfg.niche.quantile == pytest.approx(0.9) and cfg.force is True
    assert cfg.sections.tissue_qc is None and cfg.filters.min_counts == 600          # defaults
    assert cfg.input == str(tmp_path / "data" / "in.h5ad") and cfg.outdir == str(tmp_path / "out")


@pytest.mark.parametrize("rows, match", [
    ("run,sample,S\nrun,input,i\n", "missing required setting"),
    ("run,sample,S\nrun,input,i\nrun,outdir,o\nfilter,min_counts,5\n", "line 5: unknown section 'filter'"),
    ("run,sample,S\nrun,input,i\nrun,outdir,o\nfilters,min_count,5\n", "unknown parameter 'min_count'"),
    ("run,sample,S\nrun,input,i\nrun,outdir,o\nniche,quantile,95%\n", "niche.quantile = '95%': expected a number"),
    ("run,sample,S\nrun,input,i\nrun,outdir,o\nfilters,min_counts,600.5\n", "expected a whole number"),
    ("run,sample,S\nrun,input,i\nrun,outdir,o\nrun,force,maybe\n", "expected true or false"),
    ("run,sample,S\nrun,input,i\nrun,outdir,o\nrun,sample,T\n", "already set on line 2"),
    ("run,sample,S\nrun,input,i\nrun,outdir,o\nrun,celltype_pairs,NK:Bcell\n", "not in the marker sets"),
])
def test_params_csv_errors(tmp_path, rows, match):
    (tmp_path / "m.csv").write_text("NK,GNLY\nNeuron,SNAP25\n")
    (tmp_path / "p.csv").write_text("section,parameter,value,description\n" + rows)
    with pytest.raises(ConfigError, match=match):
        PipelineConfig.from_csv(tmp_path / "p.csv", tmp_path / "m.csv")


@pytest.mark.parametrize("text, match", [
    ("NK,GNLY\nNK,GZMB\n", "line 2: cell type 'NK' is already defined on line 1"),
    ("NK,GNLY\nMast,,\n", "line 2: cell type 'Mast' has no marker genes"),
    ("\n\n", "no marker sets"),
])
def test_marker_csv_errors(tmp_path, text, match):
    from marqual_st.config import read_marker_sets
    (tmp_path / "m.csv").write_text(text)
    with pytest.raises(ConfigError, match=match):
        read_marker_sets(tmp_path / "m.csv")


def test_marker_csv_header_and_bom(tmp_path):
    from marqual_st.config import read_marker_sets
    (tmp_path / "m.csv").write_bytes("\ufeffcell_type,genes\nNK,GNLY,GZMB\n".encode())     # Excel: BOM + header
    assert read_marker_sets(tmp_path / "m.csv") == {"NK": ["GNLY", "GZMB"]}


def test_niche_quantile_and_quality_defaults(tmp_path):
    cfg = PipelineConfig.from_dict(base_config("in.h5ad", tmp_path, niche={"quantile": 0.90}))
    assert cfg.niche.quantile == pytest.approx(0.90)
    assert (cfg.quality.caution_pct_filtered, cfg.quality.stop_pct_filtered, cfg.quality.min_bins_kept) == (60, 80, 2000)
    assert cfg.force is False


def test_check_paths(tmp_path, monkeypatch, capsys):
    """Missing input / unwritable outdir: a one-line config error before anything runs."""
    from marqual_st import cli, config
    inp = tmp_path / "in.h5ad"
    cfg = PipelineConfig.from_dict(base_config(str(inp), tmp_path / "run"))
    with pytest.raises(ConfigError, match="input not found"):
        cfg.check_paths()
    inp.write_text("x")
    assert cfg.check_paths() is cfg                       # parent tmp_path is writable
    monkeypatch.setattr(config.os, "access", lambda *a: False)
    with pytest.raises(ConfigError, match="cannot be created: no write permission"):
        cfg.check_paths()
    p, m = cfg.to_csv(tmp_path / "params.csv", tmp_path / "marker_sets.csv")
    assert cli.main(["validate", "-p", str(p), "-m", str(m)]) == 2
    assert "Config error: outdir" in capsys.readouterr().err
