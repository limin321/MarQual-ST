"""End-to-end: the whole pipeline + gene-pair step on synthetic data (small null sizes)."""
import pandas as pd
import pytest

from marqual_st import PipelineConfig

pytestmark = pytest.mark.slow


def test_main_outputs(pipeline_run):
    pipe, adata = pipeline_run
    fig = pipe.figdir
    fs = pd.read_csv(fig / "filter_summary.csv").iloc[0]
    assert fs["n_bins_unfiltered"] == fs["n_bins_kept"] + fs["n_bins_filtered_out"]
    for f in ["technical_qc_summary.csv", "filter_summary.csv", "stray_pieces.csv", "section_depth_summary.csv",
              "all_celltypes_moran_index_table.csv", "morans_i_summary_table.csv", "morans_i_summary_chart.pdf",
              "all_pairs_colocalization_table.csv", "colocalization_technical_controls.csv",
              "celltype_annotation_counts.csv", "SYN_QCreport.html", "immune_niche_definition.csv",
              "run_parameters.csv",
              "immune_niche_de_top20.csv", "immune_niche_depth_check.csv",
              "neuron_enriched_region_de_genes.csv", "spatialimmune_smooth.pdf"]:
        assert (fig / f).exists(), f
    assert not (fig / "epi_enriched_region_de_genes.csv").exists()     # not in panel_region_celltypes
    assert pipe.config.analysis_h5ad_path.exists()
    used = pipe.figdir.parent
    again = PipelineConfig.from_csv(used / "params_used.csv", used / "marker_sets_used.csv")
    assert again.to_dict() == pipe.config.to_dict()          # the saved settings repeat the run
    assert (pipe.figdir.parent / "marqual_st.log").exists()


def test_verdicts(pipeline_run):
    pipe, _ = pipeline_run
    idx = pd.read_csv(pipe.figdir / "all_celltypes_moran_index_table.csv")
    assert set(idx["celltype"]) == {"immune", "neuron", "epi"}
    assert (idx["n_tested"] == 3).all()
    assert idx.set_index("celltype").loc["immune", "qc_verdict"].startswith("PASS")
    co = pd.read_csv(pipe.figdir / "all_pairs_colocalization_table.csv")
    assert len(co) == 3


def test_annotation_in_adata(pipeline_run):
    _, adata = pipeline_run
    assert "celltype_annotation" in adata.obs
    assert "immune" in set(adata.obs["celltype_annotation"].astype(str))
    assert len(adata.obs["celltype_annotation_plot"].cat.categories) <= 4 + 1     # + Structural Base


def test_gene_pairs_and_report(pipeline_run):
    from marqual_st import GenePairPipeline
    pipe, _ = pipeline_run
    res = GenePairPipeline(pipe.config).run()
    assert "GNLY_ATP8A2" in res
    gp = pipe.config.gene_pair_dir
    bv = pd.read_csv(gp / "GNLY_ATP8A2_spatial_colocalization.csv")
    assert len(bv) == 2 and (bv["bv_moran_I"] > 0).all()
    page = (pipe.figdir / "SYN_QCreport.html").read_text()
    assert 'id="genepairs"' in page and "Other files" not in page


def test_gene_pair_stop_extreme_case(pipeline_run, tmp_path, monkeypatch):
    """A pair whose niche leaves too few bins for DE is STOPPED, the next pair still runs, the report
    shows the STOP badge; a later normal run clears the STOP."""
    import shutil
    from dataclasses import replace

    from marqual_st import GenePairPipeline, differential
    pipe, _ = pipeline_run
    out = tmp_path / "run"
    shutil.copytree(pipe.figdir, out / "figures", ignore=shutil.ignore_patterns("gene_pairs"))
    cfg = replace(pipe.config, outdir=str(out), analysis_h5ad=str(pipe.config.analysis_h5ad_path))
    gp = cfg.gene_pair_dir
    pairs = [["GNLY", "ATP8A2"], ["CD3E", "CD8A"]]

    monkeypatch.setattr(differential, "MIN_DE_BINS", 10**9)            # every niche "too small"
    res = GenePairPipeline(cfg).run(pairs=pairs)                       # no exception: both pairs STOP
    assert {r["status"] for r in res.values()} == {"STOP"} and len(res) == 2
    st = pd.read_csv(gp / "GNLY_ATP8A2_gene_pair_status.csv").iloc[0]
    assert st["status"] == "STOP" and "DE needs" in st["reason"]
    assert not list(gp.glob("GNLY_ATP8A2_spatial_colocalization.*"))   # no half results
    page = (out / "figures" / "SYN_QCreport.html").read_text()
    assert "Gene pair: GNLY + ATP8A2" in page and "Not analysed" in page and ">STOP<" in page

    monkeypatch.setattr(differential, "MIN_DE_BINS", 10)
    res = GenePairPipeline(cfg).run(pairs=pairs[:1])                   # normal run: STOP cleared
    assert "status" not in res["GNLY_ATP8A2"]
    assert not (gp / "GNLY_ATP8A2_gene_pair_status.csv").exists()
    assert (gp / "GNLY_ATP8A2_spatial_colocalization.csv").exists()


def test_niche_top_percent(pipeline_run, tmp_path):
    from marqual_st.differential import NicheDE
    _, adata = pipeline_run
    NicheDE(tmp_path, "immune", quantile=0.90, plot=False).run(adata.copy())
    d = pd.read_csv(tmp_path / "immune_niche_definition.csv").iloc[0]
    assert d["top_percent"] == pytest.approx(10)
    assert abs(d["n_niche_bins"] / d["n_bins"] - 0.10) < 0.01


def test_quality_stop(synth_h5ad, tmp_path):
    from conftest import base_config
    from marqual_st import MarQualPipeline, PipelineConfig
    from marqual_st.io import DataQualityStop
    cfg = PipelineConfig.from_dict(base_config(synth_h5ad, tmp_path / "stop", quality={"min_bins_kept": 5000}))
    with pytest.raises(DataQualityStop, match="data quality too low"):
        MarQualPipeline(cfg).run()
    fig = cfg.figdir
    assert pd.read_csv(fig / "filter_summary.csv")["data_quality"].iloc[0] == "STOP"
    assert (fig / "spatial_qc_depth_maps.pdf").exists()
    assert not (fig / "all_celltypes_moran_index_table.csv").exists()
    page = (fig / "SYN_QCreport.html").read_text()
    assert "STOPPED after technical QC" in page and "UNFILTERED bins" in page


def test_run_parameters(pipeline_run):
    pipe, adata = pipeline_run
    df = pd.read_csv(pipe.figdir / "run_parameters.csv", dtype=str, keep_default_na=False)
    get = lambda step, p: df[(df["step"] == step) & (df["parameter"] == p)]["value"].tolist()  # noqa: E731
    assert get("config: filters", "min_counts") == ["600"]
    assert get("config: niche", "quantile") == ["0.95"]
    assert len(get("marker_sets", "marker_list")) == 3
    assert set(get("NicheDE", "quantile")) == {"0.95"}
    assert "scanpy" in set(df[df["step"] == "software"]["parameter"])
    assert "run_parameters" in adata.uns
    page = (pipe.figdir / "SYN_QCreport.html").read_text()
    assert 'id="params"' in page and "Marker sets" in page
    params = page[page.index('id="params"'):page.index("</section>", page.index('id="params"'))]
    assert "Software versions" in params and "scanpy" in params
    assert "details" not in params and "min_counts" not in params     # the rest stays in the CSV


def test_report_table_layout(pipeline_run):
    """Every scroll box shows 5 rows; tables and figures sit side by side in pairs."""
    pipe, _ = pipeline_run
    page = (pipe.figdir / "SYN_QCreport.html").read_text()
    assert ".tw{overflow:auto;max-height:167px" in page and "fitRows" in page
    pair = page[page.index("Pair: "):]
    s, d = pair.index("_colocalization_summary.csv"), pair.index("_colocalization_by_distance.csv")
    assert s < d < pair.index("_colocalization_null_plot")
    # figures in pairs: one row, each figure as wide as its aspect ratio (same height)
    assert "grid-template-columns:repeat(2,minmax(0,1fr))" in page
    row = pair[pair.index('<div class="frow">'):]
    assert row.index("_colocalization_null_plot") < row.index("_colocalization_by_distance.pdf")
    assert row.count('style="--ar:') >= 2
    assert page.count('<div class="frow">') >= 5    # depth maps, null plots, niche, annotation, pairs


def test_pdf_only_and_merged_depth_check(pipeline_run):
    pipe, _ = pipeline_run
    fig = pipe.figdir
    assert not list(fig.rglob("*.png")), "figures must be PDF only"
    assert not list(fig.glob("*_depth_check_after_matching.csv"))
    d = pd.read_csv(fig / "immune_niche_depth_check.csv")
    assert d["stage"].tolist() == ["before_matching", "after_matching"]
    assert d.loc[1, "n_region"] == d.loc[1, "n_background"]          # matched groups have equal size


def test_pdf_quality(pipeline_run):
    pipe, _ = pipeline_run
    raw = (pipe.figdir / "spatialimmune_smooth.pdf").read_bytes()
    assert b"/FontFile2" in raw                                      # TrueType font embedded (editable text)
    assert b"/Subtype /Image" in raw or b"/Subtype/Image" in raw     # bin layer rasterized inside the PDF


def test_report_embeds_pdf_previews(pipeline_run):
    pipe, _ = pipeline_run
    page = (pipe.figdir / "SYN_QCreport.html").read_text()
    assert page.count("data:image/") >= 10
    assert "No preview" not in page
    assert "Depth check: niche vs. background" in page and "after matching" in page
