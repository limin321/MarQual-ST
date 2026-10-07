import pandas as pd

from marqual_st.report import FileIndex, ImageEmbedder, ReportBuilder


def test_report_on_minimal_folder(tmp_path):
    pd.DataFrame([{"n_bins": 10, "median_total_counts": 100.0, "median_n_genes": 50.0, "num_tissue": 1}]).to_csv(
        tmp_path / "technical_qc_summary.csv", index=False)
    pd.DataFrame([{"celltype": "immune", "qc_verdict": "PASS", "z_score_resid": 5.0, "empirical_p_value_resid": 0.001,
                   "fdr_qvalue_resid": 0.002, "n_null": 1000, "depth_R2": 0.1, "n_tested": 1}]).to_csv(
        tmp_path / "all_celltypes_moran_index_table.csv", index=False)
    (tmp_path / "unrelated.txt").write_text("x")
    out = ReportBuilder(tmp_path, sample="S1").build()
    assert out.name == "S1_QCreport.html"
    page = out.read_text()
    assert "PASS" in page and 'id="other"' in page and "unrelated.txt" in page
    # a rebuilt report never lists an earlier report as a file
    out2 = ReportBuilder(tmp_path, sample="S1").build()
    assert "S1_QCreport.html</a>" not in out2.read_text()
    assert FileIndex.is_report("S1_QCreport.html")


def test_gene_pair_verdict():
    v = ReportBuilder.gene_pair_verdict
    assert v({"fdr_qvalue": 0.01, "bv_moran_I": 0.2}) == "CO-LOCALIZED"
    assert v({"fdr_qvalue": 0.01, "bv_moran_I": -0.2}) == "SEGREGATED"
    assert v({"fdr_qvalue": 0.2, "bv_moran_I": 0.2}) == "NOT_SIGNIFICANT"


def test_image_embedder_small_png_untouched(tmp_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([0, 1])
    p = tmp_path / "a.png"
    fig.savefig(p, dpi=30)
    plt.close(fig)
    data, mime = ImageEmbedder().compact(str(p))
    assert mime == "image/png" and data == p.read_bytes()


def test_pdf_previewer_and_stray_rows(tmp_path):
    import matplotlib.pyplot as plt

    from marqual_st.plot_style import save_pdf
    from marqual_st.report import PdfPreviewer
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    pdf = save_pdf(fig, tmp_path / "morans_i_summary_chart.png")   # extension forced to .pdf
    plt.close(fig)
    assert pdf.suffix == ".pdf" and pdf.exists()
    assert PdfPreviewer().render(str(pdf))[:8] == b"\x89PNG\r\n\x1a\n"
    pd.DataFrame({"piece_rank": range(3, 30), "n_bins": range(30, 3, -1)}).to_csv(tmp_path / "stray_pieces.csv",
                                                                                  index=False)
    page = ReportBuilder(tmp_path, sample="S").build().read_text()
    assert 'class="tw rows5"' in page and "data:image/png" in page
