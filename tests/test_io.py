import pytest

from marqual_st.io import DataLoader, StereoSeqLoader, VisiumHDLoader


def test_registry():
    assert isinstance(DataLoader.for_platform("stereo"), StereoSeqLoader)
    assert isinstance(DataLoader.for_platform("visiumhd"), VisiumHDLoader)
    with pytest.raises(ValueError, match="Unknown platform"):
        DataLoader.for_platform("xenium")


def test_stereo_load(synth_h5ad):
    ad = DataLoader.for_platform("stereo").load(synth_h5ad)
    assert "GNLY" in ad.var_names
    assert {"total_counts", "n_genes_by_counts", "pct_counts_mt"} <= set(ad.obs.columns)
    assert ad.var["mt"].sum() == 3


def test_filter_removing_all_bins_explains_why(synth_h5ad):
    from marqual_st.io import Preprocessor
    ad = DataLoader.for_platform("stereo").load(synth_h5ad)
    from marqual_st.io import DataQualityStop
    with pytest.raises(DataQualityStop, match="removes ALL .* do not lower the cutoff"):
        Preprocessor(min_counts=10**9).filter_and_normalize(ad)


@pytest.mark.parametrize("min_counts, gate_kw, expected", [
    (600, {}, "PASS"),                                           # synthetic data: 0.1% filtered out
    (2000, {"caution_pct_filtered": 50}, "CAUTION"),             # ~53% filtered out
    (2000, {}, "PASS"),                                          # ~53% < 60% caution line
    (5000, {}, "STOP"),                                          # nearly all filtered out
    (600, {"min_bins_kept": 5000}, "STOP"),                      # too few bins kept
])
def test_quality_gate(synth_h5ad, min_counts, gate_kw, expected):
    from marqual_st.io import DataQualityGate, Preprocessor
    ad = DataLoader.for_platform("stereo").load(synth_h5ad)
    gate = DataQualityGate(Preprocessor(min_counts=min_counts), **gate_kw)
    stats = gate.assess(ad)
    assert stats["data_quality"] == expected
    assert ad.uns["filter_summary"]["data_quality"] == expected
    assert gate.should_stop(stats) == (expected == "STOP")
    assert not DataQualityGate(Preprocessor(min_counts=min_counts), force=True, **gate_kw).should_stop(stats)
