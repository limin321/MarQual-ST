from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from synthetic import IMMUNE, NEURON, make_synthetic  # noqa: E402

MARKER_SETS = {"immune": IMMUNE, "neuron": NEURON + ["ULK4", "FAM155A"],
               "epi": ["GENE1", "GENE2", "GENE3", "GENE10", "GENE11"]}


@pytest.fixture(scope="session")
def synth_h5ad(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "synth.h5ad"
    make_synthetic().write_h5ad(path)
    return path


def base_config(input_path, outdir, **overrides) -> dict:
    cfg = {
        "sample": "SYN", "platform": "stereo", "input": str(input_path), "outdir": str(outdir),
        "marker_sets": MARKER_SETS, "panel_region_celltypes": ["immune", "neuron"], "celltype_pairs": "all",
        "sections": {"num_tissue": 2}, "statistics": {"n_null": 60, "n_null_distance": 20, "seed": 0},
        "annotation": {"min_bins": 50, "max_plot_labels": 4},
        "gene_pairs": {"pairs": [["GNLY", "ATP8A2"]], "permutations": 99},
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture(scope="session")
def pipeline_run(synth_h5ad, tmp_path_factory):
    """One full (small) pipeline run shared by the end-to-end tests."""
    from marqual_st import MarQualPipeline, PipelineConfig
    outdir = tmp_path_factory.mktemp("run")
    cfg = PipelineConfig.from_dict(base_config(synth_h5ad, outdir))
    pipe = MarQualPipeline(cfg)
    adata = pipe.run()
    return pipe, adata
