import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from marqual_st.depth import DepthModel, MatchedGeneSetSampler, moran_i, row_normalize


def grid_weights(n):
    """Rook weights on an n x n grid."""
    idx = np.arange(n * n).reshape(n, n)
    rows, cols = [], []
    for i in range(n):
        for j in range(n):
            for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                if 0 <= i + di < n and 0 <= j + dj < n:
                    rows.append(idx[i, j])
                    cols.append(idx[i + di, j + dj])
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n * n, n * n))


def test_row_normalize():
    W = row_normalize(grid_weights(4))
    assert np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0)


def test_moran_sign():
    n = 10
    W = grid_weights(n)
    i, j = np.indices((n, n))
    checker = ((i + j) % 2).ravel().astype(float)
    gradient = i.ravel().astype(float)
    assert moran_i(checker, W) < -0.9
    assert moran_i(gradient, W) > 0.8
    assert moran_i(np.ones(n * n), W) == 0.0


def make_adata(n=500, seed=0):
    rng = np.random.default_rng(seed)
    tc = rng.lognormal(7, 0.5, n)
    obs = pd.DataFrame({"total_counts": tc, "n_genes_by_counts": tc / 3, "sec": rng.choice(["A", "B"], n)},
                       index=[f"b{i}" for i in range(n)])
    return AnnData(X=np.zeros((n, 1)), obs=obs)


def test_residualize_removes_depth():
    ad = make_adata()
    ad.obs["y"] = 2.0 * np.log1p(ad.obs["total_counts"]) + np.random.default_rng(1).normal(0, 0.01, ad.n_obs)
    r2 = DepthModel().residualize(ad, ["y"])
    assert r2["depth_R2"].iloc[0] > 0.99
    assert abs(np.corrcoef(ad.obs["y_resid"], np.log1p(ad.obs["total_counts"]))[0, 1]) < 1e-6


def test_design_with_compartment():
    ad = make_adata()
    X = DepthModel(("log_total_counts", "log_n_genes"), compartment_col="sec").design_matrix(ad)
    assert X.shape == (ad.n_obs, 4)


def test_depth_model_errors():
    with pytest.raises(ValueError):
        DepthModel(("depth",))
    ad = make_adata()
    del ad.obs["total_counts"]
    with pytest.raises(KeyError, match="calculate_qc_metrics"):
        DepthModel.total_counts(ad)


def test_sampler_excludes_markers_and_is_seeded():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(rng.gamma(1, 1, 60), size=(200, 60)).astype(float))
    ad = AnnData(X=X, var=pd.DataFrame(index=[f"G{i}" for i in range(60)]))
    ad.layers["lognorm"] = X
    markers = ["G1", "G2", "G3"]
    s1 = MatchedGeneSetSampler(ad, markers, seed=5)
    s2 = MatchedGeneSetSampler(ad, markers, seed=5)
    assert np.allclose(s1.draw(), s2.draw())
    assert all(set(markers).isdisjoint(pool) for pool in s1.genes_by_bin.values())
