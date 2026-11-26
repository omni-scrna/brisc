#!/usr/bin/env python3
"""Chain pca -> knn -> cluster on the fixture and check the stage contracts.
Run: pixi run test"""

import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import scipy.sparse as sp

ROOT = Path(__file__).parent.parent
H5 = ROOT / "tests/data/datasets_normalized_selected.h5"
K, NN = 10, 15


def run(script, out, *args):
    subprocess.run([sys.executable, ROOT / script, "--output_dir", out, "--name", "t",
                    *map(str, args)], check=True, stdout=subprocess.DEVNULL)


def table(path):
    df = pl.read_csv(path, separator="\t")
    return df[:, 0].to_list(), df[:, 1:].to_numpy()


with tempfile.TemporaryDirectory() as out:
    out = Path(out)

    # PCA: scores and loadings equal a dense standardised SVD, up to sign
    run("pca.py", out, "--normalized_selected_h5", H5, "--n_components", K, "--random_seed", 42)
    cells, E = table(out / "t_embedding.tsv")
    genes, V = table(out / "t_loadings.tsv")
    with h5py.File(H5) as h5:
        g = h5["matrix"]
        X = sp.csc_array((g["data"][:], g["indices"][:], g["indptr"][:]),
                         shape=tuple(g["shape"][:])).T.toarray()
        assert cells == list(g["barcodes"][:].astype(str))
        assert genes == list(g["genes"][:].astype(str))
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    assert np.allclose(np.abs(E), np.abs(U[:, :K] * S[:K]), atol=1e-3 * S[0])
    assert np.allclose(np.abs(V), np.abs(Vt[:K].T), atol=1e-3)

    # kNN: n_neighbors - 1 Euclidean neighbours per cell, near-exact, symmetric SNN
    run("knn.py", out, "--embedding_tsv", out / "t_embedding.tsv", "--n_neighbors", NN, "--random_seed", 42)
    with h5py.File(out / "t_neighbors.h5") as h5:
        n = len(cells)
        assert list(h5["cell_ids"][:].astype(str)) == cells
        D = sp.csr_array((h5["data"][:], h5["indices"][:], h5["indptr"][:]), shape=(n, n))
        C = h5["connectivities"]
        C = sp.csr_array((C["data"][:], C["indices"][:], C["indptr"][:]), shape=(n, n))
    assert (np.diff(D.indptr) == NN - 1).all() and D.diagonal().sum() == 0
    exact = np.sqrt(((E[:, None, :] - E[None, :200, :]) ** 2).sum(-1)).T   # first 200 cells
    truth = np.argsort(exact, axis=1)[:, 1:NN]
    recall = np.mean([len(set(D.indices[D.indptr[i]:D.indptr[i + 1]]) & set(truth[i])) / (NN - 1)
                      for i in range(200)])
    assert recall > 0.95, recall
    i = D.indices[0]
    assert np.isclose(D.data[0], np.linalg.norm(E[0] - E[i]), rtol=1e-4)   # not squared
    assert abs(C - C.T).max() == 0 and C.data.min() > 0 and C.data.max() <= 1

    # --connectivities: the same neighbours, other weights
    def graph(tag, *extra):
        (out / tag).mkdir()
        run("knn.py", out / tag, "--embedding_tsv", out / "t_embedding.tsv", "--n_neighbors", NN,
            "--random_seed", 42, *extra)
        with h5py.File(out / tag / "t_neighbors.h5") as h5:
            d = sp.csr_array((h5["data"][:], h5["indices"][:], h5["indptr"][:]), shape=(n, n))
            c = h5["connectivities"]
            return d, sp.csr_array((c["data"][:], c["indices"][:], c["indptr"][:]), shape=(n, n))

    D_snn, C_snn = graph("snn", "--connectivities", "snn")
    D_umap, C_umap = graph("umap", "--connectivities", "umap")
    assert (D_snn != D).nnz == 0 and (D_umap != D).nnz == 0
    # snn: Seurat's Jaccard of neighbour sets, each cell itself included, pruned at 1/15
    sets = [set(D.indices[D.indptr[i]:D.indptr[i + 1]]) | {i} for i in range(n)]
    for i in range(50):
        for j in range(n):
            want = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
            assert np.isclose(C_snn[i, j], want if want >= 1 / 15 else 0, atol=1e-6)
    assert abs(C_snn - C_snn.T).max() < 1e-6 and np.allclose(C_snn.diagonal(), 1)
    # umap: symmetric weights in (0, 1], umap-learn's when it is installed
    assert abs(C_umap - C_umap.T).max() < 1e-12 and C_umap.data.min() > 0 and C_umap.data.max() <= 1
    try:
        from umap.umap_ import fuzzy_simplicial_set
    except ImportError:
        pass
    else:
        sys.path.insert(0, str(ROOT))
        from knn import with_self
        idx, dst = with_self(D)
        ref, _, _ = fuzzy_simplicial_set(E, NN, 0, "euclidean", knn_indices=idx, knn_dists=dst)
        assert abs(ref - C_umap).max() < 1e-4

    # Leiden: every cell labelled, same seed = same partition, resolution moves k
    def clusters(tag, res, *extra):
        run("cluster.py", out / tag, "--neighbors_h5", out / "t_neighbors.h5",
            "--resolution", res, "--random_seed", 42, *extra)
        df = pl.read_csv(out / tag / "t_clusters.tsv", separator="\t")
        assert df.columns == ["cell_id", "cluster"] and df["cell_id"].to_list() == cells
        return df["cluster"]

    a, b = clusters("a", 1.0), clusters("b", 1.0)
    assert a.equals(b) and a.null_count() == 0
    assert 1 < clusters("c", 0.2).n_unique() < a.n_unique()
    clusters("d", 1.0, "--min_cluster_size", 1)   # the path that skips the kNN lists

print("OK")
