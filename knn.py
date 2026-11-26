#!/usr/bin/env python3
"""NNG stage, brisc-backed: SingleCell.neighbors() + shared_neighbors().

neighbors() is an approximate IVF-flat search (k-means, then exhaustive search
of the nearest clusters); shared_neighbors() is the Jaccard SNN graph brisc
clusters on (Seurat's definition, no diagonal).

Output neighbors_h5, the layout every CLUST arm and graph metric reads:
  /cell_ids                               bytes (n_cells,)
  /data /indices /indptr                  kNN distances, CSR, at the root
  /connectivities/{data,indices,indptr}   SNN Jaccard weights, CSR, symmetric

--connectivities sets the edge weights written as /connectivities, on the same
neighbours:
  native  brisc's shared_neighbors() (today's behaviour, the default).
  snn     Seurat's SNN as every kNN module writes it: |N_i & N_j| / |N_i | N_j| over
          neighbour sets with each cell itself included, edges below --snn_prune
          (default 1/15, Seurat's prune.SNN) removed, diagonal 1.
  umap    UMAP's fuzzy simplicial set (umap-learn's fuzzy_simplicial_set with its
          defaults, the weights scanpy and rapids-singlecell write), computed here
          from the neighbours so umap-learn is not a dependency.

Two conversions to the stage's conventions, both scanpy's:
  - n_neighbors counts the cell itself, brisc's num_neighbors does not, so
    num_neighbors = n_neighbors - 1 (the same row degree as nn_scanpy).
  - distances are Euclidean; brisc stores them squared.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import scipy.sparse as sp
from brisc import SingleCell

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import cli  # noqa: E402
from threads import NUM_THREADS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_base_args(p)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--embedding_tsv", dest="embedding_tsv", type=Path)
    g.add_argument("--corrected_tsv", dest="embedding_tsv", type=Path)  # NNG_C
    p.add_argument("--n_neighbors", type=int, required=True,
                   help="neighbourhood size, the cell itself included")
    p.add_argument("--random_seed", type=int, required=True)
    p.add_argument("--connectivities", choices=["native", "snn", "umap"], default="native",
                   help="edge weights: native (brisc's shared_neighbors), snn (Seurat's SNN) or umap")
    p.add_argument("--snn_prune", type=float, default=1 / 15,
                   help="--connectivities snn: drop edges below this Jaccard index (Seurat's prune.SNN)")
    return p.parse_args()


def snn_connectivities(distances, prune=1 / 15):
    """Seurat's SNN graph from a kNN graph's neighbour sets (its stored entries, each cell
    itself added): the Jaccard index |N_i & N_j| / |N_i | N_j| for every pair sharing a
    neighbour, entries below `prune` dropped. Symmetric, diagonal 1."""
    n = distances.shape[0]
    a = sp.csr_matrix((np.ones(distances.nnz, dtype=np.float32), distances.indices.copy(),
                       distances.indptr.copy()), shape=(n, n))
    a = ((a + sp.identity(n, dtype=np.float32, format="csr")) > 0).astype(np.float32)
    k = np.asarray(a.sum(axis=1)).ravel()
    shared = (a @ a.T).tocoo()
    jac = shared.data / (k[shared.row] + k[shared.col] - shared.data)
    keep = jac >= prune
    return sp.csr_matrix((jac[keep], (shared.row[keep], shared.col[keep])), shape=(n, n))


def umap_connectivities(idx, dist, n_iter=64, bandwidth=1.0, local_connectivity=1.0):
    """UMAP's fuzzy simplicial set (umap-learn's fuzzy_simplicial_set with its defaults)
    from each cell's k neighbours, itself first at distance 0 (idx, dist: n x k)."""
    n, k = idx.shape
    target = np.log2(k) * bandwidth
    rho = np.zeros(n)
    sigma = np.ones(n)
    mean_all = dist.mean()
    for i in range(n):
        d = dist[i]
        nz = d[d > 0]
        if len(nz) >= local_connectivity:
            j = int(np.floor(local_connectivity))
            frac = local_connectivity - j
            if j > 0:
                rho[i] = nz[j - 1] + (frac * (nz[j] - nz[j - 1]) if frac > 1e-5 else 0)
            else:
                rho[i] = frac * nz[0]
        elif len(nz) > 0:
            rho[i] = nz.max()
        lo, hi, mid = 0.0, np.inf, 1.0
        for _ in range(n_iter):
            psum = np.exp(-np.maximum(d[1:] - rho[i], 0) / mid).sum()
            if abs(psum - target) < 1e-5:
                break
            if psum > target:
                hi = mid
                mid = (lo + hi) / 2
            else:
                lo = mid
                mid = mid * 2 if hi == np.inf else (lo + hi) / 2
        if rho[i] > 0:
            mid = max(mid, 1e-3 * d.mean())
        else:
            mid = max(mid, 1e-3 * mean_all)
        sigma[i] = mid
    w = np.exp(-np.maximum(dist - rho[:, None], 0) / sigma[:, None])
    w[idx == np.arange(n)[:, None]] = 0
    p = sp.csr_matrix((w.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    p.eliminate_zeros()
    pt = p.T.tocsr()
    return (p + pt - p.multiply(pt)).tocsr()


def with_self(distances):
    """Each cell's neighbours as dense (n x k) index and distance arrays, itself first at
    distance 0, the rest nearest first: the input umap_connectivities expects."""
    n = distances.shape[0]
    k = np.diff(distances.indptr)[0]
    idx = distances.indices.reshape(n, k)
    dst = distances.data.reshape(n, k)
    order = np.argsort(dst, axis=1, kind="stable")
    idx, dst = np.take_along_axis(idx, order, 1), np.take_along_axis(dst, order, 1)
    return (np.hstack([np.arange(n)[:, None], idx]),
            np.hstack([np.zeros((n, 1)), dst]))


def run_knn(embedding, cells, n_neighbors, seed):
    """-> (distances, connectivities), both CSR n_cells x n_cells."""
    sc = SingleCell(X=sp.csr_array((len(cells), 1), dtype=np.float32),   # brisc wants an X
                    obs=pl.DataFrame({"cell_id": cells}),
                    var=pl.DataFrame({"gene_id": ["-"]}),
                    obsm={"pca": np.ascontiguousarray(embedding, dtype=np.float32)},
                    num_threads=NUM_THREADS)
    sc = sc.neighbors(QC_column=None, num_neighbors=n_neighbors - 1, seed=seed)
    sc = sc.shared_neighbors(QC_column=None)
    idx = sc.obsm["neighbors"]
    n, k = idx.shape
    distances = sp.csr_array(
        (np.sqrt(sc.obsm["distances"].ravel()), idx.ravel(), np.arange(0, n * k + 1, k)),
        shape=(n, n))
    return distances, sp.csr_array(sc.obsp["shared_neighbors"])


def write_csr(grp, m):
    for key in ("data", "indices", "indptr"):
        grp.create_dataset(key, data=getattr(m, key))


def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(args.embedding_tsv, separator="\t")   # first column = cell ids
    cells = df[:, 0].to_list()
    distances, connectivities = run_knn(df[:, 1:].to_numpy(), cells,
                                        args.n_neighbors, args.random_seed)
    assert (np.diff(distances.indptr) == args.n_neighbors - 1).all()
    if args.connectivities == "snn":
        connectivities = snn_connectivities(distances, args.snn_prune)
    elif args.connectivities == "umap":
        connectivities = umap_connectivities(*with_self(distances))

    oid = "neighbors_corrected_h5" if "neighbors_corrected_h5" in dict(args.outputs) else "neighbors_h5"
    out = cli.output_path(args, oid, "_neighbors.h5")
    with h5py.File(out, "w") as h5:
        h5.create_dataset("cell_ids", data=np.array(cells, dtype="S"))
        write_csr(h5, distances)
        write_csr(h5.create_group("connectivities"), connectivities)
    print(f"  wrote: {out} ({distances.nnz} kNN edges, {connectivities.nnz} {args.connectivities} edges)")


if __name__ == "__main__":
    main()
