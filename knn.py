#!/usr/bin/env python3
"""NNG stage, brisc-backed: SingleCell.neighbors() + shared_neighbors().

neighbors() is an approximate IVF-flat search (k-means, then exhaustive search
of the nearest clusters); shared_neighbors() is the Jaccard SNN graph brisc
clusters on (Seurat's definition, no diagonal).

Output neighbors_h5, the layout every CLUST arm and graph metric reads:
  /cell_ids                               bytes (n_cells,)
  /data /indices /indptr                  kNN distances, CSR, at the root
  /connectivities/{data,indices,indptr}   SNN Jaccard weights, CSR, symmetric

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

sys.path.insert(0, str(Path(__file__).parent / "src"))
import cpu_variant  # noqa: E402  (before brisc: it picks the build at import)
from brisc import SingleCell  # noqa: E402
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
    cpu_variant.add_arg(p)
    return p.parse_args()


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
    print(f"  brisc build: {cpu_variant.loaded()} (--cpu_variant {args.cpu_variant})")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(args.embedding_tsv, separator="\t")   # first column = cell ids
    cells = df[:, 0].to_list()
    distances, connectivities = run_knn(df[:, 1:].to_numpy(), cells,
                                        args.n_neighbors, args.random_seed)
    assert (np.diff(distances.indptr) == args.n_neighbors - 1).all()

    oid = "neighbors_corrected_h5" if "neighbors_corrected_h5" in dict(args.outputs) else "neighbors_h5"
    out = cli.output_path(args, oid, "_neighbors.h5")
    with h5py.File(out, "w") as h5:
        h5.create_dataset("cell_ids", data=np.array(cells, dtype="S"))
        write_csr(h5, distances)
        write_csr(h5.create_group("connectivities"), connectivities)
    print(f"  wrote: {out} ({distances.nnz} kNN edges, {connectivities.nnz} SNN edges)")


if __name__ == "__main__":
    main()
