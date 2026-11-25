#!/usr/bin/env python3
"""PCA stage, brisc-backed: SingleCell.pca() (IRLBA, float32).

brisc scales every gene to zero mean AND unit variance (ddof=1) inside pca()
by default. --scale false (pca(scale=False), btraven00/brisc fork) only centres,
the estimand of the scanpy / scrapper / rapids arms. Scores are U*S.

Outputs: embedding_tsv (cells x k, PC1..PCk), loadings_tsv (genes x k).
brisc returns scores only; the loadings are recovered exactly from them as
V = Z'E / s^2 (Z the centred or standardised matrix, E the scores, s^2 = colSums(E^2)),
without densifying Z.
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
    cli.add_base_args(p)          # --output_dir, --name, --output id=PATH
    cli.add_stage_args(p, "PCA")  # --normalized_selected_h5
    p.add_argument("--n_components", type=int, required=True)
    p.add_argument("--random_seed", type=int, required=True)
    p.add_argument("--scale", choices=["true", "false"], default="true",
                   help="unit-variance genes first (brisc default)")
    cpu_variant.add_arg(p)
    return p.parse_args()


def read_matrix(path):
    """TENx-layout h5 (genes x cells CSC) -> cells x genes CSR float32, ids."""
    with h5py.File(path, "r") as h5:
        g = h5["matrix"]
        X = sp.csc_array((g["data"][:], g["indices"][:], g["indptr"][:]),
                         shape=tuple(g["shape"][:]))
        return (X.T.tocsr().astype(np.float32),   # brisc refuses float64
                g["barcodes"][:].astype(str), g["genes"][:].astype(str))


def run_pca(X, cells, genes, n_components, seed, scale):
    sc = SingleCell(X=X, obs=pl.DataFrame({"cell_id": cells}),
                    var=pl.DataFrame({"gene_id": genes}), num_threads=NUM_THREADS)
    sc.uns["normalized"] = True   # the NORM stage did it; pca() checks the flag
    sc = sc.pca(QC_column=None, hvg_column=None, num_PCs=n_components, seed=seed,
                scale=scale, verbose=True)
    E = sc.obsm["pca"].astype(np.float64)

    # loadings: V = Z'E / s^2 with Z = (X - mu) / sd, kept sparse
    X = sp.csr_array(X, dtype=np.float64)
    n = X.shape[0]
    mu = X.sum(axis=0) / n
    sd = np.sqrt((X.multiply(X).sum(axis=0) - n * mu**2) / (n - 1)) if scale else np.ones(X.shape[1])
    ZtE = X.T @ E - np.outer(mu, E.sum(axis=0))
    # a constant gene has no standardised value: it carries no loading
    V = np.divide(ZtE, sd[:, None], out=np.zeros_like(ZtE), where=sd[:, None] > 0)
    return E, V / (E**2).sum(axis=0)


def write_tsv(path, matrix, ids, id_name):
    # header has the id column named; rows are id + values (the sibling layout)
    cols = [f"PC{i + 1}" for i in range(matrix.shape[1])]
    pl.from_numpy(matrix, schema=cols).insert_column(
        0, pl.Series(id_name, ids)).write_csv(path, separator="\t")
    print(f"  wrote: {path}")


def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    print(f"  brisc build: {cpu_variant.loaded()} (--cpu_variant {args.cpu_variant})")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    X, cells, genes = read_matrix(args.normalized_selected_h5)
    print(f"  matrix (cells x genes): {X.shape}, threads: {NUM_THREADS}")
    E, V = run_pca(X, cells, genes, args.n_components, args.random_seed,
                   args.scale == "true")
    assert E.shape == (len(cells), args.n_components) and np.isfinite(E).all()
    assert V.shape == (len(genes), args.n_components)

    write_tsv(cli.output_path(args, "embedding_tsv", "_embedding.tsv"), E, cells, "cell_id")
    write_tsv(cli.output_path(args, "loadings_tsv", "_loadings.tsv"), V, genes, "gene_id")


if __name__ == "__main__":
    main()
