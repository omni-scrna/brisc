#!/usr/bin/env python3
"""CLUST stage, brisc-backed: SingleCell.cluster() (Leiden, RB configuration).

brisc's Leiden is a serial GVE-Leiden, deterministic given the seed. It runs on
/connectivities of neighbors_h5 (symmetric weights), whichever NNG arm wrote it.

Output clusters_tsv: cell_id<TAB>cluster.

brisc also merges clusters smaller than --min_cluster_size into the cluster of
their nearest neighbour, which needs the kNN lists: those are rebuilt from the
root distance CSR, so every cell must have the same number of neighbours there.
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_base_args(p)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--neighbors_h5", dest="neighbors_h5", type=Path)
    g.add_argument("--neighbors_corrected_h5", dest="neighbors_h5", type=Path)  # CLUST_C
    p.add_argument("--resolution", type=float, required=True)
    p.add_argument("--random_seed", type=int, required=True)
    p.add_argument("--min_cluster_size", type=int, default=5,
                   help="brisc default; 1 disables the small-cluster merge")
    cpu_variant.add_arg(p)
    return p.parse_args()


def read_csr(grp, n):
    return sp.csr_array((grp["data"][:], grp["indices"][:], grp["indptr"][:]), shape=(n, n))


def neighbor_lists(distances):
    """CSR kNN distances -> (n_cells, k) neighbour indices, nearest first."""
    degree = np.diff(distances.indptr)
    if (degree != degree[0]).any():
        raise SystemExit("the kNN graph has a varying number of neighbours per cell "
                         f"({degree.min()}..{degree.max()}), so brisc's small-cluster "
                         "merge cannot use it; pass --min_cluster_size 1")
    k = int(degree[0])
    order = np.argsort(distances.data.reshape(-1, k), axis=1, kind="stable")
    return np.take_along_axis(distances.indices.reshape(-1, k), order, axis=1).astype(np.uint32)


def run_cluster(connectivities, neighbors, cells, resolution, seed, min_cluster_size):
    sc = SingleCell(X=sp.csr_array((len(cells), 1), dtype=np.float32),   # brisc wants an X
                    obs=pl.DataFrame({"cell_id": cells}),
                    var=pl.DataFrame({"gene_id": ["-"]}),
                    obsm={} if neighbors is None else {"neighbors": neighbors},
                    obsp={"shared_neighbors": connectivities.astype(np.float32)},
                    num_threads=1)   # one resolution = one thread, brisc insists
    sc = sc.cluster(QC_column=None, resolution=resolution, seed=seed,
                    min_cluster_size=min_cluster_size)
    return sc.obs["cluster"].cast(pl.String)


def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    print(f"  brisc build: {cpu_variant.loaded()} (--cpu_variant {args.cpu_variant})")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with h5py.File(args.neighbors_h5, "r") as h5:
        cells = h5["cell_ids"][:].astype(str)
        connectivities = read_csr(h5["connectivities"], len(cells))
        neighbors = None if args.min_cluster_size == 1 else neighbor_lists(read_csr(h5, len(cells)))

    labels = run_cluster(connectivities, neighbors, cells, args.resolution,
                         args.random_seed, args.min_cluster_size)
    assert len(labels) == len(cells) and labels.null_count() == 0
    print(f"  {labels.n_unique()} clusters")

    oid = "clusters_corrected_tsv" if "clusters_corrected_tsv" in dict(args.outputs) else "clusters_tsv"
    out = cli.output_path(args, oid, "_clusters.tsv")
    pl.DataFrame({"cell_id": cells, "cluster": labels}).write_csv(out, separator="\t")
    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
