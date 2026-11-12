# brisc

[brisc](https://github.com/briscverse/brisc) 0.2.5 (PyPI wheel) arms for the
omni-scrna split-stages plan, api 0.8.

| entrypoint | stage | upstream call | parameters |
| --- | --- | --- | --- |
| `pca` | PCA | `SingleCell.pca()` | `n_components`, `random_seed` |
| `knn` | NNG | `SingleCell.neighbors()` + `shared_neighbors()` | `n_neighbors`, `random_seed` |
| `cluster` | CLUST | `SingleCell.cluster()` | `resolution`, `random_seed`, `min_cluster_size` (default 5) |

Everything not listed is left at the brisc default.

What differs from the sibling arms:

- **PCA is standardised.** brisc scales genes to unit variance inside `pca()`
  and has no switch for it; scanpy, scrapper and rapids only centre. It also
  computes in float32. brisc returns no loadings: they are recovered from the
  scores (`V = Z'E / s^2`).
- **kNN is approximate** (IVF-flat). `n_neighbors` counts the cell itself, as
  in scanpy, so brisc is asked for `n_neighbors - 1`. Distances are written
  Euclidean (brisc keeps them squared).
- **`/connectivities` is the Jaccard SNN graph**, not scanpy's UMAP weights.
- **Leiden is serial and deterministic** for a seed. Clusters under
  `min_cluster_size` cells are merged into a neighbour's cluster.
- Threads come from `OMP_NUM_THREADS` (default 1). Single-threaded PCA differs
  from multi-threaded in the last float32 digits (`match_parallel=False`, the
  brisc default).

```
pixi run test         # pca -> knn -> cluster on tests/data, checks the contracts
pixi run export-env   # regenerate envs/brisc.yml
```
