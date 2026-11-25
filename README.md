# brisc

[brisc](https://github.com/briscverse/brisc) arms, built from the
[btraven00/brisc](https://github.com/btraven00/brisc) fork (`pca(scale=False)`), for the
omni-scrna split-stages plan, api 0.8.

| entrypoint | stage | upstream call | parameters |
| --- | --- | --- | --- |
| `pca` | PCA | `SingleCell.pca()` | `n_components`, `random_seed`, `scale` (default true) |
| `knn` | NNG | `SingleCell.neighbors()` + `shared_neighbors()` | `n_neighbors`, `random_seed` |
| `cluster` | CLUST | `SingleCell.cluster()` | `resolution`, `random_seed`, `min_cluster_size` (default 5) |

Every entrypoint also takes `cpu_variant`: `auto` (default, brisc's own pick) or
`x86_64_v2` / `x86_64_v3` / `x86_64_v4`, the brisc build to run. All three ship in
the one install; the job exits if the CPU cannot run the level asked for. The
builds differ in the last float32 digits (PCA scores ~1e-4 on tests/data).

Everything not listed is left at the brisc default.

What differs from the sibling arms:

- **PCA is standardised by default.** brisc scales genes to unit variance
  inside `pca()`; `scale: "false"` only centres, as scanpy, scrapper and
  rapids do. It also
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
