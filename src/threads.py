"""brisc defaults to every core on the box; take the job's share instead.
Snakemake exports OMP_NUM_THREADS = the rule's threads."""

import os

NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS") or 1)
