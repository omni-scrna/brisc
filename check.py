#!/usr/bin/env python3
# test that all runtime dependencies import cleanly
import h5py          # noqa: F401
import numpy         # noqa: F401
import polars        # noqa: F401
import scipy.sparse  # noqa: F401
import brisc

print("OK", brisc.__version__)
