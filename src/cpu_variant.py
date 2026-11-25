"""--cpu_variant: run brisc's x86_64_v2/v3/v4 build instead of the best one the CPU takes.

brisc ships all three and picks one when it is imported, from numpy's __cpu_features__
dict (brisc/utils.py, import_cython). Clearing AVX512F / AVX2 in that dict steers only
brisc's pick; numpy's own SIMD dispatch was fixed when numpy loaded. So this module
must be imported BEFORE brisc, and reads the flag from sys.argv itself.
"""

import argparse
import sys

from numpy._core._multiarray_umath import __cpu_features__ as cpu

CHOICES = ["auto", "x86_64_v2", "x86_64_v3", "x86_64_v4"]
MASK = {"x86_64_v2": ("AVX512F", "AVX2"), "x86_64_v3": ("AVX512F",)}
NEEDS = {"x86_64_v3": "AVX2", "x86_64_v4": "AVX512F"}


def add_arg(p):
    p.add_argument("--cpu_variant", choices=CHOICES, default="auto",
                   help="brisc build to run (auto: the best this CPU supports)")


def loaded():
    """the variant brisc actually imported, for the job log"""
    return sorted({m.split(".")[1] for m in sys.modules if m.startswith("brisc.x86_64_")})


_p = argparse.ArgumentParser(add_help=False)
add_arg(_p)
VARIANT = _p.parse_known_args()[0].cpu_variant
# asking for a level above the CPU would SIGILL mid-run; fail here instead
if VARIANT in NEEDS and not cpu.get(NEEDS[VARIANT]):
    sys.exit(f"--cpu_variant {VARIANT}: this CPU has no {NEEDS[VARIANT]}")
for flag in MASK.get(VARIANT, ()):
    cpu[flag] = False
