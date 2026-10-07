#!/usr/bin/env python
"""
Regression check: compare two MarQual-ST output folders (e.g. before / after a code change,
or the legacy scripts vs. the package on the same data and settings).

    python tools/compare_runs.py OLD/figures NEW/figures [--rtol 1e-9]

Reports files present in only one folder and every CSV whose values differ. Exit code 0 when
all shared CSVs match, 1 otherwise. Figures are not compared (PDF/PNG bytes carry timestamps).
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def list_files(folder: str) -> set[str]:
    out = set()
    for root, _, files in os.walk(folder):
        for f in files:
            out.add(os.path.relpath(os.path.join(root, f), folder))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--rtol", type=float, default=1e-9)
    p.add_argument("--atol", type=float, default=1e-12)
    a = p.parse_args(argv)
    old, new = list_files(a.old), list_files(a.new)
    skip = lambda f: f.endswith("_QCreport.html")  # noqa: E731 - the report carries a time stamp
    only_old = sorted(f for f in old - new if not skip(f))
    only_new = sorted(f for f in new - old if not skip(f))
    if only_old:
        print("Only in OLD:", *only_old, sep="\n  ")
    if only_new:
        print("Only in NEW:", *only_new, sep="\n  ")
    n = bad = 0
    for f in sorted(old & new):
        if not f.endswith(".csv"):
            continue
        n += 1
        x, y = pd.read_csv(os.path.join(a.old, f)), pd.read_csv(os.path.join(a.new, f))
        try:
            pd.testing.assert_frame_equal(x, y, check_exact=False, rtol=a.rtol, atol=a.atol)
        except AssertionError as e:
            bad += 1
            print(f"DIFF {f}: {str(e).splitlines()[0]}")
    print(f"{n} CSVs compared, {bad} differ; {len(only_old)} only in OLD, {len(only_new)} only in NEW.")
    return 0 if bad == 0 and not only_old and not only_new else 1


if __name__ == "__main__":
    sys.exit(main())
