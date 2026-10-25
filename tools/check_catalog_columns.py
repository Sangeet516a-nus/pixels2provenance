#!/usr/bin/env python
"""Check that the catalogue files in data/catalogs have the columns that
config.yaml maps (catalogs.<kind>.columns). Run this BEFORE the pipeline when
you add a new GZH / Gaia / CANDELS file.

    python tools/check_catalog_columns.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p2p.catalogs import K15_CLASS_COLUMNS, _candidates, _text_format, find_catalog  # noqa: E402
from p2p.context import Ctx  # noqa: E402


def header(path: Path):
    n = path.name.lower()
    if n.endswith((".fits", ".fit", ".fits.gz")):
        from astropy.table import Table
        return Table.read(path).colnames
    sep, comment = _text_format(path)
    return [c.strip() for c in pd.read_csv(path, nrows=0, sep=sep, comment=comment).columns]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ctx = Ctx(ROOT, ap.parse_args().config)
    for kind in ("gzh", "gaia", "candels"):
        p = find_catalog(ctx, kind)
        print(f"\n=== {kind}: {ctx.rel(p) if p else 'NOT FOUND in ' + str(ctx.catalogs)} ===")
        if p is None:
            continue
        cols = header(p)
        low = {c.strip().lower(): c for c in cols}
        required = {"ra", "dec"} | (set(K15_CLASS_COLUMNS) if kind == "candels" else set())
        for std, want in ctx.cfg["catalogs"][kind]["columns"].items():
            found = next((low[c.lower()] for c in _candidates(want) if c.lower() in low), None)
            if found is not None:
                print(f"  OK       {std:15s} <- {found}")
            else:
                close = difflib.get_close_matches(_candidates(want)[0], cols, n=3, cutoff=0.4)
                tag = "MISSING!" if std in required else "missing "
                print(f"  {tag} {std:15s} <- {want}   closest: {close}"
                      + ("   (REQUIRED - the pipeline stops)" if std in required else "   (optional)"))
        print(f"  ({len(cols)} columns in the file)")


if __name__ == "__main__":
    main()
