#!/usr/bin/env python
"""Download Level-3 HST _drc products from MAST into data/raw (needs internet).

    python tools/download_mast.py --ra 53.12 --dec -27.80 --radius 0.03 \
        --instrument ACS/WFC --filters F435W F606W F814W

One file per filter is kept (the smallest that covers the position), so the
three bands of one pointing arrive together. Files are typically 150-300 MB each.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", type=float, required=True)
    ap.add_argument("--dec", type=float, required=True)
    ap.add_argument("--radius", type=float, default=0.03, help="degrees")
    ap.add_argument("--instrument", default="ACS/WFC")
    ap.add_argument("--filters", nargs="+", default=["F435W", "F606W", "F814W"])
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--dry-run", action="store_true", help="list what would be downloaded")
    a = ap.parse_args()
    from astroquery.mast import Observations
    obs = Observations.query_criteria(coordinates=f"{a.ra} {a.dec}", radius=f"{a.radius} deg",
                                      obs_collection="HST", dataproduct_type="image", calib_level=3,
                                      instrument_name=a.instrument, filters=a.filters)
    print(f"{len(obs)} observations found")
    if len(obs) == 0:
        return
    prods = Observations.filter_products(Observations.get_product_list(obs), productType="SCIENCE",
                                         productSubGroupDescription="DRC", extension="fits")
    prods.sort("size")
    chosen, seen = [], set()
    for i, row in enumerate(prods):
        name = str(row["productFilename"]).lower()
        f = next((x for x in a.filters if x.lower() in name), None)
        if f and f not in seen:
            seen.add(f)
            chosen.append(i)
    if not chosen:          # names without the filter in them: take the first per obs_id
        chosen = list(range(min(len(prods), len(a.filters))))
    sel = prods[chosen]
    for r in sel:
        print(f"  {r['productFilename']}  {int(r['size']) / 1e6:.0f} MB")
    if a.dry_run:
        return
    Path(a.out).mkdir(parents=True, exist_ok=True)
    man = Observations.download_products(sel, download_dir=a.out, flat=True)
    print(man["Local Path", "Status"])


if __name__ == "__main__":
    main()
