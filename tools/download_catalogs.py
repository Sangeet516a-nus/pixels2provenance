#!/usr/bin/env python
"""Download the reference catalogues into data/catalogs (needs internet).

    python tools/download_catalogs.py gzh                                   # Galaxy Zoo: Hubble Table 4
    python tools/download_catalogs.py gaia --ra 53.12 --dec -27.80          # Gaia DR3 cone around a pointing
    python tools/download_catalogs.py candels                               # Kartaltepe+2015 (VizieR J/ApJS/221/11/table3)
    python tools/download_catalogs.py all --ra 53.12 --dec -27.80

Gaia can also be left to the pipeline: set catalogs.gaia.query_online: true in config.yaml.
After downloading, run:  python tools/check_catalog_columns.py
"""
from __future__ import annotations

import argparse
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATS = ROOT / "data" / "catalogs"
GZH_URL = "https://zooniverse-data.s3.amazonaws.com/galaxy-zoo-hubble/gz_hubble_main.csv.gz"


def gzh():
    out = CATS / "gz_hubble_main.csv.gz"
    if out.exists():
        print(f"[gzh] already there: {out.relative_to(ROOT)}")
        return
    print(f"[gzh] downloading {GZH_URL} ...")
    tmp = out.with_suffix(".part")
    with urllib.request.urlopen(GZH_URL) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(out)
    print(f"[gzh] saved {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.0f} MB)")


def gaia(ra, dec, radius):
    from astroquery.gaia import Gaia
    q = ("SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, teff_gspphot "
         "FROM gaiadr3.gaia_source "
         f"WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius}))")
    print(f"[gaia] cone ra={ra} dec={dec} r={radius} deg ...")
    t = Gaia.launch_job(q).get_results()
    out = CATS / f"gaia_dr3_{ra:.3f}_{dec:+.3f}.csv"
    t.to_pandas().to_csv(out, index=False)
    print(f"[gaia] {len(t)} sources -> {out.relative_to(ROOT)}")
    print("       (one pointing per file is fine only if you have ONE pointing; for several pointings use "
          "catalogs.gaia.query_online: true, or one larger radius that covers all of them)")


def candels():
    """Kartaltepe et al. (2015) FRACTIONAL table = VizieR J/ApJS/221/11/table3 (GOODS-S, H < 24.5),
    saved as data/catalogs/candels_k15.csv with the VizieR column labels (f(Sph), f(Dk), ...)."""
    out = CATS / "candels_k15.csv"
    if out.exists():
        print(f"[candels] already there: {out.relative_to(ROOT)}")
        return
    url = ("https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY="
           + urllib.parse.quote('SELECT * FROM "J/ApJS/221/11/table3"'))
    print("[candels] downloading VizieR J/ApJS/221/11/table3 (fractional classifications) ...")
    tmp = out.with_suffix(".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(out)
    import pandas as pd
    t = pd.read_csv(out, nrows=5)
    n = sum(1 for _ in open(out)) - 1
    print(f"[candels] {n} rows -> {out.relative_to(ROOT)}\n  columns: {', '.join(t.columns[:20])} ...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=["gzh", "gaia", "candels", "all"],
                    help="which catalogue (default: all)")
    ap.add_argument("--ra", type=float, help="pointing centre RA [deg] (for gaia)")
    ap.add_argument("--dec", type=float, help="pointing centre Dec [deg] (for gaia)")
    ap.add_argument("--radius", type=float, default=0.06, help="cone radius [deg] (for gaia)")
    a = ap.parse_args()
    CATS.mkdir(parents=True, exist_ok=True)
    if a.what in ("gzh", "all"):
        gzh()
    if a.what == "all" and (a.ra is None or a.dec is None):
        print("[gaia] skipped: no --ra/--dec given. Either rerun with them, or set "
              "catalogs.gaia.query_online: true in config.yaml and the pipeline fetches Gaia itself.")
    elif a.what in ("gaia", "all"):
        if a.ra is None or a.dec is None:
            ap.error("gaia needs --ra and --dec (the centre of your pointing; tools/inspect_fits.py prints it)")
        gaia(a.ra, a.dec, a.radius)
    if a.what in ("candels", "all"):
        candels()


if __name__ == "__main__":
    main()
