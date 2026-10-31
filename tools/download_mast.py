#!/usr/bin/env python
"""Download the best Level-3 HST _drc products for a sky position from MAST into data/raw (needs internet).

    python tools/download_mast.py --ra 53.12 --dec -27.80 --dry-run          # show what would be downloaded
    python tools/download_mast.py --ra 53.12 --dec -27.80                    # one file per colour (auto filters)
    python tools/download_mast.py --ra 53.12 --dec -27.80 --filters F435W F606W F814W

Per filter, the file must CONTAIN the position; combined products (several exposures, cosmic rays rejected)
are preferred over single exposures, then the longest exposure. The other bands prefer files that overlap the
red (detection) band's file, so all bands form one pointing. See src/p2p/archive.py.
To search, download AND run the pipeline in one go, use tools/run_field.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ra", type=float, required=True)
    ap.add_argument("--dec", type=float, required=True)
    ap.add_argument("--radius", type=float, default=0.03, help="search radius in degrees")
    ap.add_argument("--instrument", default="ACS/WFC", help="ACS/WFC | WFC3/UVIS | WFC3/IR")
    ap.add_argument("--filters", nargs="+", default=["auto"], help="'auto' = one per colour, or a list e.g. F435W F606W F814W")
    ap.add_argument("--single-ok", action="store_true", default=True, help=argparse.SUPPRESS)
    ap.add_argument("--combined-only", action="store_true", help="never take single-exposure files")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--dry-run", action="store_true", help="list what would be downloaded")
    a = ap.parse_args()
    from p2p import archive
    filters = "auto" if a.filters == ["auto"] else [f.upper() for f in a.filters]
    ch = archive.search(a.ra, a.dec, filters, a.instrument, a.radius, allow_single=not a.combined_only)
    if not ch:
        print("nothing found that covers this position")
        return
    if not a.dry_run:
        archive.download(ch, a.out)


if __name__ == "__main__":
    main()
