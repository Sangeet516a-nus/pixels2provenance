#!/usr/bin/env python
"""Download HST _drc/_drz images (one per filter) for one or more sky positions - DOWNLOAD ONLY, no pipeline run.

Each target gets its own folder data/fields/<name>/, so fields never mix with data/raw.

    # look first: prints what was found per filter and which file would be taken (downloads nothing)
    python tools/get_fits.py --target 189.23,62.24,GOODS-N --list

    # 3 fields x 3 filters = 9 files
    python tools/get_fits.py --target 189.23,62.24,GOODS-N --target 53.16,-27.78,GOODS-S-2 --target 214.83,52.83,EGS

    # other instrument / own filters
    python tools/get_fits.py --target 189.23,62.24,GOODS-N --instrument WFC3/IR
    python tools/get_fits.py --target 189.23,62.24,GOODS-N --filters F435W F606W F850LP

Selection per filter (only the filter name is matched, so ACS entries such as "F814W;CLEAR2L" count):
  the image must cover the position; images made from several exposures (cosmic rays removed) come first.
  The filters are chosen TOGETHER: per filter the 15 images covering most of a 3.4' box on the target are
  kept; every combination is scored by the sky area covered by ALL filters (target inside it); among
  combinations within --area-tolerance (10 %) of the largest such area, those without an image centred
  more than --max-offset (2.5') away win, then the deepest. The printout gives that all-filter area. Skycell mosaics (several GB) are skipped. Use --allow-single to accept
  single-exposure images when a filter has nothing else.
To run the pipeline on a downloaded field afterwards:  python tools/run_field.py ... (or copy the folder's
files into data/raw/ after moving the old ones away).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_FILTERS = {  # first available of each colour group is taken
    "ACS/WFC": [["F435W", "F475W", "F555W"], ["F606W", "F625W"], ["F814W", "F775W", "F850LP"]],
    "WFC3/UVIS": [["F438W", "F475W", "F390W"], ["F606W", "F555W"], ["F814W", "F775W"]],
    "WFC3/IR": [["F105W", "F110W"], ["F125W", "F140W"], ["F160W"]],
}


def wavelength(band: str) -> float:
    import re
    m = re.search(r"(\d+)", band)
    return float(m.group(1)) if m else 0.0


def parse_target(s):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("use RA,Dec[,name] in degrees, e.g. 189.23,62.24,GOODS-N")
    from p2p.field import parse_coord
    ra, dec = parse_coord(parts[0], parts[1])
    name = parts[2] if len(parts) > 2 and parts[2] else f"ra{ra:.3f}_dec{dec:+.3f}"
    return ra, dec, name.replace(" ", "_").replace("/", "-")


def filter_set(v) -> set:
    return {x.strip().upper() for x in str(v).replace(",", ";").split(";") if x.strip()}


def candidates(obs, ra, dec, band, allow_single):
    """Observations of one filter that cover the position, best first."""
    import numpy as np
    from p2p.archive import contains, is_combined, product_kind
    df = obs[[band in filter_set(v) for v in obs["filters"]]].copy()
    n_all = len(df)
    if df.empty:
        return df, f"{band}: 0 images"
    df["kind"] = df.obs_id.map(product_kind)
    df = df[df.kind != "mosaic"]
    if df.empty:
        return df, f"{band}: {n_all} images, all of them multi-GB skycell mosaics (skipped)"
    df["combined"] = df.obs_id.map(is_combined).astype(bool)
    cosd = np.cos(np.radians(dec))
    df["offset_arcmin"] = 60 * np.hypot((df.s_ra - ra) * cosd, df.s_dec - dec)
    df["covers"] = np.array([contains(s, ra, dec) for s in df.s_region], dtype=bool)
    n_cov = int(df.covers.sum())
    # a footprint that cannot be parsed is accepted if its centre is within 1.5' (ACS/WFC is 3.4' wide)
    df = df[df.covers | (df.offset_arcmin < 1.5)]
    n_comb = int(df.combined.sum())
    if not allow_single:
        df = df[df.combined]
    df = df.sort_values(["combined", "covers", "t_exptime", "offset_arcmin"], ascending=[False, False, False, True])
    note = (f"{band}: {n_all} images, {n_cov} cover the position, {n_comb} of those combined"
            + ("" if allow_single else " (single exposures skipped)"))
    return df, note


def pick_product(Observations, row, ext):
    prods = Observations.get_product_list(str(row.obsid)).to_pandas()
    fn = prods.productFilename.astype(str)
    for e in ext:
        hit = prods[fn.str.lower().str.startswith(str(row.obs_id).lower()) & fn.str.lower().str.endswith(e)]
        if len(hit):
            return hit.iloc[0]
    return None


def coverage_masks(df, ra, dec, half_arcmin=3.0, step_arcmin=0.05):
    """Boolean coverage of each candidate footprint on a common grid around the target (local tangent plane)."""
    import numpy as np
    from matplotlib.path import Path as MPath
    from p2p.archive import _polygons
    g = np.arange(-half_arcmin, half_arcmin + 1e-9, step_arcmin)
    xx, yy = np.meshgrid(g, g)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    cosd = np.cos(np.radians(dec))
    masks = []
    for s_region in df.s_region:
        m = np.zeros(len(pts), bool)
        for poly in _polygons(s_region):
            q = poly.copy()
            q[:, 0] = ((q[:, 0] - ra + 180) % 360 - 180) * cosd * 60
            q[:, 1] = (q[:, 1] - dec) * 60
            m |= MPath(q).contains_points(pts)
        masks.append(m)
    centre = int(np.argmin((pts ** 2).sum(1)))
    return masks, centre, step_arcmin ** 2


def central_box(ra, dec, half_arcmin=3.0, step_arcmin=0.05, box_half=1.7):
    import numpy as np
    g = np.arange(-half_arcmin, half_arcmin + 1e-9, step_arcmin)
    xx, yy = np.meshgrid(g, g)
    return ((np.abs(xx) <= box_half) & (np.abs(yy) <= box_half)).ravel()


def choose_jointly(obs, ra, dec, groups, a, Observations, ext, per_filter=15):
    import itertools
    import numpy as np
    cands = []
    for group in groups:
        rows = []
        for rank, band in enumerate(group):
            df, note = candidates(obs, ra, dec, band, a.allow_single)
            print("    " + note)
            if not df.empty:
                rows.append(df.assign(band=band, band_rank=rank))
        c = pd_concat(rows)
        if c is not None and len(c) > per_filter:
            # keep the images that cover the most of a 3.4' box centred on the target (an ACS field placed on
            # the target): a shallow image aligned with the others beats a deep one 2' away
            m, _, _ = coverage_masks(c, ra, dec)
            box = central_box(ra, dec)
            c = c.assign(central=[round(float((mi & box).sum() / box.sum()), 2) for mi in m])
            c = c.sort_values(["combined", "central", "t_exptime"], ascending=[False, False, False]).head(per_filter)
        cands.append(c)
    if any(c is None or c.empty for c in cands):
        missing = [g for g, c in zip(groups, cands) if c is None or c.empty]
        for g in missing:
            print(f"    -> nothing usable for {'/'.join(g)}" + ("" if a.allow_single else "  (try --allow-single)"))
        cands = [c for c in cands if c is not None and not c.empty]
        if not cands:
            return {}
    masks = [coverage_masks(c, ra, dec) for c in cands]
    blocked = [set() for _ in cands]
    while True:
        best = None
        for combo in itertools.product(*[range(len(c)) for c in cands]):
            if any(i in blocked[k] for k, i in enumerate(combo)):
                continue
            common = np.logical_and.reduce([masks[k][0][i] for k, i in enumerate(combo)])
            if not common[masks[0][1]]:
                continue                                        # the target itself must have every band
            area = float(common.sum() * masks[0][2])
            depth = float(sum(np.log10(max(cands[k].iloc[i].t_exptime, 1)) for k, i in enumerate(combo)))
            pref = -sum(int(cands[k].iloc[i].band_rank) for k, i in enumerate(combo))
            single = sum(not bool(cands[k].iloc[i].combined) for k, i in enumerate(combo))
            # images centred far from the target are larger multi-pointing products: most of their area has one
            # band only (single-band detections = cosmic-ray candidates that fill the source cap)
            far = sum(float(cands[k].iloc[i].offset_arcmin) > a.max_offset for k, i in enumerate(combo))
            best = best or []
            best.append((combo, area, depth, pref, single, common, far))
        if not best:
            print("    -> no combination of images covers the target in every filter")
            return {}
        top = max(b[1] for b in best)
        ok = [b for b in best if b[1] >= (1 - a.area_tolerance) * top]
        combo, area, depth, _, _, common, _ = max(ok, key=lambda b: (-b[4], -b[6], b[3], b[2], b[1]))
        picked, failed = {}, False
        for k, i in enumerate(combo):
            row = cands[k].iloc[i]
            p = pick_product(Observations, row, ext)
            if p is None:
                blocked[k].add(i)
                failed = True
                break
            frac = float((masks[k][0][i] & common).sum() / max(masks[k][0][i].sum(), 1))
            row = row.copy()
            row["ov"] = frac
            picked[k] = (row.band, row, p)
        if failed:
            continue
        print(f"    area covered by ALL {len(combo)} filters: {area:.1f} arcmin^2 "
              f"({area / 11.56:.0%} of one ACS/WFC field); best possible here {top:.1f} arcmin^2")
        return picked


def pd_concat(rows):
    import pandas as pd
    return pd.concat(rows) if rows else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", type=parse_target, required=True, help="RA,Dec[,name]")
    ap.add_argument("--instrument", default="ACS/WFC", choices=list(DEFAULT_FILTERS))
    ap.add_argument("--filters", nargs="+", help="exact filters, e.g. F435W F606W F814W (default: one per colour)")
    ap.add_argument("--radius-arcmin", type=float, default=2.0, help="search radius around the position")
    ap.add_argument("--allow-single", action="store_true", help="accept single-exposure images (cosmic rays!)")
    ap.add_argument("--out", default=str(ROOT / "data" / "fields"), help="one sub-folder per target is made here")
    ap.add_argument("--list", action="store_true", help="only show what would be downloaded")
    ap.add_argument("--gaia", action="store_true", help="also save the Gaia DR3 stars around each target")
    ap.add_argument("--max-offset", type=float, default=2.5,
                    help="images centred further than this (arcmin) are avoided when an alternative exists")
    ap.add_argument("--area-tolerance", type=float, default=0.10,
                    help="combinations within this fraction of the largest all-band area compete on depth")
    a = ap.parse_args()

    from astroquery.mast import Observations
    groups = [[f.upper()] for f in a.filters] if a.filters else DEFAULT_FILTERS[a.instrument]
    ext = ["_drz.fits", "_drc.fits"] if a.instrument == "WFC3/IR" else ["_drc.fits", "_drz.fits"]
    total = 0
    for ra, dec, name in a.target:
        print(f"\n=== {name}: RA {ra:.5f}  Dec {dec:+.5f}  ({a.instrument}, radius {a.radius_arcmin}')")
        obs = Observations.query_criteria(coordinates=f"{ra} {dec}", radius=f"{a.radius_arcmin / 60} deg",
                                          obs_collection="HST", instrument_name=a.instrument,
                                          dataproduct_type="image", calib_level=3)
        obs = obs.to_pandas()
        print(f"    {len(obs)} level-3 {a.instrument} images found nearby; filters present: "
              f"{', '.join(sorted(set().union(*[filter_set(v) for v in obs['filters']]) - {'CLEAR1L', 'CLEAR2L', 'CLEAR1S', 'CLEAR2S'})) or '-'}")
        out_dir = Path(a.out) / name
        # The images of the different filters are chosen TOGETHER: every combination of the best candidates
        # (one per filter) is scored by the sky area that ALL filters cover (the target must be inside it).
        # Filters taken by different HST programmes are often offset by 1-2' (an ACS field is 3.4' wide):
        # picking each filter by exposure alone can leave much of the field with one or two bands, where a
        # source has no colours and a one-band source counts as a cosmic-ray candidate. Among combinations
        # within --area-tolerance of the largest common area, the deepest (sum of log exposure) wins.
        picked = choose_jointly(obs, ra, dec, groups, a, Observations, ext)
        for gi in sorted(picked):
            band, row, p = picked[gi]
            dest = out_dir / str(p.productFilename)
            kind = "combined" if row.combined else "SINGLE exposure"
            print(f"    -> {band}: {p.productFilename}  [{kind}, {row.t_exptime:.0f} s, "
                  f"{row.offset_arcmin:.1f}' from centre, {row.ov:.0%} of this image has all filters, "
                  f"{float(p['size']) / 1e6:.0f} MB]")
            if a.list:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                print("       already downloaded")
            else:
                status, msg, _ = Observations.download_file(str(p.dataURI), local_path=str(dest))
                print(f"       download: {status} {msg or ''}")
                if status != "COMPLETE":
                    continue
            total += 1
        if a.gaia and not a.list:
            from p2p.archive import gaia_cone
            gaia_cone(ra, dec, 0.06, out_dir)
    if not a.list:
        print(f"\n{total} file(s) in {a.out}/<field name>/")


if __name__ == "__main__":
    main()
