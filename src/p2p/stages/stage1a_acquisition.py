"""Stage 1a - Archive acquisition (local scan, optional MAST download).

Purpose : know exactly which science products the run uses and verify them.
Method  : (optional) download Level-3 _drc products from MAST with astroquery;
          then scan data/raw, prefer _drc over _drz, check each file has a SCI
          extension, a WHT map and a celestial WCS, read the photometric
          keywords (PHOTFLAM/PHOTPLAM -> AB zero-point), group files into
          pointings and choose the band set used for the 3-channel cutouts.
Output  : manifest.json (one entry per file) and pointings.json
          (pointing -> role, bands, detection band, files).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..context import Ctx, save_json
from ..fitsio import find_products, group_pointings, is_external, read_meta
from .stage0_partition import SkyRegionPartitioner


def download_from_mast(ctx: Ctx):
    """Optional: fetch _drc products for the targets in config.yaml (needs internet)."""
    dl = ctx.cfg["acquisition"]["download"]
    try:
        from astroquery.mast import Observations
    except Exception:
        ctx.log.info("[Stage 1a] astroquery missing - pip install astroquery to download.")
        return
    ctx.raw.mkdir(parents=True, exist_ok=True)
    for t in dl["targets"]:
        obs = Observations.query_criteria(coordinates=f"{t['ra']} {t['dec']}", radius=f"{t['radius_deg']} deg",
                                          obs_collection="HST", dataproduct_type="image", calib_level=3,
                                          instrument_name=t.get("instrument", "ACS/WFC"),
                                          filters=t.get("filters"))
        ctx.log.info(f"[Stage 1a] MAST: {len(obs)} observations near ({t['ra']}, {t['dec']})")
        if len(obs) == 0:
            continue
        prods = Observations.filter_products(Observations.get_product_list(obs), productType="SCIENCE",
                                             productSubGroupDescription="DRC", extension="fits")
        # one product per filter, smallest first
        prods.sort("size")
        seen, keep = set(), []
        for i, row in enumerate(prods):
            f = str(row["productFilename"])
            filt = next((x for x in t.get("filters", []) if x.lower() in f.lower()), f)
            if filt not in seen:
                seen.add(filt)
                keep.append(i)
            if len(keep) >= dl.get("max_files_per_target", 3):
                break
        if keep:
            man = Observations.download_products(prods[keep], download_dir=str(ctx.raw), flat=True)
            ctx.log.info(f"[Stage 1a] downloaded: {[str(p) for p in man['Local Path']]}")


def choose_bands(pointings, metas, cfg):
    want = cfg["bands"]
    if want != "auto":
        return [b.upper() for b in want]
    counts = Counter(metas[i]["filter"] for p in pointings for i in p)
    lam = {}
    for m in metas:
        if m["photplam"]:
            lam[m["filter"]] = float(m["photplam"])
    best = [b for b, _ in counts.most_common(cfg.get("max_bands", 3))]
    return sorted(best, key=lambda b: lam.get(b, 1e9))          # blue -> red


def run(ctx: Ctx):
    acfg = ctx.cfg["acquisition"]
    out = ctx.stage_dir("1a")
    if acfg["download"].get("enabled"):
        download_from_mast(ctx)
    files = find_products(ctx.raw, acfg["file_patterns"])
    if not files:
        raise FileNotFoundError(f"No HST products found in {ctx.raw}.")

    metas, rejected = [], []
    for f in files:
        m = read_meta(f)
        problems = [k for k in ("has_sci", "wcs_ok") if not m[k]]
        if not m["has_wht"]:
            problems.append("no WHT (coverage will be taken from finite SCI pixels)")
        m["problems"] = problems
        (rejected if ("has_sci" in problems or "wcs_ok" in problems) else metas).append(m)
    if not metas:
        raise RuntimeError(f"All files failed the structural check: {[r['name'] for r in rejected]}")

    groups = group_pointings(metas, acfg["group_radius_arcsec"], acfg.get("group_min_overlap", 0.5))
    bands = choose_bands(groups, metas, acfg)
    rules = SkyRegionPartitioner.from_file(ctx.require("0", "split_rule.json"))
    pointings = {}
    for gi, g in enumerate(groups):
        pid = f"P{gi + 1:02d}"
        ms = [metas[i] for i in g]
        by_band = {}
        for m in sorted(ms, key=lambda m: (m["product_type"] != "drc", -m["exptime"])):
            by_band.setdefault(m["filter"], m)          # deepest _drc per band
        used = {b: by_band[b]["file"] for b in bands if b in by_band}
        if not used:
            ctx.log.info(f"[Stage 1a] {pid}: none of the bands {bands} present - pointing skipped")
            continue
        if acfg["detection_band"] != "auto" and acfg["detection_band"] in used:
            det = acfg["detection_band"]
        else:
            det = max(used, key=lambda b: by_band[b]["photplam"] or 0)          # reddest
        role = "external" if any(is_external(m, ctx.cfg["partition"].get("external_patterns")) for m in ms) else "gzh"
        cen = (float(np.mean([m["ra_centre"] for m in ms])), float(np.mean([m["dec_centre"] for m in ms])))
        for m in ms:
            m["pointing"] = pid
        pointings[pid] = {"role": role, "bands": used, "missing_bands": [b for b in bands if b not in used],
                          "detection_band": det, "centre": cen,
                          "split_at_centre": rules.assign_split(*cen),
                          "instrument": by_band[det]["instrument"], "epoch": by_band[det]["epoch"],
                          "exptime_detection": by_band[det]["exptime"], "zp_ab": {b: by_band[b]["zp_ab"] for b in used},
                          "pixscale": {b: by_band[b]["pixscale_arcsec"] for b in used}}

    manifest = {m["name"]: {k: v for k, v in m.items()} for m in metas}
    save_json(manifest, out / "manifest.json")
    save_json({"bands": bands, "pointings": pointings, "rejected": [r["name"] for r in rejected]},
              out / "pointings.json")
    for pid, p in pointings.items():
        ctx.log.info(f"[Stage 1a] {pid} ({p['role']}): bands {list(p['bands'])} detection={p['detection_band']} "
                     f"missing={p['missing_bands']} instrument={p['instrument']}")
    ctx.log.info(f"[Stage 1a] {len(metas)} verified file(s), {len(rejected)} rejected; band set = {bands}")
    return {"files": len(metas), "pointings": len(pointings), "bands": bands}
