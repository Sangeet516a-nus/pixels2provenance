"""Stage 1a - Archive acquisition: verify and catalogue the input files.

Purpose : know exactly which science products the run uses, and check them before any pixel is processed.
Method  : scan data/raw (files come from tools/get_fits.py), check each file has a SCI extension and a
          celestial WCS (rejected otherwise) and a WHT map, read the photometric keywords (PHOTFLAM/PHOTPLAM ->
          AB zero-point), pixel scale, exposure and epoch; group the files into pointings exactly as Stage 0
          does; choose the band set (blue -> red) and, per pointing, one file per band (_drc before _drz, then
          the deepest) and the detection band (reddest).
Output  : manifest.json (one entry per file) and pointings.json (pointing -> role, splits from Stage 0, bands,
          files, zero-points, detection band). Read by Stages 1b-9.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..context import Ctx, save_json
from ..fitsio import find_products, group_pointings, is_external, read_meta
from .stage0_partition import SkyRegionPartitioner


def choose_bands(pointings, metas, cfg):
    want = cfg["bands"]
    if want != "auto":
        return [b.upper() for b in want]
    counts = Counter(metas[i]["filter"] for p in pointings for i in p)
    lam = {m["filter"]: float(m["photplam"]) for m in metas if m["photplam"]}
    best = [b for b, _ in counts.most_common(cfg.get("max_bands", 3))]
    return sorted(best, key=lambda b: lam.get(b, 1e9))          # blue -> red


def run(ctx: Ctx):
    acfg = ctx.cfg["acquisition"]
    out = ctx.stage_dir("1a")
    files = find_products(ctx.raw, acfg["file_patterns"])
    if not files:
        raise FileNotFoundError(f"No HST products found in {ctx.raw} (download them with tools/get_fits.py).")

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
            if m["filter"] in by_band:
                ctx.log.info(f"[Stage 1a] WARNING {pid}: {m['name']} not used - {by_band[m['filter']]['name']} "
                             f"is the {m['filter']} file of this pointing (_drc first, then deepest)")
            by_band.setdefault(m["filter"], m)
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
        regs = {k: r for k, r in rules.regions.items() if k == pid or k.startswith(pid + "_")}
        names = {r.get("pointing") or (k[:-len(r["split_tag"]) - 1] if k.endswith("_" + r["split_tag"]) else k)
                 for k, r in regs.items()}                                   # e.g. P01_GOODS-S-2
        name = names.pop() if len(names) == 1 else pid
        pointings[pid] = {"name": name[len(pid) + 1:] if name.startswith(pid + "_") else name, "role": role,
                          "splits": sorted({r["split_tag"] for r in regs.values()}),
                          "bands": used, "missing_bands": [b for b in bands if b not in used],
                          "detection_band": det, "centre": cen,
                          "split_at_centre": rules.assign_split(*cen),
                          "instrument": by_band[det]["instrument"], "epoch": by_band[det]["epoch"],
                          "exptime_detection": by_band[det]["exptime"], "zp_ab": {b: by_band[b]["zp_ab"] for b in used},
                          "pixscale": {b: by_band[b]["pixscale_arcsec"] for b in used}}

    manifest = {m["name"]: dict(m) for m in metas}
    save_json(manifest, out / "manifest.json")
    save_json({"bands": bands, "pointings": pointings, "rejected": [r["name"] for r in rejected]},
              out / "pointings.json")
    for pid, p in pointings.items():
        ctx.log.info(f"[Stage 1a] {pid} {p['name']} ({p['role']}, splits {p['splits']}): bands {list(p['bands'])} "
                     f"detection={p['detection_band']} "
                     f"missing={p['missing_bands']} instrument={p['instrument']}")
    ctx.log.info(f"[Stage 1a] {len(metas)} verified file(s), {len(rejected)} rejected; band set = {bands}")
    return {"files": len(metas), "pointings": len(pointings), "bands": bands}
