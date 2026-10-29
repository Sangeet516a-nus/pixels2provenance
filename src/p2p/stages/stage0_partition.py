"""Stage 0 - Data governance: sky-region partition with a leakage collar.

Purpose : fix the train / dev / test / external split BY SKY REGION before any pixel is analysed. Neighbouring
          galaxies share pixels, background and PSF, so a random split of OBJECTS would leak; splitting the SKY
          does not. The rule is written once (split_rule.json) and every later stage only reads it.

Method  : 1. Each pointing (three files F435W / F606W / F775W, grouped as in Stage 1a) gets a REGION = its real
             data footprint: the convex hull of the pixels with data (WHT > 0) in ALL three bands. A drizzled ACS
             frame is a tilted quadrilateral inside a larger zero-weight array, so its RA/Dec box would be ~2x
             too big and overlap the neighbouring pointings.
          2. partition.pointing_splits (config.yaml) sets the split of each pointing, matched by file-name
             substrings: either the whole pointing (split: train | dev | test | external) or stripes along the
             pointing's longer axis (stripes: {train: 0.25, dev: 0.50, external: 0.25}). Stripe sizes follow
             the share of the pointing's GZH galaxies (partition.stripe_share: gzh_galaxies - POSITIONS only,
             never classes), so "20 %" means 20 % of the labelled galaxies, not 20 % of the area (an area stripe
             can fall outside the GZH coverage and hold no labels). Stripes are clipped to the footprint.
             With an empty pointing_splits list (e.g. a new field in apply mode) every pointing listed in
             partition.external_patterns is external and every other one is cut into fractions (70/15/15)
             by data area.
          3. A source gets the split of the region(s) it lies in. It gets NO split ('buffer') if it lies outside
             every region, or inside / within the collar (partition.collar_arcmin) of regions with DIFFERENT
             splits - the leakage guard. Overlapping regions of the SAME split keep the source.

Plan (Sangeeta, 28 Oct). Each split mixes GOODS-South and GOODS-North pointings from different HST
programmes, epochs and depths, so train, dev and test see the same variety of data. GZH catalogue entries
(positions only) per split, from the real footprints of the 24 files:

    Pointing    train   dev   test   external   no split
    GOODS-S        61   121      -         68         49   stripes 25 / 50 / 25 %
    GOODS-S-2       -    30    129          -         14   stripes 80 (test) / 20 (dev) %
    GOODS-S-3     342     -      -          -         12
    GOODS-S-4       -     -      -        319         34   Phase 7 (Kartaltepe human labels)
    GOODS-N-1     383     -      -          -         12
    GOODS-N-2       -     -      -        320         29   held-out field
    GOODS-N-3     198   168      -          -         58   stripes 50 / 50 %
    GOODS-N-4       -     -    294          -         11
    Total         984   319    423        707        219   (no split = collar or regions of different splits)

Cross-validation (Stages 4a-4d) trains on train + dev. Test and external stay sealed until the model is final
(evaluation.test_sealed). Every run writes this table to split_rule.json ("gzh_per_pointing") and the log.

Output  : outputs/stage0_partition/split_rule.json, split_regions.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import Ctx, load_json, save_json
from ..fitsio import find_products, group_pointings, is_external, read_meta

SPLITS = ["train", "dev", "test", "external"]


class SkyRegionPartitioner:
    """Polygon regions with a split tag; assigns a split (or None = buffer) to sky positions."""

    def __init__(self, collar_arcmin: float):
        self.buffer_deg = collar_arcmin / 60.0
        self.regions: dict[str, dict] = {}

    def define_region(self, name, polygon, split, pointing):
        poly = np.asarray(polygon, float)
        self.regions[name] = {"split_tag": split, "pointing": pointing,
                              "ra_range": [float(poly[:, 0].min()), float(poly[:, 0].max())],
                              "dec_range": [float(poly[:, 1].min()), float(poly[:, 1].max())],
                              "polygon": poly.tolist()}

    def assign_many(self, ra, dec):
        """Split of every position: the region's tag if all regions it lies in, or is within the collar of,
        share one tag; otherwise (or outside every region) None."""
        from matplotlib.path import Path as MPath
        ra, dec = np.atleast_1d(np.asarray(ra, float)), np.atleast_1d(np.asarray(dec, float))
        inside = [set() for _ in ra]
        near = [set() for _ in ra]
        buf = self.buffer_deg * 60.0                                       # arcmin
        for reg in self.regions.values():
            poly = np.asarray(reg.get("polygon") or _box_polygon(reg), float)
            P, X = _tangent_arcmin(poly, ra, dec)
            idx = np.where(np.hypot(*X.T) < np.hypot(*P.T).max() + buf + 1e-9)[0]   # cheap pre-selection
            if not len(idx):
                continue
            ins = MPath(P).contains_points(X[idx])
            close = ins | (_dist_to_edges(X[idx], P) < buf)
            for i, a, b in zip(idx, ins, close):
                if a:
                    inside[i].add(reg["split_tag"])
                if b:
                    near[i].add(reg["split_tag"])
        return [next(iter(n)) if i and len(n) == 1 else None for i, n in zip(inside, near)]

    def assign_split(self, ra, dec):
        return self.assign_many([ra], [dec])[0]

    def to_dict(self):
        return {"buffer_deg": self.buffer_deg, "regions": self.regions}

    @classmethod
    def from_file(cls, path):
        d = load_json(path)
        p = cls(d["buffer_deg"] * 60.0)
        p.regions = d["regions"]
        return p


# ---------------------------------------------------------------- geometry
def _box_polygon(reg):
    """Regions of split_rule.json files written before 28 Oct have only an RA/Dec box."""
    (r0, r1), (d0, d1) = reg["ra_range"], reg["dec_range"]
    return [[r0, d0], [r1, d0], [r1, d1], [r0, d1]]


def _tangent_arcmin(poly, ra, dec):
    """Polygon and points in a flat (arcmin) frame centred on the polygon (RA wrap and cos(dec) handled)."""
    ra0, dec0 = poly[:, 0].mean(), poly[:, 1].mean()
    cosd = np.cos(np.deg2rad(dec0))
    to = lambda r, d: np.column_stack([((r - ra0 + 180) % 360 - 180) * cosd * 60, (d - dec0) * 60])
    return to(poly[:, 0], poly[:, 1]), to(ra, dec)


def _dist_to_edges(X, P):
    """Distance from points X (n, 2) to the outline of the closed polygon P (m, 2), same units."""
    AB = np.roll(P, -1, axis=0) - P
    t = np.clip(((X[:, None] - P[None]) * AB[None]).sum(2) / np.maximum((AB ** 2).sum(1), 1e-12), 0, 1)
    return np.sqrt(((X[:, None] - (P[None] + t[..., None] * AB[None])) ** 2).sum(2)).min(1)


def coverage_points(ms, stride=8, log=print):
    """Sky positions of pixels with data (WHT > 0) in EVERY file of a pointing, every `stride` pixels.
    Only coverage is read here, never sources."""
    from ..fitsio import load_image, valid_mask
    ref = max(ms, key=lambda m: m["naxis1"] * m["naxis2"])
    sci, wht, w = load_image(ref["file"])
    yy, xx = np.mgrid[0:sci.shape[0]:stride, 0:sci.shape[1]:stride]
    ok = valid_mask(sci, wht)[yy, xx]
    ra, dec = w.pixel_to_world_values(xx[ok], yy[ok])
    keep = np.ones(len(ra), bool)
    for m in ms:
        if m is not ref:
            s2, w2, wcs2 = load_image(m["file"])
            xi, yi = (np.round(v).astype(int) for v in wcs2.world_to_pixel_values(ra, dec))
            inb = (xi >= 0) & (yi >= 0) & (xi < s2.shape[1]) & (yi < s2.shape[0])
            v = np.zeros(len(ra), bool)
            v[inb] = valid_mask(s2, w2)[yi[inb], xi[inb]]
            keep &= v
    if keep.sum() < 0.05 * len(keep):            # bands hardly overlap: use the largest file's coverage
        log(f"[Stage 0] WARNING: the bands of {ref['name']} overlap on only {keep.mean():.1%} of its area; "
            "its region is that file's coverage alone.")
        keep[:] = True
    return np.asarray(ra)[keep], np.asarray(dec)[keep]


def footprint_polygon(ra, dec):
    """Convex hull of the covered pixels (the intersection of the bands' convex data areas is convex)."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(np.column_stack([(ra - ra.mean()) * np.cos(np.deg2rad(dec.mean())), dec]))
    return np.column_stack([ra, dec])[hull.vertices]


def stripe_boxes(ra, dec, fractions, wra=None, wdec=None):
    """Consecutive stripes along the longer axis of the data area (ra, dec = covered pixels), in the order of
    `fractions`. Cuts at the quantiles of the weight points (wra, wdec = GZH positions) if given, else of the
    covered pixels (equal data area). The outer stripes reach the edge of the data area."""
    along_ra = np.ptp(ra) * np.cos(np.deg2rad(np.median(dec))) >= np.ptp(dec)
    t = ra if along_ra else dec
    w = t if wra is None else (wra if along_ra else wdec)
    cum = np.cumsum(list(fractions.values()))
    cuts = np.concatenate([[t.min()], np.quantile(w, cum[:-1] / cum[-1]), [t.max()]])
    r0, r1, d0, d1 = ra.min(), ra.max(), dec.min(), dec.max()
    return [(tag, (a, b, d0, d1) if along_ra else (r0, r1, a, b))
            for tag, a, b in zip(fractions, cuts[:-1], cuts[1:])]


def clip_to_box(poly, box):
    """Sutherland-Hodgman: the part of a convex polygon inside an RA/Dec box (one stripe of a pointing)."""
    r0, r1, d0, d1 = box
    out = [tuple(p) for p in np.asarray(poly, float)]
    for axis, lim, keep_greater in ((0, r0, True), (0, r1, False), (1, d0, True), (1, d1, False)):
        inp, out = out, []
        inside = (lambda q: q[axis] >= lim) if keep_greater else (lambda q: q[axis] <= lim)
        for prev, cur in zip(inp[-1:] + inp[:-1], inp):
            if inside(cur) != inside(prev):                       # the edge crosses the limit
                t = (lim - prev[axis]) / (cur[axis] - prev[axis])
                out.append(tuple(prev[j] + t * (cur[j] - prev[j]) for j in range(2)))
            if inside(cur):
                out.append(cur)
    return np.asarray(out, float)


# ---------------------------------------------------------------- the split plan
def pointing_rules(metas, groups, cfg):
    """One rule per pointing: {'name', 'split'} or {'name', 'stripes'}. From partition.pointing_splits, or, if
    that list is empty, external_patterns -> external and every other pointing -> fractions (70/15/15)."""
    plan, ext_pat = cfg.get("pointing_splits") or [], cfg.get("external_patterns") or []
    rules, problems = [], []
    for gi, g in enumerate(groups):
        ms = [metas[i] for i in g]
        pid = f"P{gi + 1:02d}"
        is_ext = any(is_external(m, ext_pat) for m in ms)
        if not plan:
            rules.append({"name": pid, "split": "external"} if is_ext else {"name": pid, "stripes": cfg["fractions"]})
            continue
        text = " ".join(m["name"] + " " + m.get("targname", "") for m in ms).lower()
        hit = [r for r in plan if any(str(x).lower() in text for x in r["match"])]
        if len(hit) != 1:
            problems.append(f"{pid} ({', '.join(m['name'] for m in ms)}) matches {len(hit)} entries (needs 1)")
        elif (hit[0].get("split") == "external") != is_ext:
            problems.append(f"{pid} ({hit[0]['name']}): split '{hit[0].get('split', 'stripes')}', but external_"
                            f"patterns {'match' if is_ext else 'do not match'} it - external_patterns must list "
                            "exactly the wholly external pointings (Stage 1a uses them)")
        else:
            rules.append(hit[0])
    unused = [r["name"] for r in plan if not any(r is q for q in rules)]
    if unused:
        problems.append(f"entries match no file in data/raw: {unused}")
    if problems:
        raise RuntimeError("[Stage 0] partition.pointing_splits does not fit the files in data/raw:\n  - "
                           + "\n  - ".join(problems))
    return rules


def build_partition(metas, groups, cfg, gzh_radec=None, log=print):
    from matplotlib.path import Path as MPath
    part = SkyRegionPartitioner(cfg["collar_arcmin"])
    by_gzh = cfg.get("stripe_share", "gzh_galaxies") == "gzh_galaxies" and gzh_radec is not None
    for gi, (g, rule) in enumerate(zip(groups, pointing_rules(metas, groups, cfg))):
        pid = f"P{gi + 1:02d}"
        name = f"{pid}_{rule['name']}".replace(" ", "_") if rule["name"] != pid else pid
        ra, dec = coverage_points([metas[i] for i in g], log=log)
        poly = footprint_polygon(ra, dec)
        if "split" in rule:
            part.define_region(name, poly, rule["split"], name)
            continue
        w, share = (None, None), "data area"
        if by_gzh:
            ins = MPath(poly).contains_points(np.column_stack(gzh_radec))
            if ins.sum() >= 20:                  # too few labelled galaxies: equal data area instead
                w, share = (gzh_radec[0][ins], gzh_radec[1][ins]), f"GZH galaxies ({int(ins.sum())} in the pointing)"
        log(f"[Stage 0] {name}: stripes {rule['stripes']} by share of {share}")
        for tag, box in stripe_boxes(ra, dec, rule["stripes"], *w):
            sp = clip_to_box(poly, box)
            if len(sp) >= 3:
                part.define_region(f"{name}_{tag}", sp, tag, name)
    mode = "planned (partition.pointing_splits)" if cfg.get("pointing_splits") else "automatic (fractions)"
    return part, mode


# ---------------------------------------------------------------- stage entry
def run(ctx: Ctx):
    cfg = ctx.cfg["partition"]
    out = ctx.stage_dir("0")
    files = find_products(ctx.raw, ctx.cfg["acquisition"]["file_patterns"])
    if not files:
        raise FileNotFoundError(f"No *_drc.fits / *_drz.fits files in {ctx.raw}. Put your HST files there "
                                "(python tools/get_fits.py ...).")
    metas = [m for m in (read_meta(f) for f in files) if m["wcs_ok"] and m["has_sci"]]   # as Stage 1a
    acq = ctx.cfg["acquisition"]
    groups = group_pointings(metas, acq["group_radius_arcsec"], acq.get("group_min_overlap", 0.5))
    for gi, g in enumerate(groups):
        ctx.log.info(f"[Stage 0] pointing P{gi + 1:02d}: {[metas[i]['filter'] for i in g]} "
                     f"({', '.join(metas[i]['name'] for i in g)})")

    from ..catalogs import load_gzh               # POSITIONS only (no classes): where the labelled galaxies are
    g = load_gzh(ctx, metas, required=False)
    gzh_radec = (g.ra.to_numpy(float), g.dec.to_numpy(float)) if g is not None and len(g) else None

    part, mode = build_partition(metas, groups, cfg, gzh_radec, log=ctx.log.info)
    rule = {**part.to_dict(), "mode": mode}
    if gzh_radec is not None:
        table = _gzh_per_pointing(part, gzh_radec)
        rule["gzh_per_pointing"] = table.to_dict(orient="index")
        ctx.log.info("[Stage 0] GZH galaxies (positions, no classes) per pointing and split:\n" + table.to_string())
    save_json(rule, out / "split_rule.json")
    _plot(part, metas, out / "split_regions.png")
    tags = pd.Series([r["split_tag"] for r in part.regions.values()]).value_counts().to_dict()
    ctx.log.info(f"[Stage 0] {len(part.regions)} regions ({mode}); collar {cfg['collar_arcmin']}'; tags {tags}")
    ctx.log.info(f"[Stage 0] Ruleset frozen -> {ctx.rel(out / 'split_rule.json')}")
    return {"mode": mode, "n_regions": len(part.regions)}


def _gzh_per_pointing(part, gzh_radec):
    """GZH galaxies per pointing and split (a galaxy inside two pointings of the same split counts once, for
    the first); 'no split' = inside a pointing but in a collar or in regions of different splits."""
    from matplotlib.path import Path as MPath
    ra, dec = gzh_radec
    split = pd.Series(part.assign_many(ra, dec), dtype=object).fillna("no split").to_numpy()
    where = np.full(len(ra), None, dtype=object)
    for reg in part.regions.values():
        P, X = _tangent_arcmin(np.asarray(reg["polygon"]), ra, dec)
        where[(where == None) & MPath(P).contains_points(X)] = reg["pointing"]    # noqa: E711
    keep = where != None                                                      # noqa: E711
    t = pd.crosstab(where[keep], split[keep]).rename_axis(index="pointing", columns="split")
    t = t.reindex(columns=[s for s in SPLITS + ["no split"] if s in t.columns], fill_value=0)
    t.loc["Total"] = t.sum()
    return t


def _plot(part, metas, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    colours = {"train": "#2E5496", "dev": "#E69F00", "test": "#009E73", "external": "#CC79A7"}
    fields = []                                   # one panel per sky field (regions > 1 degree apart)
    for r in part.regions.values():
        c = (np.mean(r["ra_range"]), np.mean(r["dec_range"]))
        f = next((f for f in fields if abs(f["c"][0] - c[0]) * np.cos(np.deg2rad(c[1])) < 1
                  and abs(f["c"][1] - c[1]) < 1), None)
        if f is None:
            fields.append(f := {"c": c, "regs": []})
        f["regs"].append(r)
    fig, axes = plt.subplots(1, len(fields), figsize=(5.2 * len(fields), 5), squeeze=False)
    for ax, f in zip(axes[0], fields):
        for m in metas:
            fp = np.asarray(m["footprint"])
            if abs(fp[:, 1].mean() - f["c"][1]) < 1:
                ax.add_patch(Polygon(fp, fill=False, ec="0.6", lw=0.6))
        pointings = {}
        for r in f["regs"]:
            poly = np.asarray(r.get("polygon") or _box_polygon(r))
            ax.add_patch(Polygon(poly, alpha=0.35, color=colours.get(r["split_tag"], "grey")))
            ax.text(poly[:, 0].mean(), poly[:, 1].mean(), r["split_tag"], ha="center", va="center", fontsize=7)
            pointings.setdefault(r.get("pointing", "?"), []).append(poly)
        for name, polys in pointings.items():    # pointing name once, above its footprint
            p = np.vstack(polys)
            ax.text(p[:, 0].mean(), p[:, 1].max(), name, ha="center", va="bottom", fontsize=7, weight="bold")
        ax.autoscale()
        ax.invert_xaxis()
        ax.set_aspect(1 / np.cos(np.deg2rad(f["c"][1])))
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
    for tag, c in colours.items():
        axes[0][0].fill([], [], color=c, alpha=0.35, label=tag)
    axes[0][0].legend(fontsize=7, loc="lower left", frameon=False)
    fig.suptitle("Stage 0: sky regions by split (grey outlines = file footprints)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
