"""Stage 0 - Data governance: sky-region partitioning with a leakage collar.

Purpose : fix the train / dev / test / external split BY SKY REGION before any
          pixel is analysed, so overlapping pointings cannot leak one object
          into two buckets.
Method  : auto mode reads only the WCS headers of the files in data/raw and
          builds RA/Dec boxes:
            * >= N GZH pointings  -> whole pointings are assigned to buckets
                                     (largest-deficit rule, 70/15/15 by area);
            * fewer pointings     -> each pointing is cut into stripes along its
                                     longer axis so that each stripe holds 70/15/15
                                     of the area that HAS DATA (WHT > 0 in every band;
                                     the empty corners of a drizzled frame don't count).
          Pointings matching partition.external_patterns become 'external'
          (CANDELS cross-catalogue test). A collar (cos(dec)-corrected) is kept
          around every box; a source in a collar or in two boxes gets no split.
Output  : outputs/stage0_partition/split_rule.json (+ a footprint plot)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import Ctx, load_json, save_json
from ..fitsio import bbox_of, find_products, group_pointings, is_external, read_meta


class SkyRegionPartitioner:
    def __init__(self, buffer_arcmin: float):
        self.buffer_deg = buffer_arcmin / 60.0
        self.regions: dict[str, dict] = {}

    def define_region(self, name, ra_min, ra_max, dec_min, dec_max, split):
        self.regions[name] = {"split_tag": split, "ra_range": [ra_min, ra_max], "dec_range": [dec_min, dec_max]}

    def assign_split(self, ra, dec):
        inner = None
        for reg in self.regions.values():
            r0, r1 = reg["ra_range"]
            d0, d1 = reg["dec_range"]
            cosd = np.cos(np.deg2rad(0.5 * (d0 + d1)))
            b_ra = self.buffer_deg / max(cosd, 1e-3)
            in_inner = (r0 <= ra <= r1) and (d0 <= dec <= d1)
            in_outer = (r0 - b_ra <= ra <= r1 + b_ra) and (d0 - self.buffer_deg <= dec <= d1 + self.buffer_deg)
            if in_inner:
                if inner is not None:
                    return None
                inner = reg["split_tag"]
            elif in_outer:
                return None
        return inner

    def assign_many(self, ra, dec):
        return [self.assign_split(float(r), float(d)) for r, d in zip(ra, dec)]

    def to_dict(self):
        return {"buffer_deg": self.buffer_deg, "regions": self.regions}

    @classmethod
    def from_file(cls, path):
        d = load_json(path)
        p = cls(d["buffer_deg"] * 60.0)
        p.regions = d["regions"]
        return p


def _area(box):
    r0, r1, d0, d1 = box
    return (r1 - r0) * np.cos(np.deg2rad(0.5 * (d0 + d1))) * (d1 - d0)


def _stripes(box, fractions):
    """Cut a box into consecutive stripes along its longer on-sky axis."""
    r0, r1, d0, d1 = box
    width_ra = (r1 - r0) * np.cos(np.deg2rad(0.5 * (d0 + d1)))
    along_ra = width_ra >= (d1 - d0)
    edges = np.concatenate([[0.0], np.cumsum(list(fractions.values()))])
    edges = edges / edges[-1]
    out = []
    for tag, a, b in zip(fractions.keys(), edges[:-1], edges[1:]):
        if along_ra:
            out.append((tag, (r0 + a * (r1 - r0), r0 + b * (r1 - r0), d0, d1)))
        else:
            out.append((tag, (r0, r1, d0 + a * (d1 - d0), d0 + b * (d1 - d0))))
    return out


def coverage_points(ms, stride=8):
    """Sky positions of pixels with data (WHT > 0) in EVERY file of a pointing, sampled
    every `stride` pixels. A drizzled ACS/WFC3 frame is a tilted quadrilateral of data
    inside a larger, zero-weight array, so the array's RA/Dec box overstates the data
    area - the corners of that box are empty. Only coverage is read here, never sources."""
    from ..fitsio import load_image, valid_mask
    ref = max(ms, key=lambda m: m["naxis1"] * m["naxis2"])
    sci, wht, w = load_image(ref["file"])
    yy, xx = np.mgrid[0:sci.shape[0]:stride, 0:sci.shape[1]:stride]
    ok = valid_mask(sci, wht)[yy, xx]
    ra, dec = w.pixel_to_world_values(xx[ok], yy[ok])
    keep = np.ones(len(ra), bool)
    for m in ms:
        if m is ref:
            continue
        s2, w2, wcs2 = load_image(m["file"])
        x2, y2 = wcs2.world_to_pixel_values(ra, dec)
        xi, yi = np.round(x2).astype(int), np.round(y2).astype(int)
        inb = (xi >= 0) & (yi >= 0) & (xi < s2.shape[1]) & (yi < s2.shape[0])
        v = np.zeros(len(ra), bool)
        v[inb] = valid_mask(s2, w2)[yi[inb], xi[inb]]
        keep &= v
    if keep.sum() < 0.05 * len(keep):            # bands hardly overlap: fall back to the largest file
        print(f"[Stage 0] WARNING: the bands of {ref['name']} overlap on only {keep.mean():.1%} of its area; "
              "stripes are cut from that file's coverage alone.")
        keep[:] = True
    return np.asarray(ra)[keep], np.asarray(dec)[keep]


def _stripes_equal_area(ra, dec, fractions):
    """Cut along the longer axis of the DATA area at the quantiles of the covered pixels,
    so each stripe holds its share (70/15/15) of the real data area, not of the box."""
    cosd = np.cos(np.deg2rad(np.median(dec)))
    along_ra = np.ptp(ra) * cosd >= np.ptp(dec)
    t = ra if along_ra else dec
    cum = np.cumsum(list(fractions.values()))
    cuts = np.concatenate([[t.min()], np.quantile(t, cum[:-1] / cum[-1]), [t.max()]])
    r0, r1, d0, d1 = ra.min(), ra.max(), dec.min(), dec.max()
    out = []
    for tag, a, b in zip(fractions.keys(), cuts[:-1], cuts[1:]):
        out.append((tag, (a, b, d0, d1) if along_ra else (r0, r1, a, b)))
    return out


def build_auto(metas, groups, cfg_part):
    fr = cfg_part["fractions"]
    part = SkyRegionPartitioner(cfg_part["collar_arcmin"])
    gzh, ext = [], []
    for gi, g in enumerate(groups):
        ms = [metas[i] for i in g]
        (ext if any(is_external(m, cfg_part.get("external_patterns")) for m in ms) else gzh).append((gi, bbox_of(ms)))
    for gi, box in ext:
        part.define_region(f"P{gi + 1:02d}_external", *box, split="external")
    if len(gzh) >= cfg_part.get("min_pointings_for_whole_assignment", 3):
        totals = {k: 0.0 for k in fr}
        total_area = sum(_area(b) for _, b in gzh)
        for gi, box in sorted(gzh, key=lambda t: -_area(t[1])):          # biggest first
            deficit = {k: fr[k] - totals[k] / total_area for k in fr}
            empty = [k for k in fr if totals[k] == 0]
            tag = empty[0] if empty and len(gzh) >= len(fr) else max(deficit, key=deficit.get)
            totals[tag] += _area(box)
            part.define_region(f"P{gi + 1:02d}", *box, split=tag)
        mode = "whole pointings"
    else:
        narrowest = np.inf
        cover = []
        for gi, box in gzh:
            try:
                ra, dec = coverage_points([metas[i] for i in groups[gi]])
                cover.append((ra, dec))
                stripes = _stripes_equal_area(ra, dec, fr)
            except Exception as e:                   # unreadable pixels: fall back to the header box
                print(f"[Stage 0] WARNING: could not read the coverage of pointing {gi + 1} ({e!r}); "
                      "cutting stripes from the header footprint.")
                stripes = _stripes(box, fr)
            for tag, sbox in stripes:
                part.define_region(f"P{gi + 1:02d}_{tag}", *sbox, split=tag)
                r0, r1, d0, d1 = sbox
                w = min((r1 - r0) * np.cos(np.deg2rad(0.5 * (d0 + d1))), d1 - d0) * 60   # arcmin
                narrowest = min(narrowest, w)
        mode = "stripes inside each pointing"
        if part.buffer_deg * 60 > 0.25 * narrowest:
            new = 0.25 * narrowest
            print(f"[Stage 0] WARNING: collar {part.buffer_deg * 60:.3f}' would swallow the narrowest stripe "
                  f"({narrowest:.3f}' wide); using {new:.3f}' instead.")
            part.buffer_deg = new / 60.0
        if cover:                                    # what each split REALLY got of the data area
            tags = pd.Series([t for ra, dec in cover for t in part.assign_many(ra, dec)]).fillna("collar")
            return part, mode, tags.value_counts(normalize=True).round(4).to_dict()
    return part, mode, None


def run(ctx: Ctx):
    cfg = ctx.cfg["partition"]
    out = ctx.stage_dir("0")
    if cfg["mode"] == "manual":
        part = SkyRegionPartitioner(cfg["collar_arcmin"])
        for r in cfg["manual_regions"]:
            part.define_region(r["name"], r["ra"][0], r["ra"][1], r["dec"][0], r["dec"][1], r["split"])
        mode, metas, groups, share = "manual", [], [], None
    else:
        files = find_products(ctx.raw, ctx.cfg["acquisition"]["file_patterns"])
        if not files:
            raise FileNotFoundError(f"No *_drc.fits / *_drz.fits files in {ctx.raw}. Put your HST files there "
                                    "(python tools/download_mast.py --ra <RA> --dec <Dec> --filters ...).")
        metas = [read_meta(f) for f in files]
        metas = [m for m in metas if m["wcs_ok"]]
        acq = ctx.cfg["acquisition"]
        groups = group_pointings(metas, acq["group_radius_arcsec"], acq.get("group_min_overlap", 0.5))
        for gi, g in enumerate(groups):
            ctx.log.info(f"[Stage 0] pointing P{gi + 1:02d}: {[metas[i]['filter'] for i in g]} "
                         f"({', '.join(metas[i]['name'] for i in g)})")
        part, mode, share = build_auto(metas, groups, cfg)

    save_json({**part.to_dict(), "mode": mode, "fractions": cfg["fractions"], "data_area_share": share},
              out / "split_rule.json")
    _plot(part, metas, out / "split_regions.png")
    tags = [r["split_tag"] for r in part.regions.values()]
    ctx.log.info(f"[Stage 0] {len(part.regions)} regions ({mode}); collar {part.buffer_deg * 60:.3f}' ; "
                 f"tags: { {t: tags.count(t) for t in set(tags)} }")
    if share:
        ctx.log.info(f"[Stage 0] share of the area WITH DATA per split (collar = no split): {share}")
    ctx.log.info(f"[Stage 0] Ruleset frozen -> {ctx.rel(out / 'split_rule.json')}")
    return {"mode": mode, "n_regions": len(part.regions)}


def _plot(part, metas, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Rectangle
    except Exception:
        return
    colours = {"train": "#2E5496", "dev": "#E69F00", "test": "#009E73", "external": "#CC79A7"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in metas:
        ax.add_patch(Polygon(np.asarray(m["footprint"]), fill=False, ec="0.6", lw=0.6))
    for name, r in part.regions.items():
        (r0, r1), (d0, d1) = r["ra_range"], r["dec_range"]
        ax.add_patch(Rectangle((r0, d0), r1 - r0, d1 - d0, alpha=0.35, color=colours.get(r["split_tag"], "grey")))
        ax.text((r0 + r1) / 2, (d0 + d1) / 2, r["split_tag"], ha="center", va="center", fontsize=7)
    ax.autoscale()
    ax.invert_xaxis()
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title("Stage 0: sky regions (grey = file footprints)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
