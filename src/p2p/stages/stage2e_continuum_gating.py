"""Stage 2e - Continuum handling and outlier gating.

Purpose : do not throw away real galaxies that sit between populations, and do
          not silently drop true anomalies either.
Method  : GMM components are summed into Star / Galaxy / Noise posteriors using
          the Stage-2d names. An object in a LOW-DENSITY region (HDBSCAN noise -1, or a
          GLOSH outlier score above the train percentile - Stage 2b density_outlier) but whose
          galaxy posterior is >= 0.80 is KEPT as a galaxy and tagged
          is_transitional. A low-density object with no posterior above 0.50
          AND a CAE reconstruction error above the 99th percentile of the train
          split is an extreme outlier -> anomaly_log.csv (anomaly_flag = 1).
          Physical consistency gate (gating.physical_gate): a GMM cluster is
          named as a whole, but one cluster can mix stars with compact galaxies.
          Each object is then checked against evidence the cluster name must
          not contradict:
            cosmic-ray candidate (no second band, Stage 1c)  -> Noise
            Gaia star                                         -> Star
            in a Star cluster without star evidence (Gaia, or the bright stellar
              locus of Stage 2d): sharper than the PSF -> Noise, otherwise -> Galaxy
              (faint point-like sources in a deep field are mostly compact galaxies)
            in a Galaxy cluster but on the stellar locus (Stage 2d)                    -> Star
            in a Noise cluster but confirmed in two bands, SNR >= 5, not sharper than
              the PSF (a real source): on the stellar locus -> Star, otherwise -> Galaxy
          Artefact rules (30 Oct, from the Stage 9 atlas: a saturated star, pieces of its diffraction spike and
          faint noise blobs had become "Galaxy"):
            bright-star mask (gating.bright_star_mask): Gaia stars brighter than bright_star_gmax that really
              move (|parallax| > 1 mas or proper motion > 5 mas/yr - bright compact GALAXIES in Gaia do not):
              the source nearest the star within bright_star_core_arcsec -> Star (a saturated star's bleed
              column can pull our centroid ~3" off, so the 0.5" Gaia match misses it); every other source
              within a + b x (gmax - G) arcsec (spikes, halo, bleed) -> Noise, unless it is a Gaia star itself
            fragment rule (gating.fragment_rule): a Galaxy inside the half-light radius of a brighter Galaxy is
              a deblending piece of it -> fragment_of = parent id; it stays a Galaxy row of the catalogue but
              leaves the galaxy track (no label, no training, no prediction, no atlas image)
            unresolved rule (gating.unresolved_rule): a Galaxy sharper than unresolved_fwhm_ratio x PSF that is
              missing a band or that the Stage 2c baseline calls Noise -> Noise
          Cosmic-ray gate (gating.cr_gate, 31 Oct, from the Stage 9 atlas): a Galaxy that loses more than
            cr_gate_fraction of its central light to the Stage 9 cleaner (single-band, sharp pixels) -> Noise.
            In single exposures two different cosmic rays in two bands can land on one segment and pass the
            two-band confirmation of Stage 1c, which is how they reached "Noise cluster, but a real source".
          The reason is kept in classification_override (empty = cluster name
          kept); cluster_semantic_2d keeps the pure cluster name, which is what
          Stage 2d scores for Objective 2.
          If HDBSCAN failed (Stage 2b), the transitional rule cannot be applied:
          is_transitional is left EMPTY (unknown, not False), and the outlier
          rule runs without the HDBSCAN condition.
Output  : segregation.csv (final cluster_semantic for every object), anomaly_log.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..catalogs import load_gaia
from ..context import Ctx, load_json, load_objects
from ..utils import as_bool

NAMES = ("Star", "Galaxy", "Noise")


def physical_gate(df, sem, override, cfg_b):
    """Object-level checks a cluster name must not contradict (see the module docstring)."""
    tol = cfg_b["point_source_tolerance"]
    ratio = df.fwhm_over_psf.to_numpy(float)
    cr = as_bool(df.cr_candidate).to_numpy() if "cr_candidate" in df else np.zeros(len(df), bool)
    gaia = as_bool(df.gaia_hit).to_numpy() if "gaia_hit" in df else np.zeros(len(df), bool)
    locus = as_bool(df.stellar_locus).to_numpy() if "stellar_locus" in df else np.zeros(len(df), bool)
    two_band = df.cr_candidate.map(lambda v: str(v).strip().lower() in ("false", "0", "0.0")).to_numpy() \
        if "cr_candidate" in df else np.ones(len(df), bool)
    resolved = np.nan_to_num(ratio, nan=0) > 1 + 2 * tol
    real_noise_cluster = two_band & (df.snr.to_numpy(float) >= 5) & (np.nan_to_num(ratio, nan=0) >= cfg_b["sub_psf_factor"])
    sem, override = sem.copy(), override.copy()

    def move(mask, to, why):
        mask = mask & (sem != to)
        sem[mask] = to
        override[mask] = why

    move((sem == "Noise") & real_noise_cluster & locus, "Star", "Noise cluster, but confirmed in 2 bands on the stellar locus")
    move((sem == "Noise") & real_noise_cluster & ~locus, "Galaxy",
         "Noise cluster, but a real source (confirmed in 2 bands, SNR >= 5, not sharper than the PSF)")
    move((sem == "Galaxy") & locus & ~resolved, "Star", "Galaxy cluster, but on the stellar locus")
    broader = np.nan_to_num(ratio, nan=0) > 1 + tol
    sharper = np.nan_to_num(ratio, nan=1) < cfg_b["sub_psf_factor"]
    no_star_evidence = (sem == "Star") & ~locus & ~gaia
    move(no_star_evidence & broader, "Galaxy", "Star cluster, but broader than a point source")
    move(no_star_evidence & sharper, "Noise", "Star cluster, but sharper than the PSF")
    move(no_star_evidence & ~broader & ~sharper, "Galaxy",
         "Star cluster, but too faint for size to separate a star from a compact galaxy")
    move(gaia, "Star", "Gaia star")
    move(cr & ~gaia, "Noise", "cosmic-ray candidate (no second band confirms it)")
    return sem, override


def artefact_rules(ctx, df, sem, override, cfg):
    """Bright-star mask + unresolved rule (see the module docstring). Returns sem, override and counts."""
    sem, override = sem.copy(), override.copy()
    counts = {}
    gaia_hit = as_bool(df.gaia_hit).to_numpy() if "gaia_hit" in df else np.zeros(len(df), bool)
    if cfg.get("bright_star_mask", True):
        metas = list(load_json(ctx.require("1a", "manifest.json")).values())
        gaia = load_gaia(ctx, metas)
        if gaia is None or gaia.empty:
            ctx.log.info("[Stage 2e] bright-star mask: no Gaia catalogue - skipped")
        else:
            gmax = float(cfg.get("bright_star_gmax", 17.0))
            a, b = (float(v) for v in cfg.get("bright_star_radius", [2.0, 4.5]))
            core_r = float(cfg.get("bright_star_core_arcsec", 3.0))
            pm = np.hypot(gaia.pmra.fillna(0).to_numpy(float), gaia.pmdec.fillna(0).to_numpy(float))
            moving = (np.abs(gaia.parallax.fillna(0).to_numpy(float)) > 1.0) | (pm > 5.0)
            stars = gaia[(gaia.g_mag.to_numpy(float) < gmax) & moving]
            ra, dec = df.ra.to_numpy(float), df.dec.to_numpy(float)
            core, zone = np.zeros(len(df), bool), np.zeros(len(df), bool)
            for gr, gd, g in stars[["ra", "dec", "g_mag"]].to_numpy(float):
                sep = np.hypot((ra - gr) * np.cos(np.radians(gd)), dec - gd) * 3600
                zone |= sep < a + b * (gmax - g)
                i = int(np.nanargmin(sep))
                if sep[i] < core_r and not gaia_hit[i]:
                    core[i] = True
            m = core & (sem != "Star")
            sem[m], override[m] = "Star", "saturated bright Gaia star (centroid off the 0.5\" Gaia match)"
            m = zone & ~core & ~gaia_hit & (sem != "Noise")
            sem[m], override[m] = "Noise", "bright-star artefact (spike / halo / bleed of a Gaia star)"
            counts.update(bright_stars=len(stars), saturated_star=int(core.sum()), bright_star_artefact=int(m.sum()))
    if cfg.get("unresolved_rule", True) and "fwhm_over_psf" in df:
        band_cols = [c for c in df.columns if c.startswith("mag_") and c[4:5].isalpha() and c[5:6].isdigit()]
        n_bands = df[band_cols].notna().sum(1).to_numpy() if band_cols else np.full(len(df), 99)
        base_noise = (df.baseline_semantic.astype(str) == "Noise").to_numpy() if "baseline_semantic" in df \
            else np.zeros(len(df), bool)
        ratio = df.fwhm_over_psf.to_numpy(float)
        m = (sem == "Galaxy") & (np.nan_to_num(ratio, nan=9) < float(cfg.get("unresolved_fwhm_ratio", 0.8))) \
            & ((n_bands < len(band_cols)) | base_noise)
        sem[m], override[m] = "Noise", "unresolved (sharper than the PSF) and a band missing or baseline Noise"
        counts["unresolved_to_noise"] = int(m.sum())
    ctx.log.info(f"[Stage 2e] artefact rules: {counts}")
    return sem, override


def cosmic_ray_gate(ctx, df, sem, override, cfg):
    """Galaxy whose central light is mostly single-band, sharp pixels -> Noise (see the module docstring).
    Uses the Stage 9 cleaner on the Stage 1c cutouts. Returns sem, override and the removed fraction per object."""
    from .stage9_galaxy_atlas import _wavelength, b_ok, clean_cosmic_rays, removed_fraction
    frac = np.full(len(df), np.nan)
    if not cfg.get("cr_gate", True):
        return sem, override, frac
    sem, override = sem.copy(), override.copy()
    atlas = ctx.cfg.get("atlas", {})
    info = load_json(ctx.require("1a", "pointings.json"))
    bands = info["bands"]
    order = sorted(range(len(bands)), key=lambda i: _wavelength(bands[i]))
    raw = np.load(ctx.require("1c", "cutouts_raw.npy"), mmap_mode="r")
    gaia = as_bool(df.gaia_hit).to_numpy() if "gaia_hit" in df else np.zeros(len(df), bool)
    idx, pts = df.cutout_index.to_numpy(int), df.pointing.to_numpy()
    for i in np.where((sem == "Galaxy") & ~gaia)[0]:
        have = [b for b in order if b_ok(bands[b], info["pointings"].get(pts[i], {}))]
        stamp = np.asarray(raw[idx[i]], np.float32)
        frac[i] = removed_fraction(stamp, clean_cosmic_rays(stamp, have or order, atlas))
    limit = float(cfg.get("cr_gate_fraction", 0.7))
    m = (sem == "Galaxy") & (np.nan_to_num(frac, nan=0.0) > limit)
    sem[m], override[m] = "Noise", f"cosmic rays (> {limit:g} of the central light is single-band, sharp pixels)"
    ctx.log.info(f"[Stage 2e] cosmic-ray gate: {int(m.sum())} of {int(np.isfinite(frac).sum())} galaxies -> Noise "
                 f"(cr_gate_fraction {limit:g})")
    return sem, override, frac


def fragments(ctx, df, sem):
    """Deblending fragments (31 Oct): a Galaxy lying inside the half-light radius of a BRIGHTER Galaxy is a
    piece of it (spiral arm, clump, spike) -> fragment_of = the brighter object's id. Fragments stay in the
    catalogue as Galaxy rows but leave the galaxy track (labels, training, predictions, atlas)."""
    from scipy.spatial import cKDTree
    frag = np.full(len(df), np.nan)
    if not ctx.cfg["gating"].get("fragment_rule", True):
        return frag
    pts = load_json(ctx.require("1a", "pointings.json"))["pointings"]
    def _scale(p):                                    # detection-band pixel scale [arcsec/px]
        v = pts.get(p, {}).get("pixscale", 0.05)
        return float(v.get(pts[p].get("detection_band"), next(iter(v.values()))) if isinstance(v, dict) else v or 0.05)
    scale = df.pointing.map(_scale).to_numpy(float)
    g = np.where(sem == "Galaxy")[0]
    if len(g) < 2:
        return frag
    dec0 = np.radians(np.nanmedian(df.dec.to_numpy(float)[g]))
    xy = np.c_[df.ra.to_numpy(float)[g] * np.cos(dec0) * 3600, df.dec.to_numpy(float)[g] * 3600]
    rh = np.nan_to_num(df.r_half.to_numpy(float)[g], nan=0) * scale[g]          # arcsec
    # a failed fit can report a huge radius and would swallow real neighbours: only a trusted radius makes a parent
    rh[rh > float(ctx.cfg["gating"].get("fragment_max_rhalf_arcsec", 2.0))] = 0
    if "flag" in df:
        rh[df.flag.to_numpy(float)[g] >= 2] = 0
    mag = np.nan_to_num(df.mag.to_numpy(float)[g], nan=99)
    oid = df.object_id.to_numpy()[g]
    for a, b in cKDTree(xy).query_pairs(max(2.0, float(np.nanmax(rh)) if len(rh) else 2.0)):
        d = float(np.hypot(*(xy[a] - xy[b])))
        par, ch = (a, b) if mag[a] <= mag[b] else (b, a)
        if d < rh[par] and (np.isnan(frag[g[ch]]) or mag[par] < mag[np.where(oid == frag[g[ch]])[0][0]]):
            frag[g[ch]] = oid[par]
    return frag


def run(ctx: Ctx):
    cfg = ctx.cfg["gating"]
    out = ctx.stage_dir("2e")
    df = load_objects(ctx, ["sources", "pass1", "clusters", "baseline", "gaia", "evidence"])
    names = {int(k): v for k, v in load_json(ctx.require("2d", "cluster_names.json")).items()}
    err = pd.Series(np.load(ctx.require("2a", "recon_error.npy")),
                    index=np.load(ctx.require("2a", "object_id.npy"))).reindex(df.object_id).to_numpy()

    post = {n: np.zeros(len(df)) for n in NAMES}
    for k, n in names.items():
        col = f"gmm_p{k}"
        if col in df and n in post:
            post[n] += df[col].to_numpy(float)
    hdb = df.cluster_id_hdbscan.to_numpy(float)
    hdb_ok = np.isfinite(hdb).any() and (hdb >= 0).any()
    if not hdb_ok:
        ctx.log.info("[Stage 2e] WARNING: HDBSCAN failed (Stage 2b) - is_transitional left empty; the outlier "
                     "rule uses GMM posterior + reconstruction error only.")
    if hdb_ok and "density_outlier" in df:
        # HDBSCAN noise OR a GLOSH outlier score above the train percentile (Stage 2b): in a deep field the
        # sources form one density continuum, so label -1 alone would mark almost nothing
        hnoise = as_bool(df.density_outlier).to_numpy()
    else:
        hnoise = (hdb == -1) if hdb_ok else np.ones(len(df), bool)
    ctx.log.info(f"[Stage 2e] low-density objects (HDBSCAN noise / GLOSH outliers): {int(hnoise.sum())}")
    P = np.column_stack([post[n] for n in NAMES])
    thr = float(np.nanpercentile(err[(df.split_tag == "train").to_numpy()], cfg["recon_percentile"]))
    transitional = hnoise & (post["Galaxy"] >= cfg["transitional_p"]) if hdb_ok else np.zeros(len(df), bool)
    anomaly = hnoise & (P.max(1) < cfg["outlier_max_p"]) & (np.nan_to_num(err, nan=-1) > thr)

    sem = df.cluster_semantic_2d.to_numpy(object).copy()
    sem[transitional] = "Galaxy"
    override = np.full(len(df), None, dtype=object)
    override[transitional] = "transitional (low density in HDBSCAN, GMM galaxy posterior >= transitional_p)"
    if cfg.get("physical_gate", True):
        sem, override = physical_gate(df, sem, override, ctx.cfg["baseline"])
        ctx.log.info(f"[Stage 2e] physical gate: {pd.Series(override).value_counts().to_dict()}")
    sem, override = artefact_rules(ctx, df, sem, override, cfg)
    sem, override, cr_frac = cosmic_ray_gate(ctx, df, sem, override, cfg)
    frag = fragments(ctx, df, sem)
    ctx.log.info(f"[Stage 2e] deblending fragments (inside a brighter galaxy's half-light radius): "
                 f"{int(np.isfinite(frag).sum())} -> fragment_of, left out of the galaxy track")
    res = pd.DataFrame({"object_id": df.object_id, "cluster_semantic": sem,
                        "gmm_p_star": post["Star"].round(4), "gmm_p_galaxy": post["Galaxy"].round(4),
                        "gmm_p_noise": post["Noise"].round(4), "reconstruction_error": err,
                        "cr_removed_frac": np.round(cr_frac, 3),
                        "fragment_of": pd.array(np.where(np.isfinite(frag), frag, np.nan), dtype="Int64"),
                        "is_transitional": transitional if hdb_ok else None, "anomaly_flag": anomaly.astype(int),
                        "classification_override": override})
    res.to_csv(out / "segregation.csv", index=False)
    log = res[res.anomaly_flag == 1].merge(df[["object_id", "ra", "dec", "pointing", "cluster_id_gmm",
                                               "cluster_id_hdbscan"]], on="object_id")
    log["reason"] = "low density (HDBSCAN noise / GLOSH) + no GMM population > 0.5 + reconstruction error above train 99th pct"
    log.to_csv(out / "anomaly_log.csv", index=False)
    ctx.log.info(f"[Stage 2e] populations {res.cluster_semantic.value_counts().to_dict()} | transitional kept "
                 f"{int(transitional.sum())} | anomalies logged {int(anomaly.sum())} (recon threshold {thr:.5f})")
    return {"populations": res.cluster_semantic.value_counts().to_dict(), "anomalies": int(anomaly.sum())}
