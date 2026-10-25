"""Stage 8 - Production compilation: the unified, provenance-tracked catalog.csv.

Purpose : one deliverable that a scientist can filter with a single column.
Method  : all tracks (Star, Galaxy, Noise; GZH and external pointings) are merged
          into one row per detected source. Every galaxy gets calibrated class
          probabilities and a merger probability from the frozen ensemble.
          Duplicates from overlapping pointings (closer than 0.2") are merged,
          keeping the higher-SNR detection. Each row gets an IAU-style
          designation (truncated coordinates, a letter appended when two would
          collide). `classification` (C18) is the final Star / Galaxy / Noise class
          after clustering and gating. Columns are written in the fixed order of
          COLUMNS (C1 object_id ... C75 baseline_semantic); the merger result is the
          probability p_merger only (no thresholded duplicate). Columns added
          after C75 (BW) follow EXTRA_COLUMNS: positional errors, per-band
          magnitudes, exposures, the cosmic-ray tests, gaia_match for every row
          and the CANDELS (Kartaltepe 2015) match for every galaxy. Every column has its own rule in COLUMNS
          (group, filled_for): detection measurements and the segregation evidence
          are kept for every row; magnitudes and colours for Star and Galaxy;
          morphology, labels and classification for Galaxy; Gaia and PSF
          calibration for Star; the noise model for Noise - null everywhere else.
          Before writing, the delivery checks of Objective 5 are asserted,
          including that rule and unique designations.
Output  : catalog.csv, catalog_columns.csv (data dictionary), delivery_checks.json
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..context import Ctx, load_json, load_objects, save_json
from ..features import TabularPrep, load_cutouts, make_tensors
from ..inference import calibrated_predict, load_models
from ..nets import MORPH_CLASSES
from ..catalogs import load_candels
from ..utils import as_bool, fwhm_fit, iau_designation, match_with_offset, predictive_entropy, unit_vectors

# One row per catalogue column: (name, group, filled_for, description).
# filled_for = the classifications whose rows MAY hold a value; every other row is null (checked before writing).
# A note in brackets narrows it further, e.g. "Star (Gaia-matched)": unmatched stars are null too.
ALL, SG = "all rows", "Star, Galaxy"
COLUMNS = [
    # Order = Sangeeta's column sequence (catalog_columns_Sangeeta.xlsx, 24 Oct 2026), pred_merger removed.
    # Stage 8 asserts that catalog.csv is written in exactly this order (C1 object_id ... C75 baseline_semantic).
    ("object_id", "identity", ALL, "unique source id (Stage 1c)"),  # C1
    ("designation", "identity", ALL, "IAU-style name from truncated RA/Dec; a letter is appended when two detections would share a name"),  # C2
    ("product_type", "provenance", ALL, "drc or drz"),  # C3
    ("ra", "identity", ALL, "right ascension [deg, ICRS]"),  # C4
    ("dec", "identity", ALL, "declination [deg, ICRS]"),  # C5
    ("pointing", "identity", ALL, "pointing id (Stage 1a)"),  # C6
    ("detection_band", "identity", ALL, "filter the source was detected in"),  # C7
    ("tile_id", "identity", ALL, "detection-band _drc file"),  # C8
    ("frame_quality_flag", "provenance", ALL, "1 = clean, 0 = in a QC-flagged tile (Stage 1b)"),  # C9
    ("edge_flag", "provenance", ALL, "cutout truncated by the image edge"),  # C10
    ("anomaly_flag", "provenance", ALL, "1 = fits no population (Stage 2e)"),  # C11
    ("cluster_id_gmm", "segregation", ALL, "GMM component (Stage 2b)"),  # C12
    ("cluster_id_hdbscan", "segregation", ALL, "HDBSCAN cluster, -1 = density noise"),  # C13
    ("gmm_p_star", "segregation", ALL, "GMM posterior of the Star population"),  # C14
    ("gmm_p_galaxy", "segregation", ALL, "GMM posterior of the Galaxy population"),  # C15
    ("gmm_p_noise", "segregation", ALL, "GMM posterior of the Noise population"),  # C16
    ("split_tag", "provenance", ALL, "train / dev / test / external; buffer = inside a collar (no split)"),  # C17
    ("classification", "identity", ALL, "final population after clustering + gating: Star / Galaxy / Noise"),  # C18
    ("morph_flag", "morphology", "Galaxy", "statmorph flag: 0 good, 1 suspect, 2 bad, 4 too faint to measure"),  # C19
    ("pred_class", "classification", "Galaxy (anomaly_flag 0)", "predicted morphology class"),  # C20
    ("p_smooth", "classification", "Galaxy (anomaly_flag 0)", "calibrated probability Smooth"),  # C21
    ("p_spiral", "classification", "Galaxy (anomaly_flag 0)", "calibrated probability FaceOnSpiral"),  # C22
    ("p_edgeon", "classification", "Galaxy (anomaly_flag 0)", "calibrated probability EdgeOn"),  # C23
    ("p_irregular", "classification", "Galaxy (anomaly_flag 0)", "calibrated probability Irregular"),  # C24
    ("p_ambiguous", "classification", "Galaxy (anomaly_flag 0)", "calibrated probability Ambiguous"),  # C25
    ("pred_entropy", "classification", "Galaxy (anomaly_flag 0)", "predictive entropy of the class probabilities"),  # C26
    ("p_merger", "classification", "Galaxy (anomaly_flag 0)", "calibrated merger probability (dual-branch model)"),  # C27
    ("star_fwhm_arcsec", "psf_calibration", "Star", "this star's FWHM, 2-D Gaussian fit, detection band [arcsec]"),  # C28
    ("used_in_psf_match", "psf_calibration", "Star", "star used for the per-band PSFs in cross-filter matching (Stage 1d)"),  # C29
    ("used_in_epsf", "psf_calibration", "Star", "star stacked into the empirical PSF (Stage 5a)"),  # C30
    ("epsf_isolated", "psf_calibration", "Star", "no neighbour near the star"),  # C31
    ("pointing_epsf_fwhm_arcsec", "psf_calibration", "Star", "FWHM of this pointing's empirical PSF [arcsec]"),  # C32
    ("pointing_matched_fwhm_arcsec", "psf_calibration", "Star", "MEASURED FWHM of the PSF all bands were matched to (Stage 1d) [arcsec]; empty = no stars, Gaussian fallback"),  # C33
    ("is_pseudo_label", "labels", "Galaxy", "False = a GZH label exists; True = model prediction only"),  # C34
    ("gz_label", "labels", "Galaxy (GZH-matched)", "Galaxy Zoo: Hubble class"),  # C35
    ("gz_vote_frac", "labels", "Galaxy (GZH-matched)", "GZH agreement along the decision path"),  # C36
    ("gz_merger_flag", "labels", "Galaxy (GZH-matched)", "GZH merger flag"),  # C37
    ("gaia_source_id", "gaia", "Star (Gaia-matched)", "Gaia DR3 source_id (exact integer)"),  # C38
    ("gaia_sep_arcsec", "gaia", "Star (Gaia-matched)", "separation to the Gaia position [arcsec]"),  # C39
    ("mag", "photometry", SG, "AB magnitude, detection band"),  # C40
    ("colour_1", "photometry", SG, "colour {c1} [mag]"),  # C41
    ("colour_2", "photometry", SG, "colour {c2} [mag]"),  # C42
    ("gini", "morphology", "Galaxy (measured)", "Gini coefficient"),  # C43
    ("m20", "morphology", "Galaxy (measured)", "M20 moment"),  # C44
    ("concentration", "morphology", "Galaxy (measured)", "concentration C"),  # C45
    ("asymmetry", "morphology", "Galaxy (measured)", "asymmetry A"),  # C46
    ("smoothness", "morphology", "Galaxy (measured)", "smoothness / clumpiness S"),  # C47
    ("is_transitional", "segregation", "Galaxy", "density-noise object kept as a galaxy (posterior >= 0.8)"),  # C48
    ("psf_pass", "morphology", "Galaxy", "2 = morphology measured with the empirical PSF (Stage 5b)"),  # C49
    ("morph_flag_sersic", "morphology", "Galaxy", "statmorph Sersic-fit flag (0 = good fit)"),  # C50
    ("sersic_n", "morphology", "Galaxy (measured)", "Sersic index"),  # C51
    ("sersic_rhalf", "morphology", "Galaxy (measured)", "Sersic half-light radius [px]"),  # C52
    ("r_half", "morphology", "Galaxy (measured)", "non-parametric half-light radius [px]"),  # C53
    ("ellipticity", "morphology", "Galaxy (measured)", "ellipticity"),  # C54
    ("shape_asymmetry", "morphology", "Galaxy (measured)", "shape asymmetry"),  # C55
    ("multimode", "morphology", "Galaxy (measured)", "multimode statistic"),  # C56
    ("intensity", "morphology", "Galaxy (measured)", "intensity statistic"),  # C57
    ("deviation", "morphology", "Galaxy (measured)", "deviation statistic"),  # C58
    ("parallax", "gaia", "Star (Gaia-matched)", "Gaia parallax [mas]"),  # C59
    ("pmra", "gaia", "Star (Gaia-matched)", "Gaia proper motion RA [mas/yr]"),  # C60
    ("pmdec", "gaia", "Star (Gaia-matched)", "Gaia proper motion Dec [mas/yr]"),  # C61
    ("gaia_g", "gaia", "Star (Gaia-matched)", "Gaia G magnitude"),  # C62
    ("gaia_teff", "gaia", "Star (Gaia-matched)", "Gaia effective temperature [K]"),  # C63
    ("noise_score", "noise", "Noise", "probability-like score that the detection is spurious (Stage 3c)"),  # C64
    ("fpr_at_threshold", "noise", "Noise", "false-positive rate of detections at this source's SNR (Stage 3c)"),  # C65
    ("qc_flagged_bands", "provenance", ALL, "bands whose QC tile was flagged"),  # C66
    ("pipeline_version", "provenance", ALL, "code version"),  # C67
    ("snr", "detection", ALL, "signal-to-noise ratio of the detection, detection band"),  # C68
    ("ndrizim", "provenance", ALL, "NDRIZIM header: input images drizzled into the detection file (ACS/WFC: 2 per exposure, one per chip) - NOT exposures, see n_exposures"),  # C69
    ("flux", "detection", ALL, "flux of the detection [e-/s], detection band"),  # C70
    ("flux_err", "detection", ALL, "flux error [e-/s]"),  # C71
    ("fwhm_over_psf", "segregation", ALL, "source FWHM / PSF FWHM, both 2-D Gaussian fits in the detection band (1 = point-like)"),  # C72
    ("band_consistency", "segregation", ALL, "weakest other-band flux / detection-band flux (~0 = single-band, e.g. a cosmic ray); empty if no other band covers it"),  # C73
    ("reconstruction_error", "segregation", ALL, "Stage 2a autoencoder reconstruction error"),  # C74
    ("baseline_semantic", "segregation", ALL, "rule-based baseline class (Stage 2c)"),  # C75
]
# Columns after C75 (BW), in this order. "{band}" entries are repeated for every band, blue -> red.
EXTRA_COLUMNS = [
    ("ra_err", "astrometry", ALL, "statistical centroid error, RA cos(dec): FWHM / (2.355 SNR) [arcsec] (absolute WCS error not included)"),
    ("dec_err", "astrometry", ALL, "statistical centroid error, Dec: FWHM / (2.355 SNR) [arcsec]"),
    ("mag_{band}", "photometry", SG, "AB magnitude in {band} (detection footprint, cosmic-ray pixels replaced)"),
    ("magerr_{band}", "photometry", SG, "error of mag_{band} [mag]"),
    ("n_exposures", "provenance", ALL, "exposures covering the source in the detection band (drizzle CTX map); 1 = no cosmic-ray rejection possible"),
    ("cr_candidate", "segregation", ALL, "True = no other covering band confirms the source at S/N >= detection.confirm_snr (cosmic-ray candidate); empty = no other band covers it"),
    ("cr_pixels_masked", "segregation", ALL, "cosmic-ray pixels replaced and masked inside the source footprint (Stage 1c)"),
    ("gaia_match", "gaia", ALL, "True = a Gaia DR3 source within semantic_audit.gaia_radius_arcsec, whatever the classification"),
    ("classification_override", "segregation", ALL, "why classification differs from the GMM cluster name (Stage 2e physical gate / transitional rule); empty = cluster name kept"),
    ("k15_class", "cross_catalogue", "Galaxy (CANDELS-matched)", "CANDELS (Kartaltepe et al. 2015) dominant expert class"),
    ("k15_merger_frac", "cross_catalogue", "Galaxy (CANDELS-matched)", "CANDELS expert merger fraction f(M)"),
    ("k15_sep_arcsec", "cross_catalogue", "Galaxy (CANDELS-matched)", "separation to the CANDELS position after the median offset is removed [arcsec]"),
]


def catalogue_columns(bands):
    """The full column list: C1-C75 (COLUMNS), then EXTRA_COLUMNS with {band} expanded."""
    extra = []
    for n, g, f, d in EXTRA_COLUMNS:
        if "{band}" in n:
            extra += [(n.format(band=b), g, f, d.format(band=b)) for b in bands]
        else:
            extra.append((n, g, f, d))
    return COLUMNS + extra

RENAME = {"flag": "morph_flag", "flag_sersic": "morph_flag_sersic"}          # internal name -> catalogue name
ORDER = [n for n, *_ in COLUMNS]                           # the fixed 75-column sequence, C1 ... C75
assert len(ORDER) == 75 and len(set(ORDER)) == 75, "COLUMNS must hold 75 distinct columns"
MORPH_COLS = ["gini", "m20", "concentration", "asymmetry", "smoothness", "sersic_n", "sersic_rhalf", "r_half",
              "shape_asymmetry", "multimode", "intensity", "deviation"]


def allowed(filled_for: str) -> set | None:
    """Classifications a column may be filled for (None = every row)."""
    base = filled_for.split("(")[0].strip()
    return None if base == ALL else {c.strip() for c in base.split(",")}


PCOLS = ["p_smooth", "p_spiral", "p_edgeon", "p_irregular", "p_ambiguous"]


def unique_designations(names: pd.Series, ra: pd.Series) -> pd.Series:
    """Truncated IAU names collide for detections closer than ~1.5": append A, B, C ... (ordered by RA)."""
    names = pd.Series(np.asarray(names, dtype=object), index=ra.index)
    dup = names.duplicated(keep=False)
    if dup.any():
        order = ra[dup].groupby(names[dup]).rank(method="first").astype(int) - 1
        letters = [chr(65 + k % 26) + ("" if k < 26 else str(k // 26)) for k in order]
        names.loc[order.index] = names.loc[order.index] + np.array(letters, dtype=object)
    return names


def deduplicate(df, radius_arcsec):
    pairs = cKDTree(unit_vectors(df.ra, df.dec)).query_pairs(2 * np.sin(np.deg2rad(radius_arcsec / 3600) / 2))
    snr = df.snr.to_numpy()
    drop = {j if snr[i] >= snr[j] else i for i, j in pairs}
    return df.drop(index=df.index[list(drop)]), len(drop)


def run(ctx: Ctx):
    cfg = ctx.cfg["catalogue"]
    out = ctx.stage_dir("8")
    parts = ["sources", "pass1", "clusters", "baseline", "gaia", "segregation", "labels", "routing", "noise", "pass2"]
    parts = [p for p in parts if p != "routing" or ctx.path("3b", "star_routing.csv").exists()]
    df = load_objects(ctx, parts)
    df["psf_pass"] = df.psf_pass.fillna(1).astype(int)

    # ---- calibrated predictions for every galaxy row ----
    gal = (df.cluster_semantic == "Galaxy") & (df.anomaly_flag == 0)
    models = load_models(ctx)
    prep = TabularPrep.load(ctx.require("4a", "tab_prep.joblib"))
    cut = load_cutouts(ctx)
    P, M = [], []
    gidx = df.index[gal]
    for i in range(0, len(gidx), 256):
        r = calibrated_predict(ctx, make_tensors(df.loc[gidx[i:i + 256]], cut, prep, with_labels=False), models)
        P.append(r["morph"])
        M.append(r["merger"])
    if len(gidx):
        P, M = np.concatenate(P), np.concatenate(M)
        for k, c in enumerate(PCOLS):
            df.loc[gal, c] = P[:, k]
        df.loc[gal, "pred_class"] = np.array(MORPH_CLASSES)[P.argmax(1)]
        df.loc[gal, "p_merger"] = M                  # probability only: a 0/1 pred_merger would repeat it
        df.loc[gal, "pred_entropy"] = predictive_entropy(P)

    # ---- identity + provenance ----
    pts = load_json(ctx.require("1a", "pointings.json"))["pointings"]
    man = load_json(ctx.require("1a", "manifest.json"))
    det_file = {p: Path(v["bands"][v["detection_band"]]).name for p, v in pts.items()}
    df["tile_id"] = df.pointing.map(det_file)
    df["product_type"] = df.tile_id.map(lambda f: man.get(f, {}).get("product_type"))
    df["ndrizim"] = df.tile_id.map(lambda f: man.get(f, {}).get("ndrizim"))
    from .. import __version__
    df["pipeline_version"] = f"p2p-{__version__}"
    # empty when HDBSCAN failed (Stage 2e could not apply the rule): keep it empty, do not turn it into False
    df["is_transitional"] = df.is_transitional.map(
        lambda v: None if pd.isna(v) or str(v).strip() == "" else str(v).strip().lower() in ("true", "1", "1.0"))
    df["classification"] = df["cluster_semantic"]    # final Star / Galaxy / Noise after clustering + gating

    # ---- PSF / optical calibration columns for the star track ----
    psfm = load_json(ctx.require("1d", "psf_match.json"))
    scale = df.pointing.map({p: v.get("pixscale_arcsec", np.nan) for p, v in psfm.items()}).astype(float)
    # each star's FWHM with the SAME 2-D Gaussian fit Stage 5a uses for the ePSF, in the detection band of the
    # un-matched cutout, so star_fwhm_arcsec and pointing_epsf_fwhm_arcsec can be compared directly
    from .stage1d_psf_match import crop_center
    info = load_json(ctx.require("1a", "pointings.json"))
    raw = np.load(ctx.require("1c", "cutouts_raw.npy"), mmap_mode="r")
    det_ch = {p: info["bands"].index(v["detection_band"]) for p, v in info["pointings"].items()}
    star = df.cluster_semantic == "Star"
    df["star_fwhm_arcsec"] = np.nan
    df.loc[star, "star_fwhm_arcsec"] = [
        fwhm_fit(crop_center(np.asarray(raw[int(i), det_ch[p]], np.float32), ctx.cfg["psf_match"]["stamp"]))
        for i, p in zip(df.loc[star, "cutout_index"], df.loc[star, "pointing"])]
    df["star_fwhm_arcsec"] = (df["star_fwhm_arcsec"] * scale).round(4)
    match_ids = {int(i) for v in psfm.values() for i in v.get("star_ids", [])}
    df["used_in_psf_match"] = df.object_id.isin(match_ids)
    # the MEASURED matched FWHM (Stage 1d), empty when the target band used the Gaussian fallback
    df["pointing_matched_fwhm_arcsec"] = df.pointing.map(
        {p: (v.get("matched_fwhm_arcsec") if v.get("target_psf_measured") else np.nan) for p, v in psfm.items()}
    ).astype(float).round(4)
    rep = ctx.path("5a", "epsf_report.json")
    if rep.exists():
        ep = {p: v.get("fwhm_px", np.nan) for p, v in load_json(rep).items()}
        df["pointing_epsf_fwhm_arcsec"] = (df.pointing.map(ep).astype(float) * scale).round(4)
    for c in ("used_in_epsf", "epsf_isolated"):
        if c in df:
            df[c] = as_bool(df[c])
    df["is_pseudo_label"] = np.where(df.cluster_semantic == "Galaxy", ~df.gz_label.notna(), None)
    df, n_dup = deduplicate(df, cfg["dedup_radius_arcsec"])
    df["designation"] = unique_designations(iau_designation(df.ra, df.dec, cfg["designation_prefix"]), df.ra)

    # exact Gaia DR3 ids (19 digits do not fit a float64; a merge with missing values would turn them into floats)
    gm = pd.read_csv(ctx.require("2c", "gaia_match.csv"), dtype={"gaia_source_id": "string"})
    ids = gm.set_index("object_id")["gaia_source_id"]
    df["gaia_source_id"] = df.object_id.map(ids).where(lambda v: v.notna() & (v != "-1"), None)
    df["gaia_match"] = as_bool(df.gaia_hit)                       # every row: misclassified Gaia stars stay visible
    df["cr_candidate"] = df.cr_candidate.map(
        lambda v: None if pd.isna(v) or str(v).strip() == "" else str(v).strip().lower() in ("true", "1", "1.0")) \
        if "cr_candidate" in df else None

    # CANDELS (Kartaltepe 2015) match for every galaxy (Stage 7a evaluates only the Phase-7 region)
    for c in ("k15_class", "k15_merger_frac", "k15_sep_arcsec"):
        df[c] = None
    k15 = load_candels(ctx, list(load_json(ctx.require("1a", "manifest.json")).values()))
    galaxy_rows = (df.cluster_semantic == "Galaxy").to_numpy()
    if k15 is not None and len(k15) and galaxy_rows.any():
        from .stage7a_taxonomy import K15
        g = df[galaxy_rows]
        kidx, ksep, khit, koff = match_with_offset(g.ra, g.dec, k15.ra, k15.dec, 0.5)
        F = k15[list(K15)].to_numpy(float)
        main = np.array(K15)[np.nanargmax(np.nan_to_num(F, nan=-1), axis=1)]
        j = np.clip(kidx, 0, None)
        df.loc[galaxy_rows, "k15_class"] = np.where(khit, main[j], None)
        df.loc[galaxy_rows, "k15_merger_frac"] = np.where(khit, k15.merger.to_numpy(float)[j], np.nan)
        df.loc[galaxy_rows, "k15_sep_arcsec"] = np.where(khit, np.round(ksep, 4), np.nan)
        ctx.log.info(f"[Stage 8] CANDELS: {int(khit.sum())} of {int(galaxy_rows.sum())} galaxies matched "
                     f"(offset removed dRA {koff[0]:+.3f}\", dDec {koff[1]:+.3f}\")")
    else:
        ctx.log.info("[Stage 8] no CANDELS table covering the field - k15_* columns empty")
    df = df.rename(columns=RENAME)
    for c in MORPH_COLS:                                          # statmorph -99 = could not measure -> empty
        if c in df:
            v = pd.to_numeric(df[c], errors="coerce")
            df[c] = v.where(~(v <= -98.9), np.nan)

    # a required column that an earlier stage did not produce is written empty, so the C1-C75 layout never shifts
    bands = load_json(ctx.require("1a", "pointings.json"))["bands"]
    cols = catalogue_columns(bands)
    missing = [n for n, *_ in cols if n not in df]
    if missing:
        ctx.log.info(f"[Stage 8] WARNING: no values from earlier stages for {missing}; written as empty columns")
        for n in missing:
            df[n] = None

    # ---- sparse schema, column by column ----
    for name, _, filled_for, _ in cols:
        if name not in df:
            continue
        ok = allowed(filled_for)
        df[name] = df[name].astype(object)
        if ok is not None:
            df.loc[~df.classification.isin(ok), name] = None
        if filled_for.startswith("Star (Gaia-matched)"):
            df.loc[~as_bool(df.gaia_hit), name] = None
        if "anomaly_flag 0" in filled_for:
            df.loc[df.anomaly_flag != 0, name] = None
    ordered = [n for n, *_ in cols]
    cat = df[ordered].sort_values("object_id")
    sparse_ok = all(cat.loc[~cat.classification.isin(allowed(f)), n].isna().all()
                    for n, _, f, _ in cols if allowed(f) is not None)

    # ---- delivery checks (Objective 5) ----
    g = cat[(cat.classification == "Galaxy") & (cat.anomaly_flag == 0)]
    psum = g[PCOLS].astype(float).sum(axis=1)
    checks = {"every_galaxy_has_calibrated_probs": bool(np.allclose(psum, 1.0, atol=1e-4)),
              "every_galaxy_has_merger_prob": bool(g.p_merger.notna().all()),
              "every_galaxy_has_heritage_flag": bool(g.is_pseudo_label.notna().all()),
              "every_measured_galaxy_at_pass_2": bool((g[g.snr >= ctx.cfg["morphology"]["min_snr"]].psf_pass == 2).all()),
              "class_columns_empty_for_other_classifications": bool(sparse_ok),
              "designations_unique": bool(cat.designation.is_unique),
              "column_order_as_specified": list(cat.columns) == ordered and list(cat.columns[:75]) == ORDER}
    pm = g.p_merger.astype(float)
    notes = {"columns": len(cat.columns), "first_extra_column": cat.columns[75] if len(cat.columns) > 75 else None,
             "p_merger_range": [round(float(pm.min()), 4), round(float(pm.max()), 4)] if len(pm) else None}
    if len(pm) and float(pm.max() - pm.min()) < 0.1:
        notes["merger_model"] = (f"p_merger spans only {pm.min():.3f}-{pm.max():.3f}: the merger head is effectively "
                                 "constant - report it as not trainable with this number of GZH mergers")
        ctx.log.info(f"[Stage 8] NOTE: {notes['merger_model']}")
    save_json({**checks, "duplicates_merged": n_dup, "notes": notes}, out / "delivery_checks.json")
    assert all(checks.values()), f"Delivery check failed: {checks}"
    cat.to_csv(out / "catalog.csv", index=False)
    fill = {"c1": f"{bands[0]} - {bands[1]}" if len(bands) > 1 else "(needs 2 bands)",
            "c2": f"{bands[1]} - {bands[2]}" if len(bands) > 2 else "(needs 3 bands)"}
    pd.DataFrame([{"column": n, "group": g_, "filled_for": f, "rows_filled": int(cat[n].notna().sum()),
                   "description": d.format(**fill)}
                  for n, g_, f, d in cols]).to_csv(out / "catalog_columns.csv", index=False)
    heritage = g.is_pseudo_label.astype(str).value_counts().to_dict()
    ctx.log.info(f"[Stage 8] catalog.csv: {len(cat)} rows x {cat.shape[1]} columns ({n_dup} duplicates merged) | "
                 f"{cat.classification.value_counts().to_dict()} | galaxy labels from GZH (False) / model (True): "
                 f"{heritage}")
    ctx.log.info(f"[Stage 8] delivery checks: {checks}")
    return {"rows": len(cat), "checks": checks}
