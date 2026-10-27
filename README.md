# From Pixels to Provenance

**Automated galaxy morphology classification in HST archival imaging** — NUS-ISS PRS Practice Module
(Sangeeta Bahrani, A0339780R).

The pipeline takes Level-3 HST drizzled images (`_drc.fits`) and produces one calibrated,
provenance-tracked catalogue (`catalog.csv`): every detected source is separated into
Star / Galaxy / Noise without labels, galaxies get a 5-class morphology and a merger
probability from a supervised ensemble, and the result is evaluated in-domain (Galaxy Zoo: Hubble)
and across catalogues (CANDELS).

Each stage of the proposal is one Python file in `src/p2p/stages/`, runs on its own, and writes
its results to its own folder under `outputs/`, so you can stop after any stage and look at what it did.

---

## 1. Set up (once)

```bash
# in VS Code: File > Open Folder... > pixels2provenance, then Terminal > New Terminal
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
# if statmorph fails to build:
pip install --no-build-isolation statmorph
```
In VS Code pick the `.venv` interpreter (Ctrl/Cmd+Shift+P → *Python: Select Interpreter*).

## 2. Run on real data

1. **Images** → `data/raw/`. For each pointing, the same field in 2–3 filters:
   ```bash
python tools/get_fits.py --out <folder> --target <RA>,<Dec>,<name> [--instrument <instrument>] [--filters <f1> <f2> <f3>] [--gaia] [--list] [--allow-single]
   ```
   or download by hand from the [MAST Portal](https://mast.stsci.edu) (HST, calibration level 3, DRC).
   Check a file with `python tools/inspect_fits.py data/raw/<file>_drc.fits`.
   Choose a field that **Galaxy Zoo: Hubble covers** (AEGIS, COSMOS, GEMS, GOODS-N, GOODS-S).
   COSMOS has HST imaging in F814W only — the pipeline then runs with one band.


2. **Catalogues** → `data/catalogs/` (see `data/catalogs/README.md`):
   * GZH Table 4: `gz_hubble_main.csv.gz` from
     https://zooniverse-data.s3.amazonaws.com/galaxy-zoo-hubble/gz_hubble_main.csv.gz (required)
   * Gaia DR3 (optional, for star validation) — run in the Gaia archive (https://gea.esac.esa.int/archive/),
     save as `data/catalogs/gaia_dr3.csv`:
     ```sql
     SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, teff_gspphot
     FROM gaiadr3.gaia_source
     WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 53.12, -27.80, 0.06))
     ```
   * CANDELS / Kartaltepe et al. (2015) FRACTIONAL table, VizieR `J/ApJS/221/11/table3`, saved as
     `data/catalogs/candels_k15.csv` (Phase 7): `python tools/download_catalogs.py candels`.
   Then `python tools/check_catalog_columns.py` to confirm the column mapping.
3. **Edit `config.yaml`** if needed: `partition.external_patterns` (which pointings are a separate CANDELS
   field; with one field leave it empty - Phase 7 then uses the held-out test split), `partition.collar_arcmin`,
   `acquisition.bands`.
4. **Run**: `python run_pipeline.py` (or the *P2P: run full pipeline* entry in VS Code's Run and Debug panel).

## 3. Running single stages

```bash
python run_pipeline.py --list            # all stages, their files and output folders
python run_pipeline.py --only 1c         # one stage (its inputs must already exist)
python run_pipeline.py --from 2a         # from 2a to the end
python run_pipeline.py --from 1a --to 1e
```
`python tools/show_stage.py 2d` prints a stage's purpose and method, the files it wrote, and the
numbers in its JSON outputs - the quickest way to understand a stage after running it.
Everything printed is also written to `outputs/pipeline.log`; `outputs/run_log.json` records the
status, run time and summary of every stage. VS Code: **Run and Debug** → *P2P: run ONE stage* lets
you pick a stage from a list and set breakpoints in its file.

## 4. Stages, files and outputs

Execution order puts **Phase 5 before Phase 4**: the supervised models train on the PSF-refined
Pass-2 features they are later deployed on.

| Stage | File (`src/p2p/stages/`) | Main outputs (`outputs/<folder>/`) |
|---|---|---|
| 0 Sky-region partitioning | `stage0_partition.py` | `split_rule.json`, `split_regions.png` |
| 1a Acquisition / scan | `stage1a_acquisition.py` | `manifest.json`, `pointings.json` |
| 1b Frame QC (CAE) | `stage1b_frame_qc.py` | `tile_qc.csv`, `*_qc.png`, `qc_threshold.json` |
| 1c Detection + aligned cutouts | `stage1c_detection.py` | `sources.csv`, `cutouts_raw.npy`, `segstamps.npy`, `P*_cutouts.png` |
| 1d PSF homogenisation | `stage1d_psf_match.py` | `cutouts.npy` (used from here on), `psf_match.json` |
| 1e Pass-1 morphology | `stage1e_pass1_morphology.py` | `features_pass1.csv`, `pass1_morphology.png` |
| 2a Representation + fusion | `stage2a_representation.py` | `hybrid_features.npy`, `rep_cae.pt` |
| 2b Clustering | `stage2b_clustering.py` | `clusters.csv`, `bic.png` |
| 2c Baseline + Gaia match | `stage2c_baseline.py` | `baseline.csv`, `gaia_match.csv` |
| 2d Semantic audit | `stage2d_semantic_audit.py` | `semantic_audit.json`, `montage_cluster_*.png` |
| 2e Continuum gating | `stage2e_continuum_gating.py` | `segregation.csv`, `anomaly_log.csv` |
| 3a GZH labels | `stage3a_gzh_labels.py` | `labels.csv`, `label_summary.json` |
| 3b Star routing | `stage3b_star_routing.py` | `star_routing.csv` |
| 3c Noise model | `stage3c_noise_model.py` | `noise_model.json`, `fpr_curve.png` |
| 5a Empirical PSF | `stage5a_empirical_psf.py` | `epsf_<pointing>.fits`, `epsf.png` |
| 5b Pass-2 refinement | `stage5b_pass2_refinement.py` | `features_pass2.csv`, `morph_delta_pass.csv` |
| 4a Dual-branch network | `stage4a_dual_branch.py` | `dual_branch.pt`, `dev_metrics.json` |
| 4b Ensemble | `stage4b_ensemble.py` | `efficientnet.pt`, `tabular_member.joblib` |
| 4c Calibration | `stage4c_calibration.py` | `temperature.json`, `reliability_dev.png` |
| 4d Ablation | `stage4d_ablation.py` | `ablation_table.csv`, `ablation.png` |
| 6a In-domain test | `stage6a_test_eval.py` | `test_metrics.json`, `confusion_matrix_test.png` |
| 6b Uncertainty audit | `stage6b_uncertainty.py` | `uncertainty_audit.json` |
| 6c Grad-CAM + SHAP | `stage6c_xai.py` | `gradcam_examples.png`, `shap_summary.csv` |
| 7a Taxonomy ceiling | `stage7a_taxonomy.py` | `taxonomy_alignment_matrix.csv`, `taxonomy_ceiling.json` |
| 7b Cross-catalogue | `stage7b_crosscat.py` | `cross_catalogue_metrics.json` |
| 7c Generalisation | `stage7c_generalisation.py` | `generalisation_report.json` |
| 8 Catalogue | `stage8_catalogue.py` | **`catalog.csv`**, `catalog_columns.csv`, `delivery_checks.json` |

Shared code: `context.py` (config, folders, stage list), `fitsio.py` (reading `_drc` files),
`detection.py`, `morphology.py` (statmorph), `catalogs.py` (GZH/Gaia/CANDELS loaders),
`nets.py` (all neural networks), `features.py`, `inference.py`, `utils.py`, `compat.py`.
Every stage file starts with a docstring: *Purpose / Method / Output*.

## 5. How objects are linked across stages

`sources.csv` (Stage 1c) gives every detection an `object_id` and a `cutout_index` (its row in
`cutouts.npy`). Every later table is keyed by `object_id`; `p2p.context.load_objects()` merges them.
`split_tag` (train / dev / test / external / empty = collar) is fixed in Stage 1c from the Stage-0
rules and never changes.

## 6. What to expect with only a few files

With one pointing, Stage 0 cuts the pointing into 70/15/15 stripes. One ACS pointing holds roughly
100–200 GZH-labelled galaxies, so the test split has only a few dozen: Stage 6a reports 95 %
bootstrap intervals — quote them. Train and test then share depth and PSF, so the in-domain test is
easier than a separate field. With one field there is no separate CANDELS pointing: `partition.phase7_region: auto` then evaluates
Phase 7 on the held-out **test** split (never trained on, so leakage-free; same depth and PSF as training,
so the gap measures the change of label scheme, not a new field). Without the Kartaltepe table
Stages 7a–7c are reported as *skipped* and the catalogue is still produced.

**Check the unsupervised stage by eye.** The clustering settings (`representation.n_pca`,
`spatial_weight`, `clustering.covariance_type`, `k_max`) were chosen on synthetic test images, where
the true star / galaxy / cosmic-ray labels are known. On real data open
`outputs/stage2d_semantic_audit/montage_cluster_*.png` and `semantic_audit.json`: each montage should
look like the name it was given. If a "Galaxy" cluster is full of point sources, adjust and rerun from 2a.

## 7. Troubleshooting

| Message | Fix |
|---|---|
| `... not found - run stage X first` | run the earlier stage, or use `--from` |
| `No labelled galaxies in split(s) [...]` | the field is not covered by GZH, or the collar removed a stripe: lower `partition.collar_arcmin` |
| `catalogs: required column(s) not found` | fix `catalogs.<kind>.columns` (see `tools/check_catalog_columns.py`) |
| statmorph build error | `pip install --no-build-isolation statmorph` |
| very slow Stage 1e / 5b | statmorph is ~0.5 s per source per core; set `project.n_jobs: -1` and/or lower `detection.max_sources_per_pointing` |
| out of memory in 1c | real frames are ~4200² px x 3 bands; close other programs or process fewer pointings at a time |
| HDBSCAN found no clusters | set `clustering.hdbscan_min_cluster_size` to a small number (e.g. 10) |

Runtime guide (4-core laptop, 1 real pointing x 3 bands, ~2000 detections): Stages 1b–1c ~5 min,
1e ~5 min, 2a ~3 min, 5b ~3 min, 4a–4d ~15 min on CPU (much less on a GPU).
