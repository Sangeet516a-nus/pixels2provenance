# From Pixels to Provenance

**Automated galaxy morphology classification in HST archival imaging** — NUS-ISS PRS Practice Module
(Sangeeta Bahrani, A0339780R).

The pipeline takes Level-3 HST drizzled images (`_drc.fits`) and produces one calibrated,
provenance-tracked catalogue (`catalog.csv`): every detected source is separated into
Star / Galaxy / Noise without labels, galaxies get a 5-class morphology and a merger
probability from a feature-level fusion (frozen ImageNet ResNet-18 + EfficientNet-B0 features + 14 physical
features -> XGBoost), and the result is evaluated in-domain (Galaxy Zoo: Hubble)
and across catalogues (CANDELS). A final stage draws a colour image of every catalogue galaxy.

Each stage of the proposal is one Python file in `src/p2p/stages/`, runs on its own, and writes
its results to its own folder under `outputs/`, so you can stop after any stage and look at what it did.

There are three ways to start the pipeline. All three run the same code and give the same catalogue:

| Way | Command | What it does |
|---|---|---|
| Command line | `python run_pipeline.py` | runs the stages on the files already in `data/raw/` |
| Web app (UI) | `streamlit run app.py` | type sky positions → search MAST → download → run → view results |
| Notebook | `Pixels_to_Provenance_Pipeline.ipynb` | the whole code as cells, for Colab or Jupyter, with an input panel |

The app and the download tools are an interface layer only: they find and download the images and then call
`run_pipeline.py`. They do not change any stage, model (CAE, clustering, classifiers) or result.

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

`requirements.txt` includes three packages that only the download tools and the UI need:
`astroquery` (MAST and Gaia queries), `streamlit` (the web app) and `ipywidgets` (the notebook panel).

## 2. Download HST images for a sky position (`tools/get_fits.py`)

`get_fits.py` **only downloads**; it does not run the pipeline. For each sky position it searches the MAST
archive and downloads one `_drc.fits` (or `_drz.fits`) file per filter into its own folder, so fields never mix.

```bash
python tools/get_fits.py --out <folder> --target <RA>,<Dec>,<name> [--instrument <instrument>] [--filters <f1> <f2> <f3>] [--gaia] [--list] [--allow-single] [--radius-arcmin <r>]
```

| Part | Meaning | Default if left out |
|---|---|---|
| `--out <folder>` | Parent folder for the downloads; each target gets its own subfolder `<folder>/<name>/` | `data/fields` |
| `--target <RA>,<Dec>,<name>` | Position in degrees (or `hh:mm:ss`, `dd:mm:ss`) plus a short field name, no spaces around the commas; repeat for more positions | required (the name is optional: without it the folder is called `ra<RA>_dec<Dec>`) |
| `--instrument <instrument>` | `ACS/WFC`, `WFC3/UVIS` or `WFC3/IR` | `ACS/WFC` |
| `--filters <f1> <f2> <f3>` | Choose the filters yourself | one blue, one green, one red, chosen automatically |
| `--gaia` | Also save the field's Gaia DR3 star catalogue | off |
| `--list` | Only show what would be downloaded; downloads nothing | off |
| `--allow-single` | Accept single-exposure files if no combined file exists (more cosmic rays) | off |
| `--radius-arcmin <r>` | Search radius around the position | `2.0` |

**Examples**

```bash
# 1. Check first (nothing is downloaded)
python tools/get_fits.py --list --out data/fields --target 189.23,62.24,GOODS-N

# 2. Download, with Gaia, into data/fields/GOODS-N/
python tools/get_fits.py --out data/fields --target 189.23,62.24,GOODS-N --gaia

# 3. Two positions, WFC3/IR, chosen filters
python tools/get_fits.py --out data/fields --target 189.23,62.24,GOODS-N --target 53.12,-27.80,GOODS-S --instrument WFC3/IR --filters F105W F125W F160W
```
Southern positions take a minus sign directly, as in `53.12,-27.80`.
`python tools/get_fits.py --help` prints the full list of options.

**How a file is chosen, per filter**

1. **Filters:** one per colour (blue, green, red), e.g. F435W / F606W / F814W on ACS/WFC, unless `--filters` is given.
2. **Coverage:** the image's footprint on the sky must contain the position; being close to it is not enough.
3. **Combined before single:** products made from several exposures (cosmic rays already rejected) come first.
   Single exposures are skipped unless `--allow-single` is given, because their cosmic rays are the largest
   source of false detections in the pipeline.
4. **Tie-breaks:** the longest total exposure time, then the image centre closest to the position.

Multi-visit skycell mosaics (several GB) are skipped. For every filter the script prints how many images it found,
how many cover the position and how many are combined, so a missing filter comes with a reason.

**To run the pipeline on a downloaded field:** move the old files out of `data/raw/`, copy that field's
`_drc.fits` files into `data/raw/`, then `python run_pipeline.py`. Or use the field runner below, which does
these steps itself.

## 3. Search, download and run in one go: the web app (UI) and the field runner

```bash
streamlit run app.py                                                    # the web app
python tools/run_field.py --target 53.12,-27.80,GOODS-S --search-only   # the same from the command line
python tools/run_field.py --target 53.12,-27.80,GOODS-S --combined-only
```

**The web app** opens in your browser and has a settings sidebar and four sections:

| Part of the page | What you do / see |
|---|---|
| Sidebar: Settings | instrument (ACS/WFC, WFC3/UVIS, WFC3/IR); filters (auto = one per colour, or choose); MAST search radius; *combined exposures only*; supervised mode (auto / train / apply); model bundle for apply mode; *keep the previous run in place*; stage to start at |
| 1. Sky positions | a table of name, RA, Dec (degrees or `hh:mm:ss` / `dd:mm:ss`); add rows for more positions; **Search the archive** |
| 2. What will be downloaded | per position: the chosen file for each filter (combined or single, exposure time, offset, size), the number of GZH and CANDELS galaxies in the field, and whether the run will train or apply; **Download & run** |
| 3. Run | progress bar (stages finished), the last log lines, **Stop the run**, auto-refresh; or run on the files already in `data/raw/` |
| 4. Results | tabs: Overview (stage status, delivery checks), Catalogue, Galaxy atlas (browse by class, hide cosmic-ray suspects), Stage figures, Log |

The run starts in the background, so the page stays usable while it runs.

**`tools/run_field.py` options:** `--target RA,Dec[,name]` (repeat for more), `--instrument`, `--filters`
(default `auto`), `--radius` (MAST search radius in degrees, default 0.03), `--combined-only`,
`--mode auto|train|apply`, `--model-bundle models/<name>`, `--search-only`, `--keep-previous`,
`--no-run` (download only), `--from <stage>`.

For every position the runner uses the same file choice as `get_fits.py` (the other bands also prefer files that
overlap the red one), downloads the images with the Gaia DR3 cone into `data/`, moves the previous run
(`data/raw` files and `outputs/`) to `runs/<time>/` unless *keep previous* is set, writes `config_run.yaml`
and runs every stage.

**Notebook panel.** `Pixels_to_Provenance_Pipeline.ipynb` has the same inputs (positions, instrument, filters,
mode) as text boxes and dropdowns.

**Train or apply.** Galaxy Zoo: Hubble labels exist only in some HST surveys (AEGIS, COSMOS, GEMS, GOODS-N/S).
With at least `field_runner.min_gzh_labels` (150) GZH galaxies in the field(s) the models are **trained** there
(the original design). Otherwise the field is run in **apply** mode: Stage 4a extracts the CNN features with the
same frozen ImageNet networks, Stages 4b-4c copy a frozen **model bundle** (`models/<name>/`: the morphology and
merger XGBoost models, `fusion_meta.json` with the ENSEMBLE block weights, calibration, `model_card.json`), Stage 4d is skipped,
and Stage 8 classifies every galaxy with it. If the field has some GZH or CANDELS labels, Stages 6a and 7a-7c use
all of them as an out-of-field test. Export a bundle from any finished training run:
`python tools/export_model.py --name goods_2026-10-30` (after the first training run of the fusion design;
bundles of the earlier dual-branch design cannot be loaded).

The field runner and the app have been tested against a simulated MAST archive; check the `--search-only`
output (or section 2 of the app) before the first real download.

## 4. Run on your own files

1. **Images** → `data/raw/`. For each pointing, the same field in 2–3 filters. Use `tools/get_fits.py`
   (section 2) and copy the downloaded field into `data/raw/`, or download by hand from the
   [MAST Portal](https://mast.stsci.edu) (HST, calibration level 3, DRC).
   The older `tools/download_mast.py --ra 53.12 --dec -27.80 --radius 0.03 --filters F435W F606W F814W`
   still works but does not check coverage or prefer combined files.
   Check a file with `python tools/inspect_fits.py data/raw/<file>_drc.fits`.
   For training, choose a field that **Galaxy Zoo: Hubble covers** (AEGIS, COSMOS, GEMS, GOODS-N, GOODS-S);
   other fields run in apply mode.
   COSMOS has HST imaging in F814W only — the pipeline then runs with one band.
2. **Catalogues** → `data/catalogs/` (see `data/catalogs/README.md`):
   * GZH Table 4: `gz_hubble_main.csv.gz` from
     https://zooniverse-data.s3.amazonaws.com/galaxy-zoo-hubble/gz_hubble_main.csv.gz (required)
   * Gaia DR3 (optional, for star validation) — `get_fits.py --gaia`, or run in the Gaia archive
     (https://gea.esac.esa.int/archive/) and save as `data/catalogs/gaia_dr3.csv`:
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

## 5. Running single stages

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

## 6. Stages, files and outputs

28 stages. Execution order puts **Phase 5 before Phase 4**: the supervised models train on the PSF-refined
Pass-2 features they are later deployed on.

| Stage | File (`src/p2p/stages/`) | Main outputs (`outputs/<folder>/`) |
|---|---|---|
| 0 Sky-region partitioning | `stage0_partition.py` | `split_rule.json`, `split_regions.png` |
| 1a Acquisition / scan | `stage1a_acquisition.py` | `manifest.json`, `pointings.json` |
| 1b Frame QC (CAE) | `stage1b_frame_qc.py` | `tile_qc.csv`, `*_qc.png`, `qc_threshold.json`, `qc_training.png` |
| 1c Detection + aligned cutouts | `stage1c_detection.py` | `sources.csv`, `cutouts_raw.npy`, `segstamps.npy`, `P*_cutouts.png` |
| 1d PSF homogenisation | `stage1d_psf_match.py` | `cutouts.npy` (used from here on), `psf_match.json` |
| 1e Pass-1 morphology | `stage1e_pass1_morphology.py` | `features_pass1.csv`, `pass1_morphology.png` |
| 2a Representation + fusion | `stage2a_representation.py` | `hybrid_features.npy`, `rep_cae.pt`, `pca_scree.png`, `cae_training.png`, `cae_diagnostics.json` |
| 2b Clustering | `stage2b_clustering.py` | `clusters.csv`, `gmm_k_selection.png` (elbow / BIC / ICL; = `bic.png`), `gmm_k_selection.csv` |
| 2c Baseline + Gaia match | `stage2c_baseline.py` | `baseline.csv`, `gaia_match.csv` |
| 2d Semantic audit | `stage2d_semantic_audit.py` | `semantic_audit.json` (incl. `clustering_validation`), `montage_cluster_*.png` |
| 2e Continuum gating | `stage2e_continuum_gating.py` | `segregation.csv`, `anomaly_log.csv` |
| 3a GZH labels | `stage3a_gzh_labels.py` | `labels.csv`, `label_summary.json` |
| 3b Star routing | `stage3b_star_routing.py` | `star_routing.csv` |
| 3c Noise model | `stage3c_noise_model.py` | `noise_model.json`, `fpr_curve.png` |
| 5a Empirical PSF | `stage5a_empirical_psf.py` | `epsf_<pointing>.fits`, `epsf.png` |
| 5b Pass-2 refinement | `stage5b_pass2_refinement.py` | `features_pass2.csv`, `morph_delta_pass.csv` |
| 4a CNN features (ResNet-18 + EfficientNet-B0, frozen) | `stage4a_cnn_features.py` | `cnn_features.npy` (N x 1,792), `object_id.npy`, `feature_info.json`, `model_summary.txt` |
| 4b Ensemble (feature fusion) + XGBoost | `stage4b_xgboost_fusion.py` | `xgb_morphology.joblib`, `xgb_merger.joblib`, `fusion_meta.json`, `weight_search.csv/.png`, `training_history_xgboost.png`, `cv_curve.png`, `model_summary.txt`, `oof_fusion.npz`, `cv_folds.csv` |
| 4c Calibration | `stage4c_calibration.py` | `temperature.json`, `reliability_dev.png` (fitted on out-of-fold predictions) |
| 4d Ablation | `stage4d_ablation.py` | `ablation_table.csv`, `ablation.png`, `summary.json` |
| 6a In-domain test | `stage6a_test_eval.py` | `test_metrics.json`, `confusion_matrix_test.png` |
| 6b Uncertainty audit | `stage6b_uncertainty.py` | `uncertainty_audit.json` |
| 6c Explainable AI (SHAP) | `stage6c_xai.py` | `shap_blocks.csv`, `shap_physical.csv`, `shap_summary.png`, `examples.png` |
| 7a Taxonomy ceiling | `stage7a_taxonomy.py` | `taxonomy_alignment_matrix.csv`, `taxonomy_ceiling.json` |
| 7b Cross-catalogue | `stage7b_crosscat.py` | `cross_catalogue_metrics.json` |
| 7c Generalisation | `stage7c_generalisation.py` | `generalisation_report.json` |
| 8 Catalogue | `stage8_catalogue.py` | **`catalog.csv`**, `catalog_columns.csv`, `delivery_checks.json` |
| 9 Galaxy image atlas | `stage9_galaxy_atlas.py` | **`galaxy_images.zip`**, `images/<class>/<designation>.png`, `galaxy_images_index.csv`, `gallery_<class>.png`, `galaxy_atlas_overview.png` |

Shared code: `context.py` (config, folders, stage list), `fitsio.py` (reading `_drc` files),
`detection.py`, `morphology.py` (statmorph), `catalogs.py` (GZH/Gaia/CANDELS loaders),
`nets.py` (all neural networks), `features.py`, `inference.py`, `utils.py`, `compat.py`,
`archive.py` (MAST search / download, Gaia cones), `field.py` (field runner), `modes.py` (train / apply).
Every stage file starts with a docstring: *Purpose / Method / Output*.

Tools (`tools/`):

| Tool | Use |
|---|---|
| `get_fits.py` | download `_drc` images for sky positions (download only, section 2) |
| `run_field.py` | search, download and run the pipeline for sky positions (section 3) |
| `download_mast.py` | older download tool by RA/Dec, radius and filters |
| `download_catalogs.py` | fetch the GZH / CANDELS catalogues |
| `check_catalog_columns.py` | confirm the catalogue column mapping |
| `inspect_fits.py` | header, WCS, centre and size of a FITS file |
| `inspect_catalog.py` | summary of a finished `catalog.csv` |
| `show_stage.py` | what a stage did, its files and numbers |
| `export_model.py` | save a trained run as a model bundle for apply mode |

## 7. How objects are linked across stages

`sources.csv` (Stage 1c) gives every detection an `object_id` and a `cutout_index` (its row in
`cutouts.npy`). Every later table is keyed by `object_id`; `p2p.context.load_objects()` merges them.
`split_tag` (train / dev / test / external / empty = collar) is fixed in Stage 1c from the Stage-0
rules and never changes.

## 8. What to expect with only a few files

With one pointing, Stage 0 cuts the pointing into 70/15/15 stripes. One ACS pointing holds roughly
100–200 GZH-labelled galaxies, so the test split has only a few dozen: Stage 6a reports 95 %
bootstrap intervals — quote them. Train and test then share depth and PSF, so the in-domain test is
easier than a separate field. With one field there is no separate CANDELS pointing: `partition.phase7_region: auto` then evaluates
Phase 7 on the held-out **test** split (never trained on, so leakage-free; same depth and PSF as training,
so the gap measures the change of label scheme, not a new field). Without the Kartaltepe table
Stages 7a–7c are reported as *skipped* and the catalogue is still produced.

**Stage 1b QC autoencoder (v2).** Trains until a separate set of validation tiles stops improving (0.5 % over
15 epochs, never while it is still < 2 % better than a tile filled with its own mean), keeps the best epoch and
logs its gain over that flat tile. `qc_training.png` shows the loss curves; `qc_threshold.json` gives the
threshold with a 95 % bootstrap interval and the flagged share (`frame_qc:` in `config.yaml`; `qc_version: 1`
reproduces the original fixed-step training).

**Stage 2a autoencoder (CAE v2).** The representation CAE sees every cutout on one noise scale per band and
pointing (local background subtracted, divided by the band's background sigma, asinh above 3 sigma), plus a
0/1 "data present" channel per band, so a band outside its footprint is not read as a dark source. Its loss
weights the target footprint 10x and ignores cosmic-ray pixels; it trains with rotations/flips, uses the dev
region as validation set and stops when the validation loss has not improved for 12 epochs (`representation:`
in `config.yaml`; `cae_version: 1` reproduces the original CAE). Check three outputs after a run:
`cae_training.png` (train and validation loss; best epoch), `pca_scree.png` (left: explained variance; right:
|Spearman rho| of each principal component with the band background levels - should be low - and with
physical features) and `cae_diagnostics.json` (reconstruction gain over a flat per-band image, linear probe
of the kept components for cosmic-ray candidates and point-like vs extended sources).

**How many PCs, how many clusters (Stages 2a-2b).** `representation.n_pca: auto` keeps the components chosen by
Horn's parallel analysis (eigenvalue above the 95th percentile of the same latent with every column shuffled;
red line in `pca_scree.png`). The GMM uses **full** covariance and sweeps k = 2-20: k is the minimum BIC;
`gmm_k_selection.png` also shows ICL, AIC, the held-out (dev) log-likelihood and the diagonal-covariance curve
(which never reaches a minimum - the reason the old setting ended on the boundary k = 14). The log reports
whether other random seeds give the same minimum and the bootstrap stability (ARI). HDBSCAN is scored by DBCV
(internal) and, in Stage 2d, against the independent Gaia / GZH / noise evidence (`clustering_validation`:
purity vs the majority-class baseline, NMI); because the sources form one density continuum, Stage 2e uses its
GLOSH outlier score (`density_outlier`) rather than its two clusters.

**Supervised stages: feature-level fusion and cross-validation.** Stage 4a resizes every galaxy cutout to
224 x 224 (R, G, B = F775W, F606W, F435W, asinh stretch, ImageNet mean/std) and passes it once through two
frozen ImageNet networks: ResNet-18 -> 512 features, EfficientNet-B0 -> 1,280 features (global average pooling;
no CNN is trained). **ENSEMBLE step (Stage 4b):** the 512 + 1,280 CNN features and the 14 physical features of
Pass 2 are concatenated into one 1,806-feature vector per galaxy; each block gets a weight w1 / w2 / w3 (sum 1)
that sets its share of the features every XGBoost tree may sample (`feature_weights` = w_block / block size).
The candidate weights (`supervised.fusion_block_weights`) are compared by 5-fold cross-validation: the train +
dev galaxies are pooled and split into 5 folds (stratified by class, grouped by 30" sky cell); the weights and
the number of trees with the lowest out-of-fold log-loss are chosen (`weight_search.png`,
`training_history_xgboost.png`). The final morphology XGBoost (5 classes) and merger XGBoost (yes / no) are
retrained on all pooled galaxies; Stage 4c calibrates on the out-of-fold probabilities and Stage 4d ablates
each source (ResNet-18 only, EfficientNet-B0 only, physical only, CNN only, 128 px input, previous ensemble) on
the same folds. The test split is still used once (6a).

**Test set sealed while developing (`evaluation.test_sealed: true`).** Stages 6a-7c are skipped: no score is
computed on the test pointing or on the Phase-7 (external) pointings, so nothing can be tuned on them. Every
decision uses the cross-validation of train + dev (Stages 4a-4d). When the model is final, set
`evaluation.test_sealed: false` and run `python run_pipeline.py --from 6a` once; `test_split_ledger.json`
records every evaluation of the test split.

**Check the unsupervised stage by eye.** On real data open
`outputs/stage2d_semantic_audit/montage_cluster_*.png` and `semantic_audit.json`: each montage should
look like the name it was given. If a "Galaxy" cluster is full of point sources, adjust and rerun from 2a.

**Galaxy images (Stage 9).** `galaxy_images.zip` holds a colour PNG (reddest band = red) of every catalogue galaxy, in
one folder per predicted class, and `galaxy_images_index.csv` links each image to its catalogue row. Cosmic rays
are removed from the images only (display), not from the data. A "galaxy" whose central light is mostly removed is
a cosmic ray that reached the galaxy track: it is flagged `cr_suspect` in the index and shown in
`gallery_cr_suspects.png` - check those before using the catalogue.

## 9. Troubleshooting

| Message | Fix |
|---|---|
| `... not found - run stage X first` | run the earlier stage, or use `--from` |
| `No labelled galaxies in split(s) [...]` | the collar removed a stripe: lower `partition.collar_arcmin`; or the field has too few GZH labels: use apply mode (`supervised.mode: apply`, `supervised.model_bundle: models/<name>`) |
| `No Galaxy Zoo: Hubble galaxies cover this field` | apply mode (as above), or let `tools/run_field.py` / the app choose |
| `apply mode: this field has N bands ...` | download the same number of filters as the bundle (`model_card.json` -> `bands`) |
| `get_fits.py`: `-> nothing usable for ...` | the line above it says why (0 images, none cover the position, or only single exposures); try another position, `--instrument`, or `--allow-single` |
| MAST search finds only single exposures | normal for some fields; tick *combined exposures only* to refuse them, or accept more cosmic rays |
| `catalogs: required column(s) not found` | fix `catalogs.<kind>.columns` (see `tools/check_catalog_columns.py`) |
| statmorph build error | `pip install --no-build-isolation statmorph` |
| `streamlit: command not found` | `pip install -r requirements.txt` in the active `.venv` |
| very slow Stage 1e / 5b | statmorph is ~0.5 s per source per core; set `project.n_jobs: -1` and/or lower `detection.max_sources_per_pointing` |
| out of memory in 1c | real frames are ~4200² px x 3 bands; close other programs or process fewer pointings at a time |
| HDBSCAN found no clusters | set `clustering.hdbscan_min_cluster_size` to a small number (e.g. 10) |
| `BIC still falling at k_max` | raise `clustering.k_max`; the elbow (knee) of the BIC curve is used meanwhile |
| want to redo only PCA / clustering | `representation.reuse_trained_cae: true`, then `--from 2a` (the CAE is loaded, not retrained) |

Runtime guide (4-core laptop, 1 real pointing x 3 bands, ~2000 detections): Stages 1b–1c ~5 min,
1e ~5 min, 2a ~5–10 min on Apple MPS / a GPU (much longer on CPU: early stopping decides the number of epochs), 5b ~3 min, 4a ~1–3 min per 1,000 galaxies on Apple MPS (the first run downloads the ImageNet weights, ~66 MB),
4b–4d ~10-20 min on CPU (XGBoost, 5-fold cross-validation for each candidate ENSEMBLE weight set).
