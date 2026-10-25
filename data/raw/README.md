# data/raw — put your HST drizzled products here

The pipeline reads every file matching `*_drc.fits`, `*_drz.fits` (or `.fits.gz`) in this folder
(sub-folders included). `_drc` is preferred when both exist for the same rootname.

* Files whose centres are closer than 60" (config: `acquisition.group_radius_arcsec`) form one **pointing**.
* Give each pointing its 2–3 filters (e.g. F435W, F606W, F814W) — the cutouts have one channel per band.
* Mark CANDELS pointings for the cross-catalogue test with `partition.external_patterns` in `config.yaml`
  (any substring of the file name or of the TARGNAME header keyword).

Get files with `python tools/download_mast.py --ra <deg> --dec <deg> --filters F435W F606W F814W`
or by hand from the MAST Portal (https://mast.stsci.edu): HST, calibration level 3, product type DRC.
