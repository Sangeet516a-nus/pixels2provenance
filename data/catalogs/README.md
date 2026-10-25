# data/catalogs — reference catalogues (CSV, CSV.GZ, FITS or VOTable)

| Catalogue | Needed by | File name must contain | Where to get it |
|---|---|---|---|
| Galaxy Zoo: Hubble, Table 4 (Willett et al. 2017) | Stage 2d (evidence), 3a (labels) — **required** | `gz_hubble` or `gzh` | https://zooniverse-data.s3.amazonaws.com/galaxy-zoo-hubble/gz_hubble_main.csv.gz |
| Gaia DR3 cone around your pointings | Stage 2c, 2d, 3b (star confirmation) — optional | `gaia` | Gaia archive ADQL (README.md, section 2), or set `catalogs.gaia.query_online: true` |
| CANDELS visual morphology (Kartaltepe et al. 2015), FRACTIONAL table | Stage 7 (cross-catalogue) — optional | `kartaltepe`, `candels` or `k15` | VizieR `J/ApJS/221/11/table3` (`python tools/download_catalogs.py candels` → `candels_k15.csv`) |

Quickest way: `python tools/download_catalogs.py gzh`, `... gaia --ra <RA> --dec <Dec>`, `... candels`.

After adding a file run `python tools/check_catalog_columns.py`: it shows which configured
column names were found and suggests the closest names for any that are missing.
The GZH column names in `config.yaml` are the real Table-4 names, and the CANDELS names are the real
VizieR table3 labels (`RAJ2000`, `f(Sph)`, `f(Dk)`, `f(Ir)`, `f(PS)`, `f(UnCl)`, `f(M)`, `NCl`) with common
alternative spellings. Use table3 (fractions), not table2 (raw votes).
