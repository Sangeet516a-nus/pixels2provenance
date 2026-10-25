#!/usr/bin/env python
"""Look inside an HST drizzled product: extensions, key header values, WCS,
pixel statistics, and a picture of SCI and WHT.

    python tools/inspect_fits.py data/raw/<file>_drc.fits
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p2p.fitsio import load_image, read_meta, valid_mask  # noqa: E402


def main(path):
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval

    path = Path(path)
    with fits.open(path) as h:
        print("=== Extensions ===")
        h.info()
    m = read_meta(path)
    print("\n=== What the pipeline reads (Stage 1a) ===")
    for k in ("instrument", "filter", "exptime", "ndrizim", "date_obs", "targname", "bunit", "photflam", "photplam",
              "zp_ab", "pixscale_arcsec", "naxis1", "naxis2", "ra_centre", "dec_centre", "product_type"):
        print(f"  {k:16s} = {m[k]}")
    sci, wht, w = load_image(path)
    val = valid_mask(sci, wht)
    d = sci[val]
    med = np.median(d)
    mad = 1.4826 * np.median(np.abs(d - med))
    print("\n=== Pixels ===")
    print(f"  valid (WHT > 0)  = {val.mean():.1%} of the frame")
    print(f"  sky median       = {med:.5f}   robust sigma = {mad:.5f}  ({m['bunit']})")
    print(f"  min / max        = {d.min():.4g} / {d.max():.4g}")
    norm = ImageNormalize(np.where(val, sci, np.nan), interval=PercentileInterval(99.5), stretch=AsinhStretch(0.05))
    ny, nx = sci.shape
    c = np.s_[ny // 2 - 256:ny // 2 + 256, nx // 2 - 256:nx // 2 + 256]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(np.where(val, sci, np.nan), origin="lower", cmap="gray", norm=norm)
    ax[0].set_title(f"SCI {m['filter']} (asinh)")
    ax[1].imshow(sci[c], origin="lower", cmap="gray", norm=norm)
    ax[1].set_title("central 512 px")
    if wht is not None:
        ax[2].imshow(wht, origin="lower", cmap="viridis")
        ax[2].set_title("WHT (0 = no data)")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(path.name)
    fig.tight_layout()
    out = path.parent / (path.name.split(".fits")[0] + "_preview.png")
    fig.savefig(out, dpi=90)
    print(f"\nPreview saved: {out}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1])
