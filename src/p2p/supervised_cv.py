"""5-fold cross-validation of the supervised stages (4b, 4d) and its plots.

The TRAIN and DEV GZH-labelled galaxies are pooled (~1,100) and split into K folds (StratifiedGroupKFold: the class
mix is the same in every fold, and galaxies in the same 30" sky cell - which share image noise and can overlap in
the cutouts - stay in the same fold, as do galaxies closer than one cutout across a cell border). Every model is trained K times on K-1 folds and predicts the held-out fold:
each galaxy gets one OUT-OF-FOLD (OOF) prediction from a model that never saw it. Settings (number of trees,
ensemble block weights) are chosen by the lowest OOF log-loss; the final model is retrained on all pooled galaxies.
The OOF predictions also feed the calibration (Stage 4c). The TEST split is never touched (Stage 6a, once).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .nets import morph_metrics, nll

SMOOTH = 3          # points in the moving average (metric macro_f1 only)


def pooled(splits):
    """train + dev labelled galaxies as one table (column 'orig_split' keeps where each came from)."""
    pool = pd.concat([splits["train"].assign(orig_split="train"), splits["dev"].assign(orig_split="dev")],
                     ignore_index=True)
    return pool


def link_close_galaxies(groups, ra, dec, link_arcsec):
    """Merge the groups of galaxies closer than link_arcsec: two galaxies on either side of a sky-cell border would
    otherwise fall into different folds although one lies inside the other's cutout."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    from .utils import unit_vectors
    cells, ids = np.unique(groups, return_inverse=True)
    n, nc = len(ids), len(cells)
    pairs = cKDTree(unit_vectors(ra, dec)).query_pairs(2 * np.sin(np.deg2rad(link_arcsec / 3600) / 2), output_type="ndarray")
    a, b = np.r_[pairs[:, 0], np.arange(n)], np.r_[pairs[:, 1], n + ids]      # galaxy - galaxy and galaxy - cell edges
    _, comp = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(n + nc, n + nc)), directed=False)
    return comp[:n].astype(str)


def make_folds(pool, n_splits=5, seed=42, cell_arcsec=30.0, link_arcsec=6.4):
    from sklearn.model_selection import StratifiedGroupKFold
    y = pool.gz_label_id.astype(int).to_numpy()
    cell = cell_arcsec / 3600.0
    ra, dec = pool.ra.to_numpy(float), pool.dec.to_numpy(float)
    # RA x cos(dec): a cell is cell_arcsec wide ON THE SKY at every declination (at Dec +62 deg, GOODS-N, a plain
    # RA bin of 30" would be only 14" wide, and close neighbours could fall into different folds)
    groups = (np.floor(ra * np.cos(np.deg2rad(dec)) / cell).astype(int).astype(str) + "_" +
              np.floor(dec / cell).astype(int).astype(str))
    if link_arcsec:
        groups = link_close_galaxies(groups, ra, dec, link_arcsec)
    n_splits = int(max(2, min(n_splits, len(set(groups)))))
    folds = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(pool, y, groups))
    fold_id = np.zeros(len(pool), int)
    for f, (_, va) in enumerate(folds):
        fold_id[va] = f
    return folds, fold_id, groups


def bootstrap_ci(y, P, n=1000, seed=0):
    """95% bootstrap interval of Macro-F1 (resampling galaxies)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(morph_metrics(y[i], P[i])["macro_f1"])
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def choose_epoch(curve, min_epoch=3, metric="log_loss"):
    """curve: list of {'macro_f1', 'nll'} per epoch (or per tree count), out-of-fold.
    metric 'log_loss' (default): the minimum OOF log-loss - a proper scoring rule, smooth in the number of
    epochs / trees, so one lucky epoch cannot win (on ~200 galaxies Macro-F1 jumps by +-0.03 between
    neighbouring epochs). metric 'macro_f1': best 3-point moving-average Macro-F1, ties to the lower log-loss.
    Returns (1-based index, the smoothed Macro-F1 curve)."""
    f1 = np.array([c["macro_f1"] for c in curve], float)
    nl = np.array([c["nll"] for c in curve], float)
    sm = pd.Series(f1).rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy()
    cand = list(range(max(0, min_epoch - 1), len(f1))) or list(range(len(f1)))
    if metric == "macro_f1":
        best = max(cand, key=lambda e: (round(sm[e], 3), -nl[e]))
    else:
        best = min(cand, key=lambda e: nl[e])
    return best + 1, sm.round(4).tolist()


def mean_history(hist):
    """Per-fold history (folds x epochs) -> one Keras-style history dict (mean over the folds per epoch)."""
    return {k: np.mean(np.asarray(v), axis=0).tolist() for k, v in hist.items() if k in HISTORY_KEYS}


HISTORY_KEYS = ("loss", "val_loss", "accuracy", "val_accuracy")


def plot_history(history, name, path, xlabel="Epochs", chosen=None):
    """The Keras plot: history = {'loss', 'val_loss', 'accuracy', 'val_accuracy'} lists (one value per epoch;
    here the mean over the CV folds, 'val' = the held-out fold). Loss = cross-entropy of the 5-class head."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = np.arange(1, len(history["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(ep, history["loss"], label="Train Loss")
    plt.plot(ep, history["val_loss"], label="Val Loss")
    if chosen:
        plt.axvline(chosen, color="0.5", ls="--", lw=1, label=f"chosen: {chosen}")
    plt.title(f"{name} Loss")
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(ep, history["accuracy"], label="Train Acc")
    plt.plot(ep, history["val_accuracy"], label="Val Acc")
    if chosen:
        plt.axvline(chosen, color="0.5", ls="--", lw=1, label=f"chosen: {chosen}")
    plt.title(f"{name} Accuracy")
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=110, facecolor="white")
    plt.close()


def log_history(log, name, history, unit="epoch"):
    """Keras-style fit() lines: one per epoch (mean over the CV folds)."""
    log(f"{name} history (mean over the CV folds; val = held-out fold):\n" + "\n".join(
        f"  {unit} {e + 1:3d}  loss {history['loss'][e]:.4f}  accuracy {history['accuracy'][e]:.4f}  "
        f"val_loss {history['val_loss'][e]:.4f}  val_accuracy {history['val_accuracy'][e]:.4f}"
        for e in range(len(history["loss"]))))


def curve_metrics(y, oof, folds):
    out = []
    for e in range(oof.shape[0]):
        m = morph_metrics(y, oof[e])
        m["nll"] = round(nll(y, oof[e]), 4)
        m["fold_macro_f1"] = [morph_metrics(y[va], oof[e][va])["macro_f1"] for _, va in folds]
        out.append(m)
    return out


def summarise(y, P, folds, ym=None, pm=None):
    m = morph_metrics(y, P, ym, pm)
    fm = [morph_metrics(y[va], P[va])["macro_f1"] for _, va in folds]
    m.update({"nll": round(nll(y, P), 4), "fold_macro_f1": fm, "fold_mean": round(float(np.mean(fm)), 4),
              "fold_sd": round(float(np.std(fm, ddof=1)), 4) if len(fm) > 1 else None,
              "macro_f1_95ci": bootstrap_ci(y, P)})
    return m


def plot_curve(curves, chosen, path, title):
    """curves: {name: per-epoch metrics}; chosen: {name: epoch}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.7))
    for (name, cur), c in zip(curves.items(), pal):
        ep = np.arange(1, len(cur) + 1)
        f1 = np.array([x["macro_f1"] for x in cur])
        sd = np.array([np.std(x["fold_macro_f1"], ddof=1) / np.sqrt(len(x["fold_macro_f1"])) for x in cur])
        a.plot(ep, f1, "-", color=c, lw=2, label=f"{name} (chosen epoch {chosen[name]})")
        a.fill_between(ep, f1 - sd, f1 + sd, color=c, alpha=0.15, lw=0)
        a.axvline(chosen[name], color=c, ls="--", lw=1)
        b.plot(ep, [x["nll"] for x in cur], "-", color=c, lw=2, label=name)
    a.set_xlabel("epoch")
    a.set_ylabel("out-of-fold Macro-F1 (band = +-1 s.e. over folds)")
    a.set_title(title, fontsize=9.5)
    a.legend(fontsize=8, frameon=False)
    b.set_xlabel("epoch")
    b.set_ylabel("out-of-fold log-loss (lower = better)")
    b.set_title("Out-of-fold log-loss", fontsize=9.5)
    for ax in (a, b):
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)


def shape_scores(y, P):
    """How the model scores on the galaxies it gives a SHAPE (any predicted class except Ambiguous):
      coverage    : share of the galaxies given a shape; always report it with the scores below, because a model
                    could raise them by calling hard galaxies Ambiguous;
      accuracy    : share of those galaxies whose derived GZH label is the same shape (a label of Ambiguous is wrong);
      macro_f1    : Macro-F1 of the four shapes on those galaxies;
      both_*      : the same on the galaxies where the derived GZH label is a shape too."""
    from sklearn.metrics import f1_score
    from .nets import MORPH_CLASSES
    amb = MORPH_CLASSES.index("Ambiguous")
    shapes = [k for k in range(len(MORPH_CLASSES)) if k != amb]
    y, pred = np.asarray(y), np.asarray(P).argmax(1)
    s, b = pred != amb, (pred != amb) & (y != amb)
    f = lambda m: round(float(f1_score(y[m], pred[m], labels=shapes, average="macro", zero_division=0)), 4) if m.any() else None
    return {"n": int(len(y)), "n_shape": int(s.sum()), "coverage": round(float(s.mean()), 4) if len(y) else None,
            "accuracy": round(float((y[s] == pred[s]).mean()), 4) if s.any() else None, "macro_f1": f(s),
            "both_n": int(b.sum()), "both_accuracy": round(float((y[b] == pred[b]).mean()), 4) if b.any() else None,
            "both_macro_f1": f(b)}


def shape_text(s):
    return (f"shape given to {s['n_shape']} of {s['n']} galaxies ({s['coverage']:.1%}); on those: accuracy {s['accuracy']}, "
            f"4-class Macro-F1 {s['macro_f1']}; where the GZH label is a shape too (n = {s['both_n']}): accuracy "
            f"{s['both_accuracy']}, Macro-F1 {s['both_macro_f1']}")


def detailed_scores(y, P, conf=None, thresholds=(0.5, 0.7)):
    """The scores behind one Macro-F1 number:
      per_class    : precision / recall / F1 / support of each GZH class (which classes pull the average down);
      four_class   : Macro-F1 without 'Ambiguous' (galaxies the volunteers themselves could not place are left out,
                     and the prediction is the best of the four real classes);
      by_agreement : Macro-F1 of the galaxies whose GZH vote fraction (weakest link) is >= each threshold - how
                     well the model does where the human label is reliable."""
    from sklearn.metrics import precision_recall_fscore_support
    from .nets import MORPH_CLASSES, morph_metrics
    y = np.asarray(y)
    pred = P.argmax(1)
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=list(range(len(MORPH_CLASSES))), zero_division=0)
    out = {"per_class": {c: {"precision": round(float(pr[i]), 4), "recall": round(float(rc[i]), 4),
                             "f1": round(float(f1[i]), 4), "support": int(sup[i])} for i, c in enumerate(MORPH_CLASSES)}}
    out["shape"] = shape_scores(y, P)
    amb = MORPH_CLASSES.index("Ambiguous")
    keep = y != amb
    if keep.any():
        P4 = P[keep].copy()
        P4[:, amb] = -1
        out["four_class"] = morph_metrics(y[keep], P4)
    if conf is not None:
        c = np.nan_to_num(np.asarray(conf, float), nan=0.0)
        out["by_agreement"] = {f">={t}": morph_metrics(y[c >= t], P[c >= t]) for t in thresholds if (c >= t).sum()}
    return out


def detailed_text(d):
    rows = ["  class            precision  recall     F1   support"]
    rows += [f"  {c:<15s} {v['precision']:9.3f} {v['recall']:7.3f} {v['f1']:6.3f} {v['support']:8d}"
             for c, v in d["per_class"].items()]
    if "shape" in d:
        rows.append("  " + shape_text(d["shape"]))
    if "four_class" in d:
        rows.append(f"  4 classes (no Ambiguous): Macro-F1 {d['four_class']['macro_f1']}, accuracy "
                    f"{d['four_class']['accuracy']} (n = {d['four_class']['n']})")
    for t, m in d.get("by_agreement", {}).items():
        rows.append(f"  GZH agreement {t}: Macro-F1 {m['macro_f1']}, accuracy {m['accuracy']} (n = {m['n']})")
    return "\n".join(rows)
