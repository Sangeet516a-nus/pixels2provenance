"""Fine-tuned CNN members of the supervised ensemble (Stage 4b): ResNet-18 and EfficientNet-B0, ImageNet weights,
the LAST stages retrained on the GZH-labelled galaxies.

Why fine-tune (30 Oct experiment, 1,118 train + dev galaxies, 5-fold CV): the frozen ImageNet features describe
everyday objects at 224 px; the galaxies are 10-20 px smudges in a ~50 px crop. Retraining the last stages on the
galaxies lifted ResNet-18 alone from Macro-F1 0.33 to 0.45 and EfficientNet-B0 alone from 0.39 to 0.46, and the
equal-weight average of both with the physical-feature XGBoost reached 0.51 (frozen-feature fusion: 0.47).

    galaxy stamp (Stage 4a preparation: cosmic rays cleaned, neighbours masked, crop of 4 r_petro)
        -> asinh, R G B = F775W F606W F435W, ImageNet mean / std, 112 x 112 (supervised.finetune.input_size)
        -> backbone (ImageNet weights; early stages FROZEN, the last stages TRAINED)
              ResNet-18      : conv1 .. layer2 frozen | layer3, layer4 trained        (trainable: last2)
              EfficientNet-B0: features.0 .. .4 frozen | features.5 .. .8 trained
        -> GlobalAveragePooling -> Dropout(0.3) -> Dense(5)                      (new head, trained)
    loss      : cross-entropy, balanced class weights, label smoothing 0.1 (GZH labels are noisy)
    optimiser : AdamW (decoupled L2 weight decay 1e-4), one-cycle learning rate (peak 3e-4), batch 32
    augment   : the 8 symmetries of the square (rotations by 90 deg, flips) - the class does not depend on them
    predict   : softmax averaged over 4 rotations (test-time augmentation)
    epochs    : fixed (12). The 20-epoch validation curves (30 Oct) were lowest at epoch 9-12, then flat while the
                training loss kept falling (memorising): more epochs only overfit.
BatchNorm layers of the frozen stages keep their ImageNet statistics (eval mode) during training.
"""
from __future__ import annotations

import time

import numpy as np

KINDS = ("resnet18", "efficientnet_b0")
MEMBER = {"resnet18": "resnet18_ft", "efficientnet_b0": "efficientnet_b0_ft"}
TRAINABLE = {"resnet18": {"last1": ("layer4",), "last2": ("layer3", "layer4"), "all": ("",)},
             "efficientnet_b0": {"last1": ("features.7", "features.8"),
                                 "last2": ("features.5", "features.6", "features.7", "features.8"), "all": ("",)}}
CAM_LAYER = {"resnet18": "backbone.layer3", "efficientnet_b0": "backbone.features.5"}   # 7 x 7 maps at 112 px
DEFAULTS = dict(enabled=True, networks=list(KINDS), input_size=112, epochs=12, batch_size=32, lr=3e-4,
                weight_decay=1e-4, label_smoothing=0.1, dropout=0.3, trainable="last2", tta_views=4,
                history_train_eval=True, soft_labels=False)


def ft_config(ctx) -> dict:
    sup = ctx.cfg.get("supervised", {})
    return {**DEFAULTS, "pretrained": bool(sup.get("pretrained", True)), **(sup.get("finetune") or {})}


def build(kind, cfg, pretrained=True):
    """Backbone with ImageNet weights + new head; requires_grad only for the trained stages and the head."""
    import torch.nn as nn
    from .nets import EfficientNetB0, ResNet18

    class FineTuneNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.backbone = ResNet18(in_ch=3, pretrained=pretrained) if kind == "resnet18" else \
                EfficientNetB0(pretrained=pretrained)
            d = self.backbone.out_dim
            self.head = nn.Sequential(nn.Dropout(float(cfg["dropout"])), nn.Linear(d, 5))
            self.trained = TRAINABLE[kind][cfg["trainable"]]
            for n, p in self.backbone.named_parameters():
                p.requires_grad_(n.startswith(self.trained))

        def train(self, mode=True):
            """Frozen stages stay in eval mode (BatchNorm keeps its ImageNet statistics)."""
            super().train(mode)
            if mode:
                for n, m in self.backbone.named_modules():
                    if isinstance(m, nn.BatchNorm2d) and not n.startswith(self.trained):
                        m.eval()
            return self

        def forward(self, x):
            return self.head(self.backbone(x))

    return FineTuneNet()


def stamps(ctx, rows, size=None, log=None):
    """ImageNet-normalised tensors (n, 3, size, size) of the galaxies in `rows` (cutout_index, r_petro, area_px),
    prepared exactly like the Stage 4a input (features.cnn_stamp with the same settings)."""
    import pandas as pd
    import torch
    from .features import cnn_stamp, imagenet_input, load_cutouts
    from .modes import cnn_input_settings
    cfg = cnn_input_settings(ctx)
    size = int(size or ft_config(ctx)["input_size"])
    cut = load_cutouts(ctx)
    seg = np.load(ctx.require("1c", "segstamps.npy"), mmap_mode="r")
    idx = np.asarray(rows["cutout_index"], int)
    rp = pd.to_numeric(rows.get("r_petro", pd.Series(np.nan, index=rows.index)), errors="coerce").to_numpy(float)
    ar = pd.to_numeric(rows.get("area_px", pd.Series(np.nan, index=rows.index)), errors="coerce").to_numpy(float)
    atlas = ctx.cfg.get("atlas", {})
    out = []
    for s in range(0, len(idx), 256):
        sl = range(s, min(s + 256, len(idx)))
        out.append(imagenet_input([cnn_stamp(cut[idx[i]], seg[idx[i]], rp[i], ar[i], cfg, atlas) for i in sl], size))
    return torch.cat(out) if out else torch.zeros((0, 3, size, size))


def _augment(x):
    import torch
    k = int(torch.randint(0, 4, (1,)))
    x = torch.rot90(x, k, (2, 3)) if k else x
    return torch.flip(x, (3,)) if bool(torch.rand(1) < 0.5) else x


def predict(model, X, device, views=4, batch=128):
    """Softmax probabilities (n, 5), averaged over `views` rotations by 90 deg."""
    import torch
    import torch.nn.functional as F
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = X[s:s + batch].to(device)
            p = 0
            for k in range(max(1, min(4, int(views)))):
                p = p + F.softmax(model(torch.rot90(xb, k, (2, 3)) if k else xb), 1)
            out.append((p / max(1, min(4, int(views)))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 5))


def vote_targets(ctx, rows):
    """GZH vote fractions of labelled galaxies as (n, 5) training targets (supervised.finetune.soft_labels)."""
    from .catalogs import load_gzh
    from .context import load_json
    from .stages.stage3a_gzh_labels import vote_fractions
    metas = list(load_json(ctx.require("1a", "manifest.json")).values())
    gzh = load_gzh(ctx, metas).set_index("gzh_row").loc[rows["gzh_row"].astype(int).to_numpy()]
    return vote_fractions(gzh, ctx.cfg["catalogs"]["gzh"]["thresholds"]).astype(np.float32)


def grad_cam(model, X, classes, device, batch=64):
    """Grad-CAM maps (n, H, W) in 0..1 of the classes given, from the network's last 7 x 7 layer (CAM_LAYER):
    where in the stamp the evidence for that class lies."""
    import torch
    import torch.nn.functional as F
    model.to(device).eval()
    layer = model.get_submodule(CAM_LAYER[model.kind])
    store = {}
    hook = layer.register_forward_hook(lambda _m, _i, o: store.update(a=o))
    out = []
    try:
        for s in range(0, len(X), batch):
            xb = X[s:s + batch].to(device).requires_grad_(True)      # gradients reach frozen layers through the input
            scores = model(xb)
            grads = torch.autograd.grad(scores[torch.arange(len(xb)), torch.as_tensor(classes[s:s + batch], device=device)].sum(),
                                        store["a"])[0]
            cam = F.relu((grads.mean((2, 3), keepdim=True) * store["a"]).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=X.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            out.append((cam / cam.flatten(1).max(1).values.clamp_min(1e-8)[:, None, None]).detach().cpu().numpy())
    finally:
        hook.remove()
    return np.concatenate(out) if out else np.zeros((0, *X.shape[-2:]))


def train(kind, X, y, cfg, device, seed=42, Xva=None, yva=None, log=None, tag="", soft=None):
    """Train one network on (X, y). With a validation set, the validation probabilities after EVERY epoch are
    returned too (for the CV curve). Returns model, history {loss, accuracy, val_loss, val_accuracy} per epoch,
    and the per-epoch validation probabilities (epochs, n_va, 5) or None. soft = (n, 5) vote fractions replace the
    hard targets in the training loss (label smoothing is then not needed); class weights and every reported score
    still use the hard labels."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build(kind, cfg, pretrained=bool(cfg.get("pretrained", True))).to(device)
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.long)
    counts = np.bincount(np.asarray(y), minlength=5).astype(float)
    cw = torch.tensor(len(y) / (5 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    bs, epochs = int(cfg["batch_size"]), int(cfg["epochs"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(cfg["lr"]),
                                                total_steps=epochs * int(np.ceil(len(y) / bs)))
    hist = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    val_probs = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(y))
        for s in range(0, len(perm), bs):
            b = torch.as_tensor(perm[s:s + bs])
            xb = _augment(X[b]).to(device)
            if soft is None:
                loss = F.cross_entropy(model(xb), y_t[b].to(device), weight=cw,
                                       label_smoothing=float(cfg["label_smoothing"]))
            else:
                w = cw[y_t[b].to(device)]
                t = torch.as_tensor(soft[b.numpy()], dtype=torch.float32, device=device)
                loss = -(w * (t * F.log_softmax(model(xb), 1)).sum(1)).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if cfg.get("history_train_eval", True):        # training loss / accuracy measured like the validation
            pt = predict(model, X, device, views=1)
            hist["loss"].append(float(-np.mean(np.log(np.clip(pt[np.arange(len(y)), np.asarray(y)], 1e-12, 1)))))
            hist["accuracy"].append(float((pt.argmax(1) == np.asarray(y)).mean()))
        if Xva is not None:
            pv = predict(model, Xva, device, views=int(cfg["tta_views"]))
            val_probs.append(pv)
            yv = np.asarray(yva)
            hist["val_loss"].append(float(-np.mean(np.log(np.clip(pv[np.arange(len(yv)), yv], 1e-12, 1)))))
            hist["val_accuracy"].append(float((pv.argmax(1) == yv).mean()))
        if log:
            msg = f"  [{kind}{tag}] epoch {ep + 1:2d}/{epochs}"
            if hist["loss"]:
                msg += f"  loss {hist['loss'][-1]:.4f}  accuracy {hist['accuracy'][-1]:.4f}"
            if hist["val_loss"]:
                msg += f"  val_loss {hist['val_loss'][-1]:.4f}  val_accuracy {hist['val_accuracy'][-1]:.4f}"
            log(msg + f"  ({time.time() - t0:.0f} s)")
    return model, hist, (np.stack(val_probs) if val_probs else None)


def cross_validate(kind, X, y, folds, cfg, device, seed=42, log=None, soft=None):
    """Out-of-fold probabilities after every epoch (epochs, n, 5) and the per-fold histories."""
    oof = np.zeros((int(cfg["epochs"]), len(y), 5))
    hists = []
    for f, (tr, va) in enumerate(folds):
        _, h, pv = train(kind, X[tr], np.asarray(y)[tr], cfg, device, seed + f, X[va], np.asarray(y)[va], log,
                         tag=f" fold {f + 1}/{len(folds)}", soft=None if soft is None else soft[tr])
        oof[:, va] = pv
        hists.append(h)
    return oof, hists


def save(model, path, cfg):
    import torch
    torch.save({"kind": model.kind, "state_dict": model.state_dict(), "config": dict(cfg)}, path)


def load(path, device="cpu"):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    m = build(d["kind"], {**DEFAULTS, **d["config"]}, pretrained=False)
    m.load_state_dict(d["state_dict"])
    return m.to(device).eval(), {**DEFAULTS, **d["config"]}


def predict_rows(ctx, model, cfg, rows, device, batch=512, log=None):
    """Probabilities of any galaxies (rows of galaxy_table), stamps built in batches (memory stays small)."""
    out = []
    t0 = time.time()
    for s in range(0, len(rows), batch):
        X = stamps(ctx, rows.iloc[s:s + batch], cfg["input_size"])
        out.append(predict(model, X, device, views=int(cfg["tta_views"])))
        if log and (s // batch) % 10 == 0:
            log(f"  [{model.kind}] predicted {min(s + batch, len(rows)):,} / {len(rows):,} galaxies ({time.time() - t0:.0f} s)")
    return np.concatenate(out) if out else np.zeros((0, 5))
