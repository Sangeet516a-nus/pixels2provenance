"""Neural networks and training loops shared by Stages 1b, 2a and 4a-4d."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score

MORPH_CLASSES = ("Smooth", "FaceOnSpiral", "EdgeOn", "Irregular", "Ambiguous")
N_CLASSES = len(MORPH_CLASSES)


# ================================================================ autoencoders
class ConvAutoencoder(nn.Module):
    """Under-complete convolutional autoencoder, 128x128 input.
    Conv C->16->32->64 (128 -> 16 px), Linear bottleneck to latent_dim, mirror decoder.
    Stage 1b uses C=1 (one tile of one band); Stage 2a uses C = number of bands."""

    def __init__(self, in_ch=1, latent_dim=64, size=128):
        super().__init__()
        s = size // 8
        self.s = s
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())
        self.enc_fc = nn.Linear(64 * s * s, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, 64 * s * s)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, in_ch, 3, 2, 1, output_padding=1), nn.Sigmoid())

    def encode(self, x):
        return self.enc_fc(self.encoder_conv(x).flatten(1))

    def decode(self, z):
        return self.decoder_conv(self.dec_fc(z).view(-1, 64, self.s, self.s))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def train_autoencoder(model, X, epochs, batch_size, device, lr=1e-3, weight_decay=1e-5, log=print, tag="CAE"):
    """X: float32 array (N, C, H, W) already scaled to [0, 1]."""
    model.to(device).train()
    data = torch.tensor(X)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(epochs):
        perm = torch.randperm(len(data))
        run = 0.0
        for i in range(0, len(data), batch_size):
            xb = data[perm[i:i + batch_size]].to(device)
            recon, _ = model(xb)
            loss = F.mse_loss(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * len(xb)
        if epochs <= 30 or (ep + 1) % max(1, epochs // 15) == 0 or ep + 1 == epochs:   # ~15 log lines
            log(f"  [{tag}] epoch {ep + 1:2d}/{epochs}  recon MSE = {run / len(data):.5f}")
    return model


@torch.no_grad()
def ae_errors_and_latents(model, X, device, batch=256):
    model.to(device).eval()
    errs, zs = [], []
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i:i + batch]).to(device)
        r, z = model(xb)
        errs.append(((xb - r) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
        zs.append(z.cpu().numpy())
    return np.concatenate(errs), np.concatenate(zs)


# ================================================================ supervised
class DualBranchNet(nn.Module):
    """ResNet-18 image branch (512-d) + Dense(n_tab -> 64) tabular branch, each
    LayerNorm-ed, concatenated, then Head A (5-class) and Head B (merger).
    mode='image' / 'tabular' removes a branch for the controlled ablation."""

    def __init__(self, n_tab, in_ch=3, n_classes=N_CLASSES, mode="hybrid", pretrained=False, dropout=0.3):
        super().__init__()
        self.mode = mode
        dim = 0
        if mode in ("hybrid", "image"):
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = torchvision.models.resnet18(weights=w)
            if in_ch != 3:
                self.backbone.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
            self.backbone.fc = nn.Identity()
            self.img_norm = nn.LayerNorm(512)
            dim += 512
        if mode in ("hybrid", "tabular"):
            self.tab_branch = nn.Sequential(nn.Linear(n_tab, 64), nn.ReLU())
            self.tab_norm = nn.LayerNorm(64)
            dim += 64
        self.drop = nn.Dropout(dropout)
        self.head_morph = nn.Linear(dim, n_classes)
        self.head_merger = nn.Linear(dim, 1)

    def forward(self, x_img, x_tab):
        parts = []
        if self.mode in ("hybrid", "image"):
            parts.append(self.img_norm(self.backbone(x_img)))
        if self.mode in ("hybrid", "tabular"):
            parts.append(self.tab_norm(self.tab_branch(x_tab)))
        z = self.drop(torch.cat(parts, dim=1))
        return {"morph": self.head_morph(z), "merger": self.head_merger(z).squeeze(1)}


class EfficientNetMember(nn.Module):
    def __init__(self, in_ch=3, n_classes=N_CLASSES, pretrained=False):
        super().__init__()
        w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = torchvision.models.efficientnet_b0(weights=w)
        if in_ch != 3:
            old = self.net.features[0][0]
            self.net.features[0][0] = nn.Conv2d(in_ch, old.out_channels, 3, 2, 1, bias=False)
        self.net.classifier[1] = nn.Linear(self.net.classifier[1].in_features, n_classes)

    def forward(self, x_img, x_tab=None):
        return {"morph": self.net(x_img)}


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, reduction="none")
        pt = torch.exp(-ce)
        w = self.alpha[y] if self.alpha is not None else 1.0
        return (w * (1 - pt) ** self.gamma * ce).mean()


def augment(x):
    """90-degree rotations and flips: morphology is orientation-invariant."""
    out = []
    for xi in x:
        xi = torch.rot90(xi, int(np.random.randint(4)), dims=(1, 2))
        out.append(torch.flip(xi, dims=(2,)) if np.random.rand() < 0.5 else xi)
    return torch.stack(out)


@torch.no_grad()
def predict_model(model, data, device, batch=64):
    model.to(device).eval()
    morph, merger = [], []
    for i in range(0, len(data["img"]), batch):
        out = model(data["img"][i:i + batch].to(device), data["tab"][i:i + batch].to(device))
        morph.append(F.softmax(out["morph"], 1).cpu().numpy())
        if "merger" in out:
            merger.append(torch.sigmoid(out["merger"]).cpu().numpy())
    res = {"morph": np.concatenate(morph)}
    if merger:
        res["merger"] = np.concatenate(merger)
    return res


def _balanced_acc(y, pred):
    import warnings
    if len(set(np.asarray(y).tolist())) < 2:
        return np.nan
    with warnings.catch_warnings():                     # "y_pred contains classes not in y_true"
        warnings.simplefilter("ignore", UserWarning)
        return round(float(balanced_accuracy_score(y, pred)), 4)


def morph_metrics(y, P, ym=None, pm=None):
    y = np.asarray(y)
    pred = P.argmax(1)
    present = sorted(set(y.tolist()) | set(pred.tolist()))      # classes absent from both are not scored
    out = {"macro_f1": round(float(f1_score(y, pred, average="macro", labels=present, zero_division=0)), 4),
           "balanced_acc": _balanced_acc(y, pred),
           "n": int(len(y))}
    if pm is not None and ym is not None and 0 < np.asarray(ym).sum() < len(ym):
        out["merger_pr_auc"] = round(float(average_precision_score(ym, pm)), 4)
        out["merger_prevalence"] = round(float(np.mean(ym)), 4)
    return out


def fit_classifier(model, tr, dev, device, epochs=15, batch_size=32, lr=1e-3, weight_decay=1e-4,
                   gamma=2.0, merger_weight=0.5, seed=0, log=print, tag="model"):
    """Focal loss on Head A (+ weighted BCE on Head B if present). Every model gets the
    same fixed epoch budget and keeps its best dev-Macro-F1 checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    y = tr["y"].numpy()
    counts = np.bincount(y, minlength=N_CLASSES)
    alpha = torch.tensor(len(y) / (N_CLASSES * np.maximum(counts, 1)), dtype=torch.float32).to(device)
    focal = FocalLoss(alpha, gamma)
    pos = float(tr["ym"].sum())
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - pos) / max(pos, 1.0)).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_nll, best_state, n = -1.0, -np.inf, copy.deepcopy(model.state_dict()), len(y)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        run = 0.0
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            if len(b) < 2:          # BatchNorm needs > 1 sample
                continue
            out = model(augment(tr["img"][b]).to(device), tr["tab"][b].to(device))
            loss = focal(out["morph"], tr["y"][b].to(device))
            if "merger" in out:
                loss = loss + merger_weight * bce(out["merger"], tr["ym"][b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * len(b)
        sched.step()
        p = predict_model(model, dev, device)
        yd = dev["y"].numpy()
        m = morph_metrics(yd, p["morph"])
        dev_nll = float(-np.mean(np.log(np.clip(p["morph"][np.arange(len(yd)), yd], 1e-12, 1)))) if len(yd) else np.inf
        log(f"  [{tag}] epoch {ep + 1:2d}/{epochs}  loss={run / n:.4f}  dev macro-F1={m['macro_f1']:.3f}  "
            f"dev NLL={dev_nll:.3f}")
        # best dev Macro-F1; ties (common with a small dev split) go to the lower dev log-loss
        if (m["macro_f1"], -dev_nll) > (best, best_nll):
            best, best_nll, best_state = m["macro_f1"], -dev_nll, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


# ================================================================ calibration
def softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))
