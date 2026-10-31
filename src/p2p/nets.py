"""Neural networks: the autoencoders trained in Stages 1b and 2a, and the two frozen ImageNet feature extractors
of Stage 4a (ResNet-18, EfficientNet-B0); metrics shared by Stages 4-6."""
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


class RepAutoencoder(nn.Module):
    """Stage 2a representation autoencoder (cae_version 2).

    Input : bands data channels (field-noise asinh scale) + bands 0/1 "data present" channels.
    Encoder: 4 x [3x3 conv stride 2 + GroupNorm + SiLU] (128 -> 8 px), 1x1 conv to 32 channels,
             Linear(32*8*8 -> latent_dim).  The 1x1 reduction replaces the old 16,384 -> 64 dense layer
             (1.0 M weights -> 65 k) without throwing away WHERE the light is, which global average
             pooling would do (the cutouts are centred on the source, so position is information).
    Decoder: mirror image with nearest-neighbour upsampling + 3x3 conv; linear output (the input
             can be slightly negative), data channels only."""

    def __init__(self, bands=3, latent_dim=32, size=128, channels=(16, 32, 64, 64), mask_channels=True):
        super().__init__()
        self.bands, self.mask_channels = bands, mask_channels
        cin = bands * (2 if mask_channels else 1)
        layers, c = [], cin
        for ch in channels:
            layers += [nn.Conv2d(c, ch, 3, stride=2, padding=1), nn.GroupNorm(min(8, ch), ch), nn.SiLU()]
            c = ch
        self.encoder_conv = nn.Sequential(*layers, nn.Conv2d(c, 32, 1), nn.SiLU())
        self.s = size // 2 ** len(channels)
        self.enc_fc = nn.Linear(32 * self.s * self.s, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, 32 * self.s * self.s)
        dec, c = [nn.Conv2d(32, channels[-1], 1), nn.SiLU()], channels[-1]
        for ch in list(channels[::-1][1:]) + [channels[0]]:
            dec += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, ch, 3, padding=1),
                    nn.GroupNorm(min(8, ch), ch), nn.SiLU()]
            c = ch
        dec += [nn.Conv2d(c, bands, 3, padding=1)]
        self.decoder_conv = nn.Sequential(*dec)

    def encode(self, x):
        return self.enc_fc(self.encoder_conv(x).flatten(1))

    def decode(self, z):
        return self.decoder_conv(self.dec_fc(z).view(-1, 32, self.s, self.s))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def _rep_batch(X, P, W, idx, device, augment=False):
    """Float32 tensors for one batch; X data, P present mask, W loss weights (all (n, bands, S, S))."""
    x = torch.from_numpy(np.asarray(X[idx], np.float32))
    p = torch.from_numpy(np.asarray(P[idx], np.float32))
    w = torch.from_numpy(np.asarray(W[idx], np.float32)) if W is not None else torch.ones_like(x)
    if augment:                                   # morphology does not depend on orientation
        k = int(np.random.randint(4))
        if k:
            x, p, w = (torch.rot90(t, k, dims=(2, 3)) for t in (x, p, w))
        if np.random.rand() < 0.5:
            x, p, w = (torch.flip(t, dims=(3,)) for t in (x, p, w))
    return x.to(device), p.to(device), w.to(device)


def _rep_input(model, x, p):
    return torch.cat([x, p], 1) if model.mask_channels else x


def rep_loss(model, x, p, w):
    """Weighted MSE per cutout (weights already mean-normalised per cutout)."""
    recon, z = model(_rep_input(model, x, p))
    return (w * (recon - x) ** 2).mean(dim=(1, 2, 3)), z, recon


def train_rep_autoencoder(model, X, P, W, train_idx, val_idx, device, max_epochs=150, min_epochs=20, patience=12,
                          batch_size=64, lr=1e-3, weight_decay=1e-5, lr_patience=4, augment=True, min_delta=0.005,
                          checkpoint=None, no_stop_above=None, restore_lowest=False, log=print, tag="Rep-CAE"):
    """Adam + ReduceLROnPlateau on the validation loss; early stopping when the validation loss has not
    improved by at least `min_delta` (relative) for `patience` epochs; best weights kept (and written to
    `checkpoint` at every improvement, so an interrupted run keeps its best model). While the best validation
    loss is still above `no_stop_above` (e.g. 98 % of a trivial flat-image model) training is not stopped early:
    an autoencoder can sit on that plateau for many epochs before it starts to learn structure.
    restore_lowest=False keeps the last epoch that improved by min_delta (as Keras' restore_best_weights);
    True keeps the epoch with the LOWEST validation loss - min_delta then only drives the patience count.
    Returns the training history (train / validation loss per epoch)."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=lr_patience)
    best, bad, hist = np.inf, 0, {"train": [], "val": [], "lr": []}
    kept, best_state, best_ep = np.inf, None, 0             # the weights that are returned
    train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
    for ep in range(max_epochs):
        model.train()
        perm = np.random.permutation(train_idx)
        run = 0.0
        for i in range(0, len(perm), batch_size):
            b = np.sort(perm[i:i + batch_size])
            x, p, w = _rep_batch(X, P, W, b, device, augment)
            loss = rep_loss(model, x, p, w)[0].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * len(b)
        tr = run / len(train_idx)
        va = float(rep_errors(model, X, P, W, val_idx, device)[0].mean())
        sched.step(va)
        hist["train"].append(tr)
        hist["val"].append(va)
        hist["lr"].append(opt.param_groups[0]["lr"])
        improved = va < best * (1 - min_delta)                # counts for early stopping
        if improved:
            best, bad = va, 0
        else:
            bad += 1
        keep = va < kept if restore_lowest else improved
        if keep:
            kept, best_state, best_ep = va, copy.deepcopy(model.state_dict()), ep + 1
            if checkpoint is not None:
                torch.save(best_state, checkpoint)
        # "*" = weights kept (lowest validation loss so far); "no gain k/patience" = epochs since the last
        # improvement of at least min_delta - training stops early when k reaches patience
        log(f"  [{tag}] epoch {ep + 1:3d}  train {tr:.5f}  val {va:.5f}  lr {opt.param_groups[0]['lr']:.1e}"
            f"  no gain {bad}/{patience}" + ("  *" if keep else ""))
        on_plateau = no_stop_above is not None and best > no_stop_above
        if ep + 1 >= min_epochs and on_plateau and bad == patience:
            log(f"  [{tag}] no improvement for {patience} epochs, but still no better than the trivial model "
                f"({best:.5f} > {no_stop_above:.5f}) - continuing")
        if ep + 1 >= min_epochs and bad >= patience and not on_plateau:
            log(f"  [{tag}] early stop: validation loss not improved by {min_delta:.1%} for {patience} epochs")
            break
    if best_state is None:
        raise RuntimeError(f"[{tag}] the validation loss was never finite - no weights to keep")
    model.load_state_dict(best_state)
    hist["best_epoch"] = int(best_ep)
    hist["best_val"] = float(kept)
    return hist


@torch.no_grad()
def rep_errors(model, X, P, W, idx, device, batch=128):
    """Weighted reconstruction error and latent code for cutouts `idx` (no augmentation)."""
    model.to(device).eval()
    errs, zs = [], []
    idx = np.asarray(idx)
    for i in range(0, len(idx), batch):
        x, p, w = _rep_batch(X, P, W, idx[i:i + batch], device)
        e, z, _ = rep_loss(model, x, p, w)
        errs.append(e.cpu().numpy())
        zs.append(z.cpu().numpy())
    return np.concatenate(errs), np.concatenate(zs)


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
class ResidualBlock(nn.Module):
    """residual_block(x, filters, stride) - written as in Keras:
        shortcut = x
        x = Conv2D(filters, 3, strides=stride, padding='same')(x); x = BatchNormalization()(x); x = Activation('relu')(x)
        x = Conv2D(filters, 3, strides=1, padding='same')(x);      x = BatchNormalization()(x)
        if stride != 1 (or the channel count changes):
            shortcut = Conv2D(filters, 1, strides=stride)(shortcut); shortcut = BatchNormalization()(shortcut)
        x = Add()([x, shortcut]); return Activation('relu')(x)
    Layer names follow torchvision (conv1, bn1, conv2, bn2, downsample), so ImageNet weights load unchanged."""

    def __init__(self, in_filters, filters, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_filters, filters, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters, filters, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.downsample = None
        if stride != 1 or in_filters != filters:
            self.downsample = nn.Sequential(nn.Conv2d(in_filters, filters, 1, stride=stride, bias=False),
                                            nn.BatchNorm2d(filters))

    def forward(self, x):
        shortcut = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        return self.relu(x + shortcut)                      # Add()([x, shortcut]) + relu


class ResNet18(nn.Module):
    """ResNet-18 feature extractor (Stage 4a), 11.2 M parameters, written like a Keras model:
        inputs (N, 3, 224, 224)                                       (supervised.input_size; PyTorch fixes only
        x = Conv2D(64, 7, strides=2) + BatchNormalization + relu      -> (N, 64, 112, 112)   the channel counts,
        x = MaxPooling2D(3, strides=2)                                -> (N, 64, 56, 56)     any image size works)
        x = residual_block(x, 64);       x = residual_block(x, 64)    -> (N, 64, 56, 56)
        x = residual_block(x, 128, 2);   x = residual_block(x, 128)   -> (N, 128, 28, 28)
        x = residual_block(x, 256, 2);   x = residual_block(x, 256)   -> (N, 256, 14, 14)
        x = residual_block(x, 512, 2);   x = residual_block(x, 512)   -> (N, 512, 7, 7)
        x = GlobalAveragePooling2D()(x)                               -> (N, 512)
    No classifier: the 512 pooled values are the features. pretrained=True loads torchvision's ImageNet weights
    (same layer names); the 3 input channels are R, G, B = reddest, middle, bluest band."""

    def __init__(self, in_ch=3, pretrained=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))
        self.layer2 = nn.Sequential(ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128))
        self.layer3 = nn.Sequential(ResidualBlock(128, 256, stride=2), ResidualBlock(256, 256))
        self.layer4 = nn.Sequential(ResidualBlock(256, 512, stride=2), ResidualBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d(1)               # GlobalAveragePooling2D
        self.flatten = nn.Flatten()
        if pretrained:
            w = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).state_dict()
            self.load_state_dict({k: v for k, v in w.items() if not k.startswith("fc.")})
        if in_ch != 3:          # another band count: a new first convolution, started from the ImageNet filters
            old = self.conv1.weight.data                     # (64, 3, 7, 7)
            self.conv1 = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
            with torch.no_grad():                            # every channel gets the mean RGB filter, scaled so the
                self.conv1.weight.copy_(old.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * 3.0 / in_ch)  # response
                                                             # to a grey image is unchanged
        self.out_dim = 512

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.flatten(self.avgpool(x))


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 feature extractor (Stage 4a), torchvision's architecture with ImageNet weights.
        inputs (N, 3, 224, 224)
        x = Conv2D(32, 3, strides=2) + BatchNormalization + SiLU                    -> (N, 32, 112, 112)
        x = 16 MBConv blocks (depthwise conv + squeeze-excitation, SiLU)            -> (N, 320, 7, 7)
        x = Conv2D(1280, 1) + BatchNormalization + SiLU                             -> (N, 1280, 7, 7)
        x = GlobalAveragePooling2D()(x)                                             -> (N, 1280)
    The ImageNet classifier (Dropout + Dense 1280 -> 1000) is not used: the 1,280 pooled values are the features.
    4,007,548 parameters."""

    def __init__(self, pretrained=True):
        super().__init__()
        w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.efficientnet_b0(weights=w)
        self.features = net.features                         # convolution stages
        self.avgpool = net.avgpool                           # GlobalAveragePooling2D
        self.flatten = nn.Flatten()
        self.out_dim = 1280

    def forward(self, x):
        return self.flatten(self.avgpool(self.features(x)))


def frozen(model):
    """Feature extractor mode: no gradients, BatchNorm uses its ImageNet statistics, Dropout off."""
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


@torch.no_grad()
def model_summary(model, *inputs, name=None, max_depth=2):
    """Keras-style model.summary(): one row per layer (down to `max_depth` levels of nesting; None = every leaf
    layer, as Keras lists them) with its
    output shape and weight count, from one forward pass of `inputs` (dummy tensors are fine)."""
    def shape(o):
        if torch.is_tensor(o):
            return str(list(o.shape))
        if isinstance(o, dict):
            return ", ".join(f"{k}: {shape(v)}" for k, v in o.items())
        if isinstance(o, (tuple, list)):
            return ", ".join(shape(v) for v in o)
        return "-"

    rows, hooks = [], []
    for path, mod in model.named_modules():
        depth = path.count(".") + 1 if path else 0
        if max_depth is None:                       # full depth: every leaf layer (each Conv2d, BN, ReLU, ...)
            if depth == 0 or len(list(mod.children())):
                continue
        elif depth == 0 or depth > max_depth or (depth < max_depth and len(list(mod.children()))):
            continue
        n = sum(p.numel() for p in mod.parameters())
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, path=path, n=n: rows.append((path, type(m).__name__, shape(o), n))))
    was_training = model.training
    model.eval()
    try:
        out = model(*inputs)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    w1 = max([len(f"{p} ({t})") for p, t, _, _ in rows] + [20])
    w2 = max([len(s) for _, _, s, _ in rows] + [12])
    line = "=" * (w1 + w2 + 18)
    txt = [line, f'Model: "{name or type(model).__name__}"', line,
           f"{'Layer (type)':<{w1}}  {'Output shape':<{w2}}  {'Param #':>12}", "-" * len(line)]
    txt += [f"{f'{p} ({t})':<{w1}}  {s:<{w2}}  {n:>12,}" for p, t, s, n in rows]
    txt += [line, f"Input : {', '.join(str(list(x.shape)) for x in inputs)}", f"Output: {shape(out)}",
            f"Total params: {total:,} ({total * 4 / 2 ** 20:.2f} MB float32)",
            f"Trainable params: {trainable:,}", f"Non-trainable params: {total - trainable:,}", line]
    return "\n".join(txt)


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
           "accuracy": round(float((pred == y).mean()), 4) if len(y) else np.nan,
           "balanced_acc": _balanced_acc(y, pred),
           "n": int(len(y))}
    if pm is not None and ym is not None and 0 < np.asarray(ym).sum() < len(ym):
        out["merger_pr_auc"] = round(float(average_precision_score(ym, pm)), 4)
        out["merger_prevalence"] = round(float(np.mean(ym)), 4)
    return out


def nll(y, P):
    y = np.asarray(y)
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)))) if len(y) else np.inf


# ================================================================ calibration
def softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))
