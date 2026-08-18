#!/usr/bin/env python3
"""
train.py — Fine-tune the mc3_18 (Kinetics-400) speed regressor.

Two-phase transfer learning:
  Phase 1 (warmup): backbone frozen, train only the regression head.
  Phase 2 (finetune): unfreeze everything, train at a lower LR.

Split by session_id: VAL_SESSION held out for validation (cross-day,
same user).  X is stored uint8 in the cache and converted to float [0..1]
here; the model bakes in Kinetics normalisation internally.

Usage:
  python3 train.py [--cache dataset/cache_side.npz]
                   [--warmup 5] [--finetune 25]
                   [--out models/speed_cnn.pt]
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader

from model import SpeedCNN

SEED        = 42
VAL_SESSION = 7   # session_20260818_001601 — held-out day, different outfit (adidas shorts)


def load_cache(path):
    d = np.load(path)
    X   = torch.from_numpy(d["X"])            # uint8 (N,3,T,H,W)
    y   = torch.from_numpy(d["y"])            # float32
    sid = torch.from_numpy(d["session_id"])
    print(f"Loaded {path}: X={tuple(X.shape)}  y={tuple(y.shape)}  dtype={X.dtype}")
    print(f"  speed range {y.min():.1f}→{y.max():.1f}  mean={y.mean():.2f}  std={y.std():.2f}")
    print(f"  sessions: {sorted(set(sid.tolist()))}")
    return X, y, sid


def augment(x):
    """In-place-ish augmentation on (3, T, H, W) float32 [0..1] clip."""
    if torch.rand(1) < 0.5:
        x = x.flip(dims=[3])                            # horizontal flip
    x = (x * (0.8 + torch.rand(1) * 0.4)).clamp(0, 1)   # brightness jitter
    return x


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, do_aug=False):
        self.X, self.y, self.do_aug = X, y, do_aug

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].float() / 255.0        # uint8 → [0..1]
        if self.do_aug:
            x = augment(x)
        return x, self.y[i:i+1]


def run_epoch(model, dl, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_mae, total_loss, n = 0.0, 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(xb)
            total_mae  += (pred - yb).abs().sum().item()
            n += len(xb)
    if device == "cuda":
        torch.cuda.empty_cache()
    return total_loss / n, total_mae / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",    default="dataset/cache_side.npz")
    parser.add_argument("--warmup",   type=int,   default=5)
    parser.add_argument("--finetune", type=int,   default=25)
    parser.add_argument("--bs",       type=int,   default=16)
    parser.add_argument("--lr_head",  type=float, default=1e-3)
    parser.add_argument("--lr_full",  type=float, default=1e-4)
    parser.add_argument("--out",      default="models/speed_cnn.pt")
    parser.add_argument("--workers",  type=int, default=0,
                        help="DataLoader workers (raise to ~8 on a GPU box)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # MPS is avoided (mc3_18 3D-conv backward hangs — PyTorch 2.8 bug on Mac).
    # On an NVIDIA box CUDA is used automatically; otherwise fall back to CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Device: cpu")

    X, y, sid = load_cache(args.cache)
    n_frames = X.shape[2]

    mask_val = sid == VAL_SESSION
    X_train, y_train = X[~mask_val], y[~mask_val]   # mask indexing already copies
    X_val,   y_val   = X[mask_val],  y[mask_val]
    del X, y                                        # free the 1.2GB combined tensor
    print(f"Train: {len(X_train)}  Val (session {VAL_SESSION}, cross-day): {len(X_val)}")

    pin = device == "cuda"
    train_dl = DataLoader(ClipDataset(X_train, y_train, do_aug=True),
                          batch_size=args.bs, shuffle=True,  num_workers=args.workers,
                          pin_memory=pin)
    val_dl   = DataLoader(ClipDataset(X_val,   y_val,   do_aug=False),
                          batch_size=args.bs, shuffle=False, num_workers=args.workers,
                          pin_memory=pin)

    model = SpeedCNN(n_frames=n_frames, pretrained=True).to(device)
    criterion = nn.HuberLoss(delta=1.0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    history = []
    best_val_mae = float("inf")

    def save_best(epoch, val_mae):
        nonlocal best_val_mae
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_mae": val_mae, "args": {"n_frames": n_frames}}, args.out)
            print(f"  ✓ best (val_mae={val_mae:.3f})")

    # ── Phase 1: warmup — head only ──────────────────────────────────────────
    print("\n=== Phase 1: warmup (backbone frozen) ===")
    model.freeze_backbone()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr_head, weight_decay=1e-4)
    for epoch in range(1, args.warmup + 1):
        t0 = time.time()
        _, tr_mae = run_epoch(model, train_dl, device, criterion, opt)
        _, va_mae = run_epoch(model, val_dl,   device, criterion)
        print(f"[warmup {epoch:2d}/{args.warmup}] train_mae={tr_mae:.3f}  "
              f"val_mae={va_mae:.3f}  {time.time()-t0:.1f}s")
        history.append({"phase": "warmup", "epoch": epoch,
                        "train_mae": round(tr_mae, 4), "val_mae": round(va_mae, 4)})
        save_best(epoch, va_mae)

    # ── Phase 2: finetune — full network ─────────────────────────────────────
    print("\n=== Phase 2: finetune (backbone unfrozen) ===")
    model.unfreeze_backbone()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr_full, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.finetune)
    for epoch in range(1, args.finetune + 1):
        t0 = time.time()
        _, tr_mae = run_epoch(model, train_dl, device, criterion, opt)
        _, va_mae = run_epoch(model, val_dl,   device, criterion)
        sched.step()
        print(f"[finetune {epoch:2d}/{args.finetune}] train_mae={tr_mae:.3f}  "
              f"val_mae={va_mae:.3f}  lr={sched.get_last_lr()[0]:.6f}  {time.time()-t0:.1f}s")
        history.append({"phase": "finetune", "epoch": epoch,
                        "train_mae": round(tr_mae, 4), "val_mae": round(va_mae, 4)})
        save_best(args.warmup + epoch, va_mae)

    hist_path = args.out.replace(".pt", "_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val MAE: {best_val_mae:.3f} km/h  →  {args.out}")


if __name__ == "__main__":
    main()
