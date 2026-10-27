"""Zheng et al. 2018 Wide & Deep CNN on the SGCC pools (plan §2 N2; Stage 2 run sheet §5).

The non-circular check of the consumer level: real SGCC theft labels, Zheng's
80 : 20 split (``sgcc_detector_train/test.parquet``; the test pool is used
for nothing else).  Source: Zheng, Yang, Niu, Dai, Zhou, "Wide and Deep
Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart
Grids", IEEE Trans. Industrial Informatics 2018 (author manuscript
TII-17-2030).

- **Preprocessing:** Zheng §IV-A Eqs. (1)-(3) per consumer were already
  applied when the pools were written (``sgcc.zheng_preprocess``, in the
  published order); they are not applied again (Eq. (2) is not idempotent;
  user decision 27 Oct).  :func:`load_pool` checks every value is in [0, 1].
- **Wide:** one fully connected layer of alpha = 90 ReLU units on the
  1,034-day series (Table III).
- **Deep:** the series reshaped by week to 148 x 7 (zero-padded at the end);
  R = 3 convolutional layers (Fig. 5) of gamma = 90 filters, 3 x 3, zero
  padding 1; each filter is applied to g1(B) + g2(B) of the 3 x 3 window B
  (Eqs. 7-9): g1 replaces each row by twice itself minus the other two rows,
  g2 the same for columns.  Both are linear, g1(B) = M B and g2(B) = B M with
  M = [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], so sum(K * (MB + BM)) =
  sum((MK + KM) * B): the effective kernel is M K + K M (M symmetric).  tanh
  activation (Eq. 7), global max pooling, then a fully connected layer of
  beta = 60 ReLU units.
- **Output:** wide and deep concatenated, one sigmoid unit (Eq. 11), binary
  cross-entropy.
- **Training [our choices; Zheng does not state them]:** Adam, learning rate
  1e-3, batch 128, at most 30 epochs, early stopping on validation AUC with
  patience 5 (best epoch restored).  Validation = 10 % of the training pool,
  stratified by FLAG, rng 20261027 [our choice; Zheng has no validation set].
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 90                  # wide units (Table III)
GAMMA_FILTERS = 90          # deep filters per layer (Table III)
BETA = 60                   # deep fully connected units (Table III)
R_LAYERS = 3                # convolutional layers (Fig. 5)
WEEK = 7
SEED = 20261027             # run sheet §5 [our choice]
VAL_SHARE = 0.10            # [our choice]
LR, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 128, 30, 5     # [our choices]
G = np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]])


def load_pool(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(consumer ids, X float32 (n, days), FLAG) of a written detector pool."""
    df = pd.read_parquet(path)
    x = df.drop(columns=["CONS_NO", "FLAG"]).to_numpy(np.float32)
    if np.isnan(x).any() or x.min() < 0.0 or x.max() > 1.0:
        raise ValueError(f"{Path(path).name}: not the Eq. (1)-(3) preprocessed pool (NaN or outside [0, 1])")
    return df.CONS_NO.to_numpy(), x, df.FLAG.to_numpy(np.int64)


def stratified_split(flag: np.ndarray, share: float = VAL_SHARE, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """(train rows, validation rows): ``share`` of each FLAG class, seeded."""
    rng = np.random.default_rng(seed)
    val = []
    for c in np.unique(flag):
        rows = np.flatnonzero(flag == c)
        val.append(rng.permutation(rows)[: int(round(share * len(rows)))])
    val = np.sort(np.concatenate(val))
    return np.setdiff1d(np.arange(len(flag)), val), val


def weekly(x, n_days: int):
    """(n, days) -> (n, 1, weeks, 7), zero-padded at the end."""
    import torch
    weeks = -(-n_days // WEEK)
    pad = weeks * WEEK - n_days
    x = torch.nn.functional.pad(x, (0, pad))
    return x.reshape(x.shape[0], 1, weeks, WEEK)


def build_model(n_days: int):
    import torch
    from torch import nn

    class GConv(nn.Conv2d):
        """3 x 3 convolution applied to g1(B) + g2(B): effective kernel M K + K M."""
        def __init__(self, cin, cout):
            super().__init__(cin, cout, 3, padding=1)
            self.register_buffer("m", torch.tensor(G, dtype=torch.float32))

        def forward(self, x):
            w = self.m @ self.weight + self.weight @ self.m
            return nn.functional.conv2d(x, w, self.bias, padding=1)

    class WideDeep(nn.Module):
        def __init__(self):
            super().__init__()
            self.wide = nn.Sequential(nn.Linear(n_days, ALPHA), nn.ReLU())
            layers, cin = [], 1
            for _ in range(R_LAYERS):
                layers += [GConv(cin, GAMMA_FILTERS), nn.Tanh()]
                cin = GAMMA_FILTERS
            self.conv = nn.Sequential(*layers)
            self.deep_fc = nn.Sequential(nn.Linear(GAMMA_FILTERS, BETA), nn.ReLU())
            self.out = nn.Linear(ALPHA + BETA, 1)

        def forward(self, x):
            h = self.conv(weekly(x, n_days))
            h = torch.amax(h, dim=(2, 3))                        # global max pooling
            z = torch.cat([self.wide(x), self.deep_fc(h)], dim=1)
            return self.out(z).squeeze(1)                        # logit; sigmoid in the loss

    return WideDeep()


def predict(model, x: np.ndarray, device, batch: int = 1024) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).to(device)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def train(x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray, seed: int, device=None,
          log=print) -> tuple[object, dict]:
    """Train one seed; early stopping on validation AUC, best epoch restored."""
    import copy

    import torch
    from sklearn.metrics import roc_auc_score
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = build_model(x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    xt, yt = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    best, best_state, wait, hist = -1.0, None, 0, []
    for ep in range(1, MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()
        order = np_rng.permutation(len(x))
        total = 0.0
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            xb, yb = xt[b].to(device), yt[b].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
        auc = float(roc_auc_score(yv, predict(model, xv, device)))
        hist.append({"epoch": ep, "train_loss": total / len(x), "val_auc": auc,
                     "seconds": round(time.perf_counter() - t0, 1)})
        log(f"seed {seed} epoch {ep}: loss {total / len(x):.4f} val AUC {auc:.4f} ({hist[-1]['seconds']} s)")
        if auc > best:
            best, best_state, wait = auc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, {"best_val_auc": best, "best_epoch": int(max(hist, key=lambda h: h["val_auc"])["epoch"]),
                   "epochs_run": len(hist), "history": hist, "device": str(device)}
