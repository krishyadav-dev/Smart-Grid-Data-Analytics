"""Day-ahead DT load forecasts for the thermal engine (docs/Layer3_plan.md §3,
T3, T5; Stage 3 run sheet §4).

Target: per DT and phase, the load factor K_p for the 96 steps of day D,
issued at 00:00 on D from data up to D - 1 23:45: 39 series (13 DTs x 3
phases).  Inputs are the metered DT currents only (via
:func:`src.analytics.thermal.read_dt_currents`); no air temperature (T5).

- Persistence: K_p(D - 1, k).  Seasonal naive: K_p(D - 7, k).
- LSTM [architecture our choice within the plan, run sheet §4]: one global
  model over the 39 series; per input step the series' K_p, time of day as
  sin/cos of the step, weekday one-hot, DT one-hot and phase one-hot, over the
  last 672 steps; 96 outputs in one shot; MSE on K_p; Adam 1e-3, batch 256, at
  most 50 epochs, early stopping with patience 5 on the validation loss (the
  best epoch's weights are kept); three seeds, primary forecast = seed mean.
  Dropout 0.1 on the last hidden state (and between layers for 2 layers)
  [our choice of placement].
- Training samples (user decision 28 Oct): every input and target day from
  2024-01-02 (2024-01-01 is the thermal warm-up), so the first target day is
  2024-01-09.  Training targets end 15 days before the first origin of a
  training round; early stopping on the 14 days before that origin.  Samples
  are batched in a seeded random order within the training set; the split
  itself is by date, never shuffled.
- Negative forecasts are clipped at 0 (a current magnitude; user decision 28 Oct).
- Variants (user decision 28 Oct, step C, after the direct LSTM underperformed
  persistence on the April validation origins): ``direct`` (the LSTM outputs
  K_p(D, k)) and ``residual`` (the LSTM is trained on K_p(D, k) - K_p(D - 1, k);
  forecast = K_p(D - 1, k) + output, clipped at 0 after adding).  Same inputs,
  sizes, seeds and training settings; the variant is chosen with the size.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

LOOKBACK = 672                       # 7 days of 15-min steps (plan §3)
HORIZON = 96
STEPS_PER_DAY = 96
SEEDS = (20261028, 20261029, 20261030)          # run sheet §4
GRID_LAYERS = (1, 2)
GRID_UNITS = (32, 64, 128)
DROPOUT = 0.1
LR = 1e-3
BATCH = 256
MAX_EPOCHS = 50
PATIENCE = 5
VAL_DAYS = 14                        # early stopping on the 14 days before the origin
GAP_DAYS = 15                        # training targets end 15 days before the origin
DATA_START = pd.Timestamp("2024-01-02")          # first day used (inputs and targets)
SELECTION_TRAIN = (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-03-17"))
SELECTION_VAL = (pd.Timestamp("2024-03-18"), pd.Timestamp("2024-03-31"))
SELECTION_ORIGINS = (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-04-30"))
TEST_ORIGINS = (pd.Timestamp("2024-05-01"), pd.Timestamp("2024-12-30"))
MODELS = ("persistence", "seasonal_naive", "lstm")
VARIANTS = ("direct", "residual")


# --------------------------------------------------------------------------
# Series and baselines
# --------------------------------------------------------------------------

def daily_series(k: np.ndarray) -> np.ndarray:
    """K_p [T, DT, phase] -> [day, 96, series] with series = DT * 3 + phase."""
    T_, n_dt, n_ph = k.shape
    return k.reshape(T_ // STEPS_PER_DAY, STEPS_PER_DAY, n_dt * n_ph)


def to_dt_phase(x: np.ndarray, n_dt: int, n_ph: int = 3) -> np.ndarray:
    """[..., series] -> [..., DT, phase]."""
    return x.reshape(x.shape[:-1] + (n_dt, n_ph))


def day_index(days: pd.DatetimeIndex, d) -> int:
    return int(days.get_loc(pd.Timestamp(d)))


def persistence(kd: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """K_p(D - 1, k) for origin day indices ``origins``: [origin, 96, series]."""
    return kd[origins - 1]


def seasonal_naive(kd: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """K_p(D - 7, k): [origin, 96, series]."""
    return kd[origins - 7]


# --------------------------------------------------------------------------
# LSTM
# --------------------------------------------------------------------------

def features(kd: np.ndarray, days: pd.DatetimeIndex, n_dt: int, n_ph: int = 3) -> np.ndarray:
    """Per-step input features [series, T, F]: K_p, sin/cos of the step of day,
    weekday one-hot (7), DT one-hot, phase one-hot."""
    n_days, _, n_s = kd.shape
    T_ = n_days * STEPS_PER_DAY
    step = np.tile(np.arange(STEPS_PER_DAY), n_days)
    ang = 2 * np.pi * step / STEPS_PER_DAY
    wd = np.repeat(np.eye(7)[np.asarray(days.weekday)], STEPS_PER_DAY, axis=0)          # [T, 7]
    common = np.concatenate([np.sin(ang)[:, None], np.cos(ang)[:, None], wd], axis=1)     # [T, 9]
    ident = np.concatenate([np.repeat(np.eye(n_dt), n_ph, axis=0), np.tile(np.eye(n_ph), (n_dt, 1))], axis=1)
    load = kd.reshape(T_, n_s).T[:, :, None]                                               # [S, T, 1]
    return np.concatenate([load, np.broadcast_to(common, (n_s, T_, 9)),
                           np.broadcast_to(ident[:, None, :], (n_s, T_, ident.shape[1]))],
                          axis=2).astype(np.float32)


def samples(day_range: tuple, days: pd.DatetimeIndex, n_series: int) -> np.ndarray:
    """(series, target day index) pairs for target days in ``day_range``
    (inclusive) whose 7-day lookback starts on or after :data:`DATA_START`."""
    first = max(day_index(days, day_range[0]), day_index(days, DATA_START) + LOOKBACK // STEPS_PER_DAY)
    d = np.arange(first, day_index(days, day_range[1]) + 1)
    s, dd = np.meshgrid(np.arange(n_series), d, indexing="ij")
    return np.stack([s.ravel(), dd.ravel()], axis=1)


def _torch():
    import torch
    return torch


def build_model(n_features: int, layers: int, units: int, dropout: float = DROPOUT):
    torch = _torch()
    nn = torch.nn

    class LoadLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, units, num_layers=layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(units, HORIZON)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(self.drop(out[:, -1]))

    return LoadLSTM()


def n_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


class Windows:
    """Lookback windows and targets gathered on the device from the feature tensor."""

    def __init__(self, feats: np.ndarray, kd: np.ndarray, device, variant: str = "direct"):
        torch = _torch()
        self.torch = torch
        self.variant = variant
        self.x = torch.from_numpy(feats).to(device)                                   # [S, T, F]
        self.y = torch.from_numpy(np.ascontiguousarray(
            target(kd, variant).transpose(2, 0, 1)).astype(np.float32)).to(device)   # [S, day, 96]
        self.offs = torch.arange(-LOOKBACK, 0, device=device)
        self.device = device

    def batch(self, pairs: np.ndarray):
        t = self.torch.from_numpy(pairs).to(self.device)
        s, d = t[:, 0], t[:, 1]
        idx = d[:, None] * STEPS_PER_DAY + self.offs[None, :]
        return self.x[s[:, None], idx], self.y[s, d]


def train(win: Windows, train_pairs: np.ndarray, val_pairs: np.ndarray, layers: int, units: int,
          seed: int, log=None) -> tuple[object, dict]:
    """Train one LSTM; returns the model with the best validation weights and a log."""
    torch = _torch()
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = build_model(win.x.shape[2], layers, units).to(win.device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    gen = np.random.default_rng(seed)
    best, best_state, wait, hist = np.inf, None, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = gen.permutation(len(train_pairs))
        tot = 0.0
        for i in range(0, len(order), BATCH):
            xb, yb = win.batch(train_pairs[order[i:i + BATCH]])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(xb)
        val = evaluate_loss(model, win, val_pairs)
        hist.append({"epoch": epoch, "train_mse": tot / len(order), "val_mse": val})
        if log:
            log(f"    epoch {epoch:2d} train {tot / len(order):.6f} val {val:.6f}")
        if val < best:
            best, wait = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE or not np.isfinite(val):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_mse": float(best), "epochs": len(hist),
                   "best_epoch": int(np.argmin([h["val_mse"] for h in hist]) + 1) if np.isfinite(best) else None,
                   "history": hist, "seconds": round(time.perf_counter() - t0, 1),
                   "n_train": int(len(train_pairs)), "n_val": int(len(val_pairs))}


def evaluate_loss(model, win: Windows, pairs: np.ndarray) -> float:
    pred = predict(model, win, pairs)
    y = win.y[win.torch.from_numpy(pairs[:, 0]).to(win.device),
              win.torch.from_numpy(pairs[:, 1]).to(win.device)].cpu().numpy()
    return float(np.mean((pred - y) ** 2))


def predict(model, win: Windows, pairs: np.ndarray) -> np.ndarray:
    """Raw (unclipped) forecasts [n_pairs, 96]."""
    torch = _torch()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), 4 * BATCH):
            xb, _ = win.batch(pairs[i:i + 4 * BATCH])
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def forecast_origins(model, win: Windows, origins: np.ndarray, n_series: int) -> np.ndarray:
    """Raw LSTM forecasts [origin, 96, series] for origin day indices ``origins``."""
    s, d = np.meshgrid(np.arange(n_series), origins, indexing="ij")
    pairs = np.stack([s.ravel(), d.ravel()], axis=1)
    return predict(model, win, pairs).reshape(n_series, len(origins), HORIZON).transpose(1, 2, 0)


def target(kd: np.ndarray, variant: str) -> np.ndarray:
    """Training target per day: K_p (``direct``) or K_p(D) - K_p(D - 1) (``residual``;
    day 0 has no predecessor and is never a target)."""
    if variant == "direct":
        return kd
    if variant != "residual":
        raise ValueError(variant)
    out = np.zeros_like(kd)
    out[1:] = kd[1:] - kd[:-1]
    return out


def compose(raw: np.ndarray, kd: np.ndarray, origins: np.ndarray, variant: str) -> np.ndarray:
    """K_p forecast from the LSTM output [origin, 96, series] (before clipping)."""
    return raw + kd[origins - 1] if variant == "residual" else raw


def clip(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Negative K_p forecasts set to 0 (user decision 28 Oct); returns the count clipped."""
    return np.maximum(x, 0.0), int((x < 0).sum())


def training_round(origin: pd.Timestamp) -> tuple[tuple, tuple]:
    """(training target days, early-stopping days) for a round whose first origin is ``origin``."""
    return ((DATA_START, origin - pd.Timedelta(days=GAP_DAYS)),
            (origin - pd.Timedelta(days=VAL_DAYS), origin - pd.Timedelta(days=1)))
