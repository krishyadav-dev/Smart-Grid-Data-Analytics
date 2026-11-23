"""Report-mode views over the git-ignored detail files (plan.md §3.2).

These read the regenerable parquet / npz outputs of Layers 1-3 (year cache,
fault set, theft and thermal detail).  As in :mod:`results`, nothing is
re-estimated: values are selected, aggregated (daily max / sum) and re-shaped.
The one formula applied is the voltage-unbalance identity from
:mod:`src.analytics.unbalance` (``vuf_pct_from_ll``), on the metered DT
voltages, for the per-meter heatmap.

Ground truth (fault buses, true theft per DT-day, thief labels) appears only in
fields whose names start with ``true_`` / ``truth`` and is labelled on screen as
simulation ground truth (plan.md §3.4).

Every view returns ``{"available": False, "how": ...}`` when its file is missing.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from .results import VU_LIMIT_PCT, missing

STEPS_PER_DAY = 96
HALF_CYCLE_MS = 10.0                     # 50 Hz: burst samples are half-cycle RMS
ENERGY_THEFT_FOLDS = ("h1", "h2", "h3", "h4")


def _mtime(p: Path) -> float:
    return p.stat().st_mtime


def _r(x, nd: int = 3):
    """Round floats (and arrays) for JSON; NaN -> None."""
    a = np.asarray(x, dtype=float)
    out = np.where(np.isfinite(a), np.round(a, nd), np.nan).tolist()
    if isinstance(out, float):
        return None if out != out else out
    return _nan_to_none(out)


def _nan_to_none(v):
    if isinstance(v, list):
        return [_nan_to_none(x) for x in v]
    return None if isinstance(v, float) and v != v else v


def dt_names(config_dir: Path) -> list[str]:
    """The 13 DT load ids in ``meters.yaml`` order (the column order of the thermal npz)."""
    import yaml
    m = yaml.safe_load((config_dir / "meters.yaml").read_text(encoding="utf-8"))
    return [x["load_id"] for x in m["meters"] if x["role"] == "dt"]


def physical_meters(config_dir: Path) -> list[dict]:
    import yaml
    m = yaml.safe_load((config_dir / "meters.yaml").read_text(encoding="utf-8"))
    return [x for x in m["meters"] if x["role"] != "bus_virtual"]


# --------------------------------------------------------------------------- grid (year cache)

@lru_cache(maxsize=2)
def _state_daily(path: Path, mtime: float, names: tuple[str, ...]) -> dict:
    import pandas as pd
    s = pd.read_parquet(path)
    s["day"] = s.timestamp.dt.normalize()
    g = s.groupby("day")
    head_mw = g.p_head_mw.max()
    out = {
        "days": [d.strftime("%Y-%m-%d") for d in head_mw.index],
        "head_peak_mw": _r(head_mw),
        "head_energy_mwh": _r(g.p_head_mw.sum() / 4.0),
        "losses_mwh": _r(g.losses_mw.sum() / 4.0),
        "v_min_lv_pu": _r(g.v_min_lv_pu.min(), 4),
        "v_max_lv_pu": _r(g.v_max_lv_pu.max(), 4),
        "vuf_max_11kv_pct": _r(g.vuf_max_11kv_pct.max()),
        "oltc_tap_moves": [int(x) for x in g.tap_moved.sum()],
        "dt_load_pct_max": {},
    }
    for dt in names:
        cols = [c for c in s.columns if c.startswith(f"load_pct_{dt}_")]
        if cols:
            out["dt_load_pct_max"][dt] = _r(s[cols].max(axis=1).groupby(s.day).max(), 1)
    return out


def grid_daily(results: Path, config_dir: Path, data_dir: Path) -> dict:
    p = results / "year_2024" / "state.parquet"
    if not p.exists():
        return missing(p, "python scripts/run_year.py")
    out = {"available": True, "source": "results/year_2024/state.parquet",
           "note": "simulated year 2024 (power flow every 15 min); daily max / sum",
           **_state_daily(p, _mtime(p), tuple(dt_names(config_dir)))}
    amb = ambient_daily(data_dir)
    out["ambient_max_c"] = amb["max_c"] if amb else None
    return out


@lru_cache(maxsize=2)
def _ambient(path: Path, mtime: float) -> tuple:
    import pandas as pd
    e = pd.read_parquet(path, columns=["timestamp", "t2m_c"]).set_index("timestamp").t2m_c
    return e.index, e.to_numpy(float)


def ambient_daily(data_dir: Path) -> Optional[dict]:
    p = data_dir / "era5" / "era5_t2m_delhi_2024_ist_15min.parquet"
    if not p.exists():
        return None
    idx, v = _ambient(p, _mtime(p))
    d = v[: len(v) // STEPS_PER_DAY * STEPS_PER_DAY].reshape(-1, STEPS_PER_DAY)
    return {"max_c": _r(d.max(1), 1), "min_c": _r(d.min(1), 1)}


def ambient_day(data_dir: Path, day_index: int) -> Optional[list]:
    p = data_dir / "era5" / "era5_t2m_delhi_2024_ist_15min.parquet"
    if not p.exists():
        return None
    _, v = _ambient(p, _mtime(p))
    return _r(v[day_index * STEPS_PER_DAY:(day_index + 1) * STEPS_PER_DAY], 2)


def grid_day(results: Path, config_dir: Path, data_dir: Path, day: str) -> dict:
    """15-min feeder head power, voltage band and per-DT loading for one day."""
    import pandas as pd
    p = results / "year_2024" / "state.parquet"
    if not p.exists():
        return missing(p, "python scripts/run_year.py")
    s = pd.read_parquet(p)
    d0 = pd.Timestamp(day)
    s = s[(s.timestamp >= d0) & (s.timestamp < d0 + pd.Timedelta(days=1))]
    if s.empty:
        return {"available": False, "how": f"no simulated steps on {day} (2024 only)"}
    dt_load = {}
    for dt in dt_names(config_dir):
        cols = [c for c in s.columns if c.startswith(f"load_pct_{dt}_")]
        dt_load[dt] = _r(s[cols].max(axis=1), 1)
    day_index = (d0 - pd.Timestamp("2024-01-01")).days
    return {"available": True, "day": day, "times": [t.strftime("%H:%M") for t in s.timestamp],
            "p_head_mw": _r(s.p_head_mw), "losses_mw": _r(s.losses_mw, 4),
            "v_min_lv_pu": _r(s.v_min_lv_pu, 4), "v_max_lv_pu": _r(s.v_max_lv_pu, 4),
            "oltc_tap": [int(x) for x in s.oltc_tap], "dt_load_pct_max_phase": dt_load,
            "ambient_c": ambient_day(data_dir, day_index)}


# --------------------------------------------------------------------------- power quality

@lru_cache(maxsize=2)
def _vuf_daily(path: Path, mtime: float, ids: tuple[str, ...]) -> dict:
    import pandas as pd
    from src.analytics.unbalance import vuf_pct_from_ll

    f = pd.read_parquet(path, columns=["timestamp", "load_id", "phase", "voltage_v", "voltage_angle_deg"],
                        filters=[("load_id", "in", list(ids))])
    f = f.sort_values(["load_id", "timestamp", "phase"], kind="stable").reset_index(drop=True)
    if len(f) % 3 or not (f.phase.to_numpy().reshape(-1, 3) == np.array(["a", "b", "c"])).all():
        raise ValueError("meter frame: expected phases a, b, c at every (load_id, timestamp)")
    ph = (f.voltage_v.to_numpy() * np.exp(1j * np.radians(f.voltage_angle_deg.to_numpy()))).reshape(-1, 3)
    va, vb, vc = ph[:, 0], ph[:, 1], ph[:, 2]
    vu = vuf_pct_from_ll(np.abs(va - vb), np.abs(vb - vc), np.abs(vc - va))
    first = f.iloc[::3]
    t = pd.DataFrame({"load_id": first.load_id.to_numpy(), "timestamp": first.timestamp.to_numpy(), "vu": vu})
    t["day"] = t.timestamp.dt.normalize()
    g = t.groupby(["load_id", "day"]).vu
    mx, p95 = g.max().unstack(0), g.quantile(0.95).unstack(0)
    return {"days": [d.strftime("%Y-%m-%d") for d in mx.index],
            "max_pct": {k: _r(mx[k]) for k in ids if k in mx},
            "p95_pct": {k: _r(p95[k]) for k in ids if k in p95}}


def vuf_daily(results: Path, config_dir: Path) -> dict:
    p = results / "year_2024" / "meter_frame_metered.parquet"
    if not p.exists():
        return missing(p, "python scripts/run_year.py")
    meters = physical_meters(config_dir)
    ids = tuple(m["load_id"] for m in meters)
    return {"available": True, "source": "results/year_2024/meter_frame_metered.parquet",
            "method": "VU = V_N / V_P from line-to-line magnitudes (src/analytics/unbalance.vuf_pct_from_ll), "
                      "per 15-min metered reading; daily max and daily 95th percentile",
            "limit_pct": VU_LIMIT_PCT,
            "meters": [{"meter_id": m["meter_id"], "load_id": m["load_id"], "role": m["role"]} for m in meters],
            **_vuf_daily(p, _mtime(p), ids)}


# --------------------------------------------------------------------------- fault bursts

@lru_cache(maxsize=2)
def _burst_index(path: Path, mtime: float) -> dict[int, list[int]]:
    z = np.load(path)
    idx = json.loads(str(z["index"]))
    by_event: dict[int, list[int]] = {}
    for k, rec in enumerate(idx):
        by_event.setdefault(int(rec["event_id"]), []).append(k)
    return {"records": idx, "by_event": by_event}


def fault_burst(results: Path, event_id: int) -> dict:
    """Half-cycle RMS voltage and current per DT meter and phase for one fault
    of the fault set (``event_records.npz``, written by run_faults.py)."""
    p = results / "faults_2024" / "event_records.npz"
    if not p.exists():
        return missing(p, "python scripts/run_faults.py")
    bi = _burst_index(p, _mtime(p))
    ks = bi["by_event"].get(int(event_id))
    if not ks:
        return {"available": False, "how": f"no burst records for fault event {event_id}"}
    z = np.load(p)
    meters = []
    for k in ks:
        rec = bi["records"][k]
        v, i = z[f"v{k}"], z[f"i{k}"]
        base = float(rec.get("v_base_ln_v") or 1.0)
        meters.append({"load_id": rec["load_id"], "t_start": rec["t_start"], "phases": rec.get("phases"),
                       "v_pu": {ph: _r(v[:, j] / base, 4) for j, ph in enumerate("abc")},
                       "i_a": {ph: _r(i[:, j], 1) for j, ph in enumerate("abc")}})
    n = max(len(m["v_pu"]["a"]) for m in meters)
    return {"available": True, "event_id": int(event_id), "source": "results/faults_2024/event_records.npz",
            "t_ms": [round(k * HALF_CYCLE_MS, 1) for k in range(n)], "meters": meters,
            "note": "half-cycle RMS (10 ms at 50 Hz); voltage per unit of the meter's v_base_ln_v"}


# --------------------------------------------------------------------------- theft (injection 2)

def ntl_dir(results: Path) -> Path:
    """Injection 2 (Stage 4 fix) when its summary exists, else the v1 theft run."""
    d = results / "layer3" / "ntl_inj2"
    return d if (d / "summary.json").exists() else results / "layer3" / "ntl"


def year_dir(results: Path) -> Path:
    return results / ("year_2024_inj2" if ntl_dir(results).name == "ntl_inj2" else "year_2024")


@lru_cache(maxsize=2)
def _true_theft_dt_day(labels: Path, lm: float, hh: Path, hm: float) -> "object":
    import pandas as pd
    h = pd.read_parquet(hh, columns=["household", "dt", "attack"])
    h = h[h.attack.isin(ENERGY_THEFT_FOLDS)]
    lab = pd.read_parquet(labels, columns=["household", "date", "theft_kwh"],
                          filters=[("household", "in", h.household.tolist())])
    lab = lab.merge(h[["household", "dt"]], on="household")
    return lab.groupby(["date", "dt"]).theft_kwh.sum().unstack("dt")


def theft_dt_daily(results: Path) -> dict:
    """Engine daily NTL estimate per DT (lambda = 0, netted) with the simulation's
    true energy theft (h1-h4) per DT-day beside it."""
    import pandas as pd
    nd = ntl_dir(results)
    p = nd / "dt_day.parquet"
    if not p.exists():
        return missing(p, "python scripts/run_layer3_ntl_inj2.py balance" if nd.name == "ntl_inj2"
                       else "python scripts/run_layer3_ntl.py balance")
    df = pd.read_parquet(p)
    days = [d.strftime("%Y-%m-%d") for d in df.index]
    dts = list(df["ntl_dt_lambda_0"].columns)
    out = {"available": True, "source": str(p.relative_to(results.parent)), "injection": nd.name,
           "days": days, "dts": dts,
           "estimate_kwh": {dt: _r(df["ntl_dt_lambda_0"][dt], 2) for dt in dts},
           "b_unnetted_kwh": {dt: _r(df["B"][dt], 2) for dt in dts},
           "billed_kwh": {dt: _r(df["billed_dt"][dt], 1) for dt in dts},
           "offline_est_kwh": {dt: _r(df["eoff_dt"][dt], 2) for dt in dts},
           "truth_available": False, "true_theft_kwh": None,
           "test_window": ["2024-07-01", "2024-12-31"]}
    yd = year_dir(results)
    lp, hp = yd / "household_daily_labels.parquet", yd / "households.parquet"
    if lp.exists() and hp.exists():
        t = _true_theft_dt_day(lp, _mtime(lp), hp, _mtime(hp)).reindex(index=df.index, columns=dts).fillna(0.0)
        out["truth_available"] = True
        out["true_theft_kwh"] = {dt: _r(t[dt], 2) for dt in dts}
        out["truth_note"] = "simulation ground truth: energy theft h1-h4 (theft_kwh), summed per DT-day"
    return out


@lru_cache(maxsize=2)
def _jokar_rank(path: Path, mtime: float, variant: str, qs: tuple) -> "object":
    """Per household and energy-theft fold: score = share of online test days
    flagged (``ntl_engine.household_score``); flagged when score >= the fold's
    frozen q (the rule scored in ``summary.json``)."""
    import pandas as pd
    z = np.load(path)
    cols = {"household": z["ids"]}
    folds_flagging = np.zeros(len(z["ids"]), dtype=np.int32)
    scores = []
    for f, q in qs:
        fl = z[f"flag__{variant}__{f}"]
        on = (fl >= 0).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sc = np.where((fl[:, 0] != -2) & (on > 0), (fl == 1).sum(1) / np.maximum(on, 1), np.nan)
        cols[f"score_{f}"] = sc
        folds_flagging += (sc >= q)
        scores.append(sc)
    cols["folds_flagging"] = folds_flagging
    with np.errstate(all="ignore"):
        cols["mean_score"] = np.nanmean(np.vstack(scores), axis=0)
    return pd.DataFrame(cols)


def theft_inspection(results: Path, dt: Optional[str] = None, limit: int = 25) -> dict:
    """Jokar inspection ranking (Jul-Dec test window), primary variant: households
    ordered by how many energy-theft folds (h1-h4) flag them, then by mean score."""
    import pandas as pd
    from .results import load
    nd = ntl_dir(results)
    if nd.name != "ntl_inj2":
        return {"available": False, "how": "inspection ranking needs injection 2 (results/layer3/ntl_inj2/)"}
    p = nd / "jokar_test.npz"
    if not p.exists():
        return missing(p, "python scripts/run_layer3_ntl_inj2.py jokar-test")
    s = load(nd / "summary.json")["jokar"]
    variant, gate = "update", s.get("primary_gate", "ungated")
    if gate != "ungated":
        return {"available": False, "how": "gated ranking needs the transformer gate per DT-day; not built"}
    qs = tuple((f, float(s[variant][gate]["per_fold"][f]["q"])) for f in ENERGY_THEFT_FOLDS)
    r = _jokar_rank(p, _mtime(p), variant, qs)
    hp = year_dir(results) / "households.parquet"
    truth = hp.exists()
    h = (pd.read_parquet(hp, columns=["household", "dt", "thief", "attack"]) if truth
         else pd.read_parquet(results / "layer3" / "ntl" / "consumer_index.parquet")[["household", "dt"]])
    r = r.merge(h, on="household", how="left")
    scored = r[r.mean_score.notna()]
    agg = {"households_scored": ("household", "size"),
           "flagged_by_any_fold": ("folds_flagging", lambda x: int((x > 0).sum())),
           "flagged_by_all_folds": ("folds_flagging", lambda x: int((x == len(qs)).sum()))}
    if truth:
        agg["true_energy_thieves"] = ("attack", lambda x: int(x.isin(ENERGY_THEFT_FOLDS).sum()))
    per_dt = scored.groupby("dt").agg(**agg)
    sel = scored if dt is None else scored[scored.dt == dt]
    sel = sel.sort_values(["folds_flagging", "mean_score"], ascending=False).head(limit)
    rows = []
    for x in sel.itertuples():
        row = {"household": x.household, "dt": x.dt, "folds_flagging": int(x.folds_flagging),
               "mean_score": round(float(x.mean_score), 3),
               "scores": {f: (None if np.isnan(getattr(x, f"score_{f}")) else round(float(getattr(x, f"score_{f}")), 3))
                          for f, _ in qs}}
        if truth:
            row["true_attack"] = x.attack or None
        rows.append(row)
    return {"available": True, "source": str(p.relative_to(results.parent)), "variant": variant, "gate": gate,
            "folds": list(ENERGY_THEFT_FOLDS), "q": dict(qs), "window": "test window Jul-Dec 2024 (184 days)",
            "rule": "per fold: score = share of online test days flagged; flagged when score >= q (frozen)",
            "truth_available": truth,
            "truth_note": "true_attack is simulation ground truth (households.parquet), shown for review only",
            "honesty": "Macro precision about 2 %: this is an order to inspect in, not a list of thieves.",
            "per_dt": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in per_dt.to_dict(orient="index").items()},
            "rows": rows}


# --------------------------------------------------------------------------- thermal

@lru_cache(maxsize=2)
def _thermal_ref(path: Path, mtime: float) -> dict:
    z = np.load(path)
    return {k: z[k] for k in ("days", "theta_h_max", "theta_h_end", "theta_o", "lol_h")}


def thermal_daily(results: Path, config_dir: Path, data_dir: Path) -> dict:
    p = results / "layer3" / "thermal" / "reference_main.npz"
    if not p.exists():
        return missing(p, "python scripts/run_layer3_thermal.py reference")
    z = _thermal_ref(p, _mtime(p))
    names = dt_names(config_dir)
    hmax = z["theta_h_max"].reshape(-1, STEPS_PER_DAY, len(names)).max(1)
    omax = z["theta_o"].reshape(-1, STEPS_PER_DAY, len(names)).max(1)
    lol = z["lol_h"].reshape(-1, STEPS_PER_DAY, len(names)).sum(1)
    amb = ambient_daily(data_dir)
    return {"available": True, "source": "results/layer3/thermal/reference_main.npz",
            "note": "IS 2026-7 reference trajectories (metered load, ERA5 ambient); daily max",
            "days": [str(d) for d in z["days"]], "dts": names,
            "theta_h_max_c": {dt: _r(hmax[:, k], 1) for k, dt in enumerate(names)},
            "theta_o_max_c": {dt: _r(omax[:, k], 1) for k, dt in enumerate(names)},
            "loss_of_life_h": {dt: _r(lol[:, k], 4) for k, dt in enumerate(names)},
            "ambient_max_c": amb["max_c"] if amb else None,
            "thresholds_c": [98, 120, 140]}


@lru_cache(maxsize=2)
def _thermal_fc(path: Path, mtime: float) -> dict:
    z = np.load(path)
    return {"origins": z["origins"], **{m: z[f"{m}__theta_h_end"] for m in ("persistence", "seasonal_naive", "lstm")}}


def thermal_day(results: Path, config_dir: Path, data_dir: Path, day: str) -> dict:
    """15-min hot-spot and top-oil per DT for one day; the day-ahead forecast
    hot-spot trajectories beside the reference when the day is a test origin."""
    import pandas as pd
    p = results / "layer3" / "thermal" / "reference_main.npz"
    if not p.exists():
        return missing(p, "python scripts/run_layer3_thermal.py reference")
    z = _thermal_ref(p, _mtime(p))
    names = dt_names(config_dir)
    days = [str(d) for d in z["days"]]
    if day not in days:
        return {"available": False, "how": f"{day} is outside the simulated year"}
    k = days.index(day)
    sl = slice(k * STEPS_PER_DAY, (k + 1) * STEPS_PER_DAY)
    times = [t.strftime("%H:%M") for t in pd.date_range(day, periods=STEPS_PER_DAY, freq="15min")]
    out = {"available": True, "day": day, "times": times, "dts": names,
           "theta_h_c": {dt: _r(z["theta_h_end"][sl, j], 2) for j, dt in enumerate(names)},
           "theta_o_c": {dt: _r(z["theta_o"][sl, j], 2) for j, dt in enumerate(names)},
           "ambient_c": ambient_day(data_dir, k), "forecast": None}
    fp = results / "layer3" / "thermal" / "trajectories_test_main.npz"
    if fp.exists():
        fc = _thermal_fc(fp, _mtime(fp))
        hit = np.flatnonzero(fc["origins"] == k)
        if hit.size:
            o = int(hit[0])
            out["forecast"] = {m: {dt: _r(fc[m][o, :, j], 2) for j, dt in enumerate(names)}
                               for m in ("persistence", "seasonal_naive", "lstm")}
            out["forecast_note"] = ("day-ahead K_p forecast -> IS 2026-7 hot-spot, from the reference state at the "
                                    "end of the previous day (test origins May-Dec)")
        out["forecast_days"] = [days[int(i)] for i in fc["origins"]]
    return out
