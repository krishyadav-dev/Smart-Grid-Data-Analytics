"""Stage A validation against Kersting's published power flow (Phase 2A).

Used by ``scripts/validate_stage_a.py`` and ``tests/test_stage_a.py``.  Every
published number comes from :mod:`ieee13_data`.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from . import ieee13_data as ieee
from .feeder_model import build_network, solve_with_load_models

PHASES = "abc"


def _sum_phases(df: pd.DataFrame, fmt: str) -> float:
    return float(sum(df[fmt.format(p)].sum() for p in PHASES))


def validate_stage_a() -> Dict[str, Any]:
    """Solve Stage A with its load models and compare with the published case.

    Returns a dict with ``totals`` (DataFrame), ``bus_avg`` (phase-average
    magnitude per bus), ``per_phase`` (magnitude error, error minus RG60's error
    on the same phase, angle error), ``capacitors`` (kvar per phase) and
    ``n_power_flows``.  Errors are model minus published.
    """
    net = build_network("A")
    n_pf = solve_with_load_models(net)
    res = net.res_bus_3ph
    bus_idx = dict(zip(net.bus.name, net.bus.index))

    # ---- totals ------------------------------------------------------------ #
    p_in = _sum_phases(net.res_ext_grid_3ph, "p_{}_mw") * 1e3
    q_in = _sum_phases(net.res_ext_grid_3ph, "q_{}_mvar") * 1e3
    loads = net.asymmetric_load[net.asymmetric_load.name.str.startswith("load_")]
    p_load = _sum_phases(loads, "p_{}_mw") * 1e3
    q_load = _sum_phases(loads, "q_{}_mvar") * 1e3
    rows = [
        ("input kW", p_in, ieee.PUBLISHED_INPUT_KW_TOTAL),
        ("input kvar", q_in, ieee.PUBLISHED_INPUT_KVAR_TOTAL),
        ("load kW", p_load, ieee.PUBLISHED_LOAD_KW_TOTAL),
        ("load kvar", q_load, ieee.PUBLISHED_LOAD_KVAR_TOTAL),
        ("losses kW", p_in - p_load, ieee.PUBLISHED_LOSS_KW_TOTAL),
    ]
    for ph, pub_p, pub_q in zip(PHASES, ieee.PUBLISHED_INPUT_KW, ieee.PUBLISHED_INPUT_KVAR):
        rows.append((f"input kW {ph}", float(net.res_ext_grid_3ph[f"p_{ph}_mw"].sum()) * 1e3, pub_p))
        rows.append((f"input kvar {ph}", float(net.res_ext_grid_3ph[f"q_{ph}_mvar"].sum()) * 1e3, pub_q))
    totals = pd.DataFrame(rows, columns=["quantity", "model", "published"])
    totals["error"] = totals.model - totals.published
    totals["error_pct"] = 100 * totals.error / totals.published.abs()

    # ---- voltages -------------------------------------------------------- #
    rg = bus_idx["RG60"]
    rg_err = {p: float(res.at[rg, f"vm_{p}_pu"])
              - ieee.PUBLISHED_VOLTAGES["RG60"]["abc".index(p)][0] for p in PHASES}
    per_phase, bus_avg = [], []
    for bus, pub in ieee.PUBLISHED_VOLTAGES.items():
        i = bus_idx[bus]
        mags, pubs = [], []
        for ph, pv in zip(PHASES, pub):
            if pv is None:
                continue
            vm = float(res.at[i, f"vm_{ph}_pu"])
            va = float(res.at[i, f"va_{ph}_degree"])
            err = vm - pv[0]
            d_ang = (va - pv[1] + 180.0) % 360.0 - 180.0
            # 650 is upstream of the regulator: no tap-averaging error to remove.
            rg = np.nan if bus == "650" else rg_err[ph]
            per_phase.append({
                "bus": bus, "phase": ph, "vm_model": vm, "vm_published": pv[0],
                "vm_error_pct": 100 * err / pv[0],
                "rg60_error_pct": 100 * rg / pv[0],
                "vm_error_minus_rg60_pct": 100 * (err - rg) / pv[0],
                "va_model": va, "va_published": pv[1], "va_error_deg": d_ang,
            })
            mags.append(vm)
            pubs.append(pv[0])
        bus_avg.append({
            "bus": bus, "phases": "".join(p for p, v in zip(PHASES, pub) if v),
            "avg_model": float(np.mean(mags)), "avg_published": float(np.mean(pubs)),
            "avg_error_pct": 100 * (np.mean(mags) / np.mean(pubs) - 1),
        })

    # ---- capacitors: actual kvar per phase (constant-Z) ------------------ #
    cap675 = ieee.CAPACITORS["675"]
    i675 = bus_idx["675"]
    cap611 = net.asymmetric_load[net.asymmetric_load.name == "cap_611_c"].iloc[0]
    caps = []
    for k, ph in enumerate(PHASES):
        q675 = cap675[k] * float(res.at[i675, f"vm_{ph}_pu"]) ** 2
        q611 = -float(cap611[f"q_{ph}_mvar"]) * 1e3
        caps.append({"phase": ph, "model": q675 + q611,
                     "published": ieee.PUBLISHED_CAP_KVAR[k]})
    capacitors = pd.DataFrame(caps)
    capacitors["error_pct"] = 100 * (capacitors.model / capacitors.published - 1)

    return {
        "totals": totals,
        "bus_avg": pd.DataFrame(bus_avg),
        "per_phase": pd.DataFrame(per_phase),
        "capacitors": capacitors,
        "n_power_flows": n_pf,
        "net": net,
    }
