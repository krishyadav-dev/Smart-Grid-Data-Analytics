"""
Step-voltage control with line-drop compensation (Phase 2B, step B7).

The IEEE regulator settings (``ieee13_data.REGULATOR``: voltage level 122 V,
bandwidth 2 V on a 120 V base with PT ratio 20, R 3 V / X 9 V, CT 700 A) are
applied to one controlled transformer tap:

- Set point 122/120 = 1.0167 pu, band +/- (2/2)/120 = 0.0083 pu.
- Line-drop compensator as an impedance: Z_LDC = (R + jX) x PT / CT ohm at
  4.16 kV.  Its per-unit value is kept when the feeder moves to 11 kV (the B2
  length scaling keeps per-unit feeder impedance), so Z = Z_4.16 (V/4.16)^2.
  X_LDC is scaled by f/60 like the line reactances in B3, so the compensator
  keeps matching the feeder it models.
- Compensated voltage per phase |V - Z_LDC I| from the solved voltage and the
  sending-end current of the monitored line; the three phases are averaged
  because pandapower has one tap per transformer (as in Stage A).
- At most ``max_moves`` tap steps per call; OLTC time delay is ignored at
  15-minute resolution.  Step 0.625 % and +/-16 steps are carried over from
  the IEEE regulator (``verify`` against a real 33/11 kV OLTC).
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np
import pandapower as pp
import pandas as pd

from . import ieee13_data as ieee

_REG = ieee.REGULATOR
SET_POINT_PU = _REG["voltage_level_v"][0] / 120.0                 # 1.0167
HALF_BAND_PU = _REG["bandwidth_v"] / 2.0 / 120.0                  # 0.0083


def ldc_impedance_ohm(v_kv: float, f_hz: float) -> complex:
    """Line-drop compensator impedance (ohm per phase) at ``v_kv`` and ``f_hz``.

    (R + jX) x PT / CT at 4.16 kV and 60 Hz, kept in per unit when the feeder
    voltage changes and with X scaled by f/60.
    """
    z_416 = complex(_REG["r_setting_v"][0], _REG["x_setting_v"][0]) \
        * _REG["pt_ratio"] / _REG["ct_primary_a"]
    scale = (v_kv / ieee.V_PRIMARY_KV) ** 2
    return complex(z_416.real * scale,
                   z_416.imag * scale * f_hz / ieee.FREQUENCY_HZ)


def attach_oltc(net: pp.pandapowerNet, trafo_name: str, line_name: str) -> None:
    """Mark ``trafo_name``'s tap as controlled, monitoring ``line_name``.

    The compensated voltage is taken at the line's sending end.  Stored as a
    one-row table ``net["oltc"]`` so it survives ``pp.to_json``.
    """
    t = int(net.trafo.index[net.trafo.name == trafo_name][0])
    ln = int(net.line.index[net.line.name == line_name][0])
    v_kv = float(net.bus.at[net.line.at[ln, "from_bus"], "vn_kv"])
    z = ldc_impedance_ohm(v_kv, float(net.f_hz))
    net["oltc"] = pd.DataFrame([{
        "trafo": t, "line": ln, "r_ldc_ohm": z.real, "x_ldc_ohm": z.imag,
        "set_point_pu": SET_POINT_PU, "half_band_pu": HALF_BAND_PU,
    }])


def has_oltc(net: pp.pandapowerNet) -> bool:
    return "oltc" in net and isinstance(net["oltc"], pd.DataFrame) and len(net["oltc"]) == 1


def relay_voltage(net: pp.pandapowerNet) -> Tuple[float, Tuple[float, float, float]]:
    """Average and per-phase compensated voltage (pu) from the solved net."""
    c = net["oltc"].iloc[0]
    ln = int(c.line)
    bus = int(net.line.at[ln, "from_bus"])
    v_base = float(net.bus.at[bus, "vn_kv"]) * 1e3 / math.sqrt(3)
    z = complex(c.r_ldc_ohm, c.x_ldc_ohm)
    rb, rl = net.res_bus_3ph.loc[bus], net.res_line_3ph.loc[ln]
    out = []
    for p in "abc":
        v = rb[f"vm_{p}_pu"] * np.exp(1j * np.deg2rad(rb[f"va_{p}_degree"])) * v_base
        s = complex(rl[f"p_{p}_from_mw"], rl[f"q_{p}_from_mvar"]) * 1e6
        i = np.conj(s / v)
        out.append(abs(v - z * i) / v_base)
    return float(np.mean(out)), tuple(out)


def regulate(net: pp.pandapowerNet, max_moves: int,
             solve: Callable[[pp.pandapowerNet], object] = pp.runpp_3ph
             ) -> Tuple[int, float]:
    """Move the controlled tap until the compensated voltage is in band.

    ``net`` must already be solved.  Moves at most ``max_moves`` steps,
    re-solving with ``solve`` after each move.  Returns (moves, V_relay).
    """
    c = net["oltc"].iloc[0]
    t = int(c.trafo)
    lo, hi = c.set_point_pu - c.half_band_pu, c.set_point_pu + c.half_band_pu
    t_min, t_max = net.trafo.at[t, "tap_min"], net.trafo.at[t, "tap_max"]
    moves = 0
    v, _ = relay_voltage(net)
    while moves < max_moves:
        tap = net.trafo.at[t, "tap_pos"]
        if v < lo and tap < t_max:
            net.trafo.at[t, "tap_pos"] = tap + 1
        elif v > hi and tap > t_min:
            net.trafo.at[t, "tap_pos"] = tap - 1
        else:
            break
        moves += 1
        solve(net)
        v, _ = relay_voltage(net)
    return moves, v


def settle(net: pp.pandapowerNet,
           solve: Callable[[pp.pandapowerNet], object] = pp.runpp_3ph,
           max_moves: int = 40) -> Tuple[int, float]:
    """Solve and regulate to steady state (build-time initialisation only)."""
    solve(net)
    return regulate(net, max_moves=max_moves, solve=solve)
