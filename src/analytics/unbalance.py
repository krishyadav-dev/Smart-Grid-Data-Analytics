"""IS 17036 voltage unbalance from line-to-line magnitudes (Layer 3 P1, F6).

VU = V_N / V_P x 100 %, IS 17036:2018 Cl. 3.33 Note 2 and Table 8 (PDF pp. 6-7;
limit 2 % for 95 % of each week, Cl. 4.2.4, Table 8, PDF p. 7).  The telemetry
carries line-to-line magnitudes only, so the ratio is computed from them with

    beta = (a^4 + b^4 + c^4) / (a^2 + b^2 + c^2)^2
    VU   = 100 sqrt((1 - sqrt(3 - 6 beta)) / (1 + sqrt(3 - 6 beta)))

This is exact (not an approximation) for line-to-line voltages, because they
sum to zero and so carry no zero-sequence component: their magnitudes fix the
ratio of negative- to positive-sequence magnitude.  3 - 6 beta is clipped at 0
before the square root (rounding can push it marginally below 0 for a fully
unbalanced set) and at 1 (rounding for a balanced set, where VU = 0).
"""

from __future__ import annotations

import numpy as np

VU_LIMIT_PCT = 2.0            # IS 17036:2018 Cl. 4.2.4, Table 8, PDF p. 7
WEEK_SHARE = 0.95             # Table 8: 95 % of each week's 10-min (here 15-min) values


def vuf_pct_from_ll(v_ab, v_bc, v_ca):
    """VU = V_N / V_P x 100 % from line-to-line RMS magnitudes (scalars or arrays)."""
    a2, b2, c2 = (np.asarray(x, float) ** 2 for x in (v_ab, v_bc, v_ca))
    beta = (a2 ** 2 + b2 ** 2 + c2 ** 2) / (a2 + b2 + c2) ** 2
    # beta >= 1/3 exactly, so 3 - 6 beta <= 1; the upper clip only absorbs rounding
    root = np.sqrt(np.clip(3.0 - 6.0 * beta, 0.0, 1.0))
    vu = 100.0 * np.sqrt((1.0 - root) / (1.0 + root))
    return float(vu) if np.ndim(vu) == 0 else vu
