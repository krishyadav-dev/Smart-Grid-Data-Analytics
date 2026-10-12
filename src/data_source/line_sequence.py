"""
Sequence parameters from published phase-impedance matrices (Phase 2A.1).

pandapower lines are symmetric three-phase elements described by positive- and
zero-sequence data.  The IEEE 13-node package gives Kron-reduced 3x3 phase
matrices (ohm/mile, uS/mile) instead.  This module reduces them to sequence
data with the Fortescue transform, restricted to the phases present:

    Zs = mean of the self terms of the phases present
    Zm = mean of the mutual terms between the phases present (0 if one phase)
    Z1 = Zs - Zm,   Z0 = Zs + 2 Zm

For a symmetric 3x3 matrix these are exactly the diagonal terms Z11 and Z00 of
A^-1 Z A.  The off-diagonal sequence terms (the coupling that an untransposed
line introduces) are dropped, and missing phases of 1- and 2-phase laterals
become phantom phases with the same self/mutual values.  For a single-phase
config Z1 = Z0 = Zii, so current on the real phase sees exactly the published
self impedance.  Limitation: audit L1-10 (phasing and coupling are lost).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

MILE_KM = 1.609344
_PHASE_INDEX = {"a": 0, "b": 1, "c": 2}


def upper_to_matrix(upper: Sequence[complex]) -> np.ndarray:
    """(aa, ab, ac, bb, bc, cc) upper triangle -> symmetric 3x3 matrix."""
    aa, ab, ac, bb, bc, cc = upper
    return np.array([[aa, ab, ac], [ab, bb, bc], [ac, bc, cc]], dtype=complex)


def self_mutual(m: np.ndarray, phases: str) -> tuple[complex, complex]:
    """Mean self and mean mutual term over the phases present."""
    idx = [_PHASE_INDEX[p] for p in phases]
    zs = np.mean([m[i, i] for i in idx])
    pairs = list(combinations(idx, 2))
    zm = np.mean([m[i, j] for i, j in pairs]) if pairs else 0j
    return complex(zs), complex(zm)


@dataclass(frozen=True)
class SequenceParams:
    """Per-km sequence data in pandapower's units."""
    r1_ohm_per_km: float
    x1_ohm_per_km: float
    r0_ohm_per_km: float
    x0_ohm_per_km: float
    c1_nf_per_km: float
    c0_nf_per_km: float


def sequence_params(z_upper: Sequence[complex], b_upper: Sequence[float],
                    phases: str, f_hz: float) -> SequenceParams:
    """Positive/zero-sequence R, X (ohm/km) and C (nF/km) of one line config.

    ``z_upper`` in ohm/mile and ``b_upper`` in uS/mile, as published.
    """
    zs, zm = self_mutual(upper_to_matrix(z_upper), phases)
    bs, bm = self_mutual(upper_to_matrix(b_upper), phases)
    z1, z0 = zs - zm, zs + 2 * zm
    b1, b0 = (bs - bm).real, (bs + 2 * bm).real
    us_per_mile_to_nf_per_km = 1e3 / (2 * math.pi * f_hz) / MILE_KM  # uS -> nF
    return SequenceParams(
        r1_ohm_per_km=z1.real / MILE_KM,
        x1_ohm_per_km=z1.imag / MILE_KM,
        r0_ohm_per_km=z0.real / MILE_KM,
        x0_ohm_per_km=z0.imag / MILE_KM,
        c1_nf_per_km=b1 * us_per_mile_to_nf_per_km,
        c0_nf_per_km=b0 * us_per_mile_to_nf_per_km,
    )
