"""Pinned household counts (Layer 3 P3, user decision 2026-10-27).

config/household_counts.yaml keeps N = 8,020 from the pre-fix run; the P3
cleaning fix raises the pool-average peak by 0.008 %, which would change
buses 634, 671 and 675 by one household each.  Each pinned count must stay
within 1 of the rule N_bus = round(P_bus / pool-average peak).
"""

import json

import pytest

from src.data_source import config
from src.data_source import households as H

N_PINNED = 8020


def test_pinned_counts_sum():
    counts = H.pinned_household_counts()
    assert sum(counts.values()) == N_PINNED
    assert set(counts) == set(H.lv_phase_loads())


def test_check_pinned_counts_rejects_more_than_one():
    pinned = H.pinned_household_counts()
    assert H.check_pinned_counts(pinned, dict(pinned)) == {b: 0 for b in pinned}
    off = dict(pinned)
    off["671"] += 2
    with pytest.raises(ValueError, match="more than 1"):
        H.check_pinned_counts(pinned, off)


def test_pinned_counts_within_one_of_rule():
    """Against the pool-average peak of the cached run (skip without the cache)."""
    path = config.YEAR_CACHE_DIR / "run_info.json"
    if not path.exists():
        pytest.skip("year cache absent")
    info = json.loads(path.read_text(encoding="utf-8"))
    peak_kw = info["design_peak"]["pool_avg_peak_kw"]
    bus_p = {b: float(v[0].sum()) for b, v in H.lv_phase_loads().items()}
    rule = {b: int(round(p / peak_kw)) for b, p in bus_p.items()}
    pinned = H.pinned_household_counts()
    diff = {b: pinned[b] - rule[b] for b in rule}
    print(f"\npool-average peak {peak_kw:.7f} kW; pinned - rule: {diff}")
    assert all(abs(d) <= 1 for d in diff.values())
    assert sum(pinned.values()) == N_PINNED
