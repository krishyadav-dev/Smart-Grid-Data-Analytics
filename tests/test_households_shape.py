"""
eMARC household shape library and its use (brief v2 X.3.3 item 1).

(a) the library excludes the COVID lockdown dates (and uses Jan 2018 - Feb 2020);
(b) every shape has mean 1;
(c) the assignment is reproducible with the seed;
(d) sum of household load = DT load still holds exactly (households grouped by shape).
Plus the data-validity rule (zero-kWh days dropped) and the nearest-month fill.

Source: Prayas (Energy Group), eMARC, Harvard Dataverse doi:10.7910/DVN/YJ5SP1 (CC0 1.0).
Needs the git-ignored data in ``data/emarc/``.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_source import config, emarc
from src.data_source import simulation as S

pytestmark = pytest.mark.skipif(not (config.EMARC_DIR / emarc.BLOCKS_FILE).exists(),
                                reason="eMARC data not present in data/emarc/")


def test_sldc_reference_transcription():
    """data/delhi_sldc_2024_25_reference.csv (the one versioned data file): spot values
    checked against the SLDC Annual Report 2024-25 page images (pp. 67, 79, 80)."""
    ldc, pk = S.load_sldc_reference()
    assert len(ldc) == 73 and len(pk) == 12
    row = ldc.set_index("level_mw")
    assert row.at[8600, "hours"] == 0.25 and row.at[5000, "hours"] == 2838.00     # p. 79
    assert row.at[3000, "pct_time"] == 84.41 and row.at[1400, "hours"] == 8760.0  # p. 80
    jun = pk[pk.date == "2024-06-19"].iloc[0]                                     # p. 67
    assert (jun.time, jun.demand_met_mw, jun.unrestricted_mw) == ("15:05:55", 8656, 8713)
    assert pk[pk.date == "2025-01-01"].iloc[0].time == "11:44:08"


@pytest.fixture(scope="module")
def lib():
    return emarc.load_library()


def test_a_lockdown_excluded(lib):
    assert lib.lockdown_days_used == 0
    assert lib.first_date >= pd.Timestamp("2018-01-01")
    assert lib.last_date <= pd.Timestamp("2020-02-29") < emarc.LOCKDOWN[0]
    print(f"\neMARC library: {lib.info['households_in_library']} households, "
          f"{lib.info['profiles']} household-month profiles, {lib.info['complete_days']} days "
          f"{lib.info['first_date']}..{lib.info['last_date']}")


def test_b_every_shape_mean_one(lib):
    assert lib.profiles.shape[1] == 96
    assert np.allclose(lib.profiles.mean(axis=1), 1.0, atol=1e-12)
    assert (lib.profiles >= 0).all()
    assert (lib.profile_days >= emarc.MIN_DAYS).all()


def test_zero_day_rule_and_month_fill(lib):
    d = lib.info["zero_kwh_rule_dropped"]
    print(f"\nzero-kWh rule dropped {d['household_days_total']} household-days "
          f"{d['household_days_by_region']}, {d['household_months_total']} household-months "
          f"{d['household_months_by_region']}; peak/mean {lib.info['profile_peak_to_mean']}")
    assert d["household_days_total"] > 0
    # every household has a profile for every month; own month where available
    assert (lib.month_profile >= 0).all()
    own = lib.has_month
    rows, months = np.nonzero(own)
    assert (lib.profile_month[lib.month_profile[rows, months]] == months + 1).all()
    assert (lib.profile_household[lib.month_profile] == np.arange(lib.n_households)[:, None]).all()


def test_nearest_month_fill_rule():
    has = np.zeros((1, 12), bool)
    has[0, [0, 6]] = True                          # Jan and Jul only
    near = emarc._nearest_fill(has)[0]
    assert near[0] == 0 and near[6] == 6
    assert near[2] == 0 and near[4] == 6          # Mar -> Jan (2 away), May -> Jul (2 away)
    assert near[3] == 0                           # Apr: Jan and Jul both 3 away -> earlier (Jan)


def test_c_assignment_reproducible(lib):
    a = emarc.assign(1000, lib, config.rng("households.shape"))
    b = emarc.assign(1000, lib, config.rng("households.shape"))
    c = emarc.assign(1000, lib, config.rng("something.else"))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.min() >= 0 and a.max() < lib.n_households


@pytest.fixture(scope="module")
def pop():
    return S.build_population("2024-03-01", "2024-03-02 23:45")


def test_c_population_assignment_reproducible(pop):
    again = S.build_population("2024-03-01", "2024-03-01 23:45")
    assert np.array_equal(pop.hh.emarc_row, again.hh.emarc_row)
    assert np.array_equal(pop.hh.profile_id[:, 0], again.hh.profile_id[:, 0])
    # March uses each household's March profile, or its nearest month
    lib = emarc.load_library()
    assert np.array_equal(pop.hh.profile_id[:, 0], lib.month_profile[pop.hh.emarc_row, 2])


def test_d_sum_household_equals_dt_load(pop):
    hh = pop.hh
    f = pop.dt_loads(pd.Timestamp("2024-03-01"))
    ts = pd.date_range("2024-03-01", "2024-03-02 23:45", freq="15min")
    worst = 0.0
    for day in range(len(hh.days)):
        kw = hh.true_kw_day(day)
        kvar = kw * hh.qp_ratio[:, None]
        for s in range(96):
            p, q = f(day * 96 + s, ts[day * 96 + s])
            for k in range(hh.n_dt):
                for ph in range(3):
                    m = (hh.dt_index == k) & (hh.phase == ph)
                    exp_p, exp_q = kw[m, s].sum(), kvar[m, s].sum()
                    worst = max(worst, abs(p[k, ph] - exp_p) / max(exp_p, 1.0),
                                abs(q[k, ph] - exp_q) / max(abs(exp_q), 1.0))
    print(f"\nDT load vs sum of household load (grouped by shape): max rel {worst:.1e}")
    assert worst < 1e-12
