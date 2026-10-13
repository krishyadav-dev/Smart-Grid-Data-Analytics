"""
Jokar h1-h6 (Part 2C.4): each attack reproduced on a hand-made daily vector.

Source: Jokar, Arianpoo & Leung, IEEE TSG 7(1) 2016, early-access version,
§VI items 1-6, PDF p. 7.  Parameters are hourly (held for the four 15-min
samples of the hour); h5 = daily mean; h6 = reversal.
"""

import numpy as np
import pytest

from src.data_source import theft_injector as T

SPH = 4                                    # 15-min samples per hour


@pytest.fixture
def day():
    """Hand-made day: hour h has kW = h + 1 in all four of its samples."""
    return np.repeat(np.arange(1.0, 25.0), SPH)


class _FixedRng:
    """Stand-in Generator returning chosen parameter values."""

    def __init__(self, uniform=None, integers=None):
        self._u, self._i = list(uniform or []), list(integers or [])

    def uniform(self, lo, hi, size=None):
        v = np.asarray(self._u.pop(0), float)
        return np.broadcast_to(v, size).copy() if size is not None else float(v)

    def integers(self, lo, hi, size=None, endpoint=False):
        return np.asarray(self._i.pop(0)).reshape(size)


def test_h1_scales_by_alpha(day):
    rep, par = T.h1(day, _FixedRng(uniform=[0.3]))
    assert np.allclose(rep[0], 0.3 * day)
    assert par["alpha"][0] == 0.3


def test_h2_zero_inside_off_window(day):
    # t_s = 5 h, duration 6 h -> hours 5..10 report 0, the rest unchanged
    rep, par = T.h2(day, _FixedRng(integers=[[5], [6]]))
    hour = np.repeat(np.arange(24), SPH)
    off = (hour >= 5) & (hour < 11)
    assert np.all(rep[0][off] == 0)
    assert np.allclose(rep[0][~off], day[~off])
    assert par["t_end_h"][0] == 11


def test_h2_window_clipped_at_midnight(day):
    rep, _ = T.h2(day, _FixedRng(integers=[[19], [24]]))
    hour = np.repeat(np.arange(24), SPH)
    assert np.all(rep[0][hour >= 19] == 0) and np.allclose(rep[0][hour < 19], day[hour < 19])


def test_h2_parameter_ranges():
    rng = np.random.default_rng(0)
    x = np.ones((5000, 96))
    _, par = T.h2(x, rng)
    assert par["t_start_h"].min() >= 0 and par["t_start_h"].max() <= 23 - T.MIN_OFF_TIME_H
    assert par["duration_h"].min() >= T.MIN_OFF_TIME_H and par["duration_h"].max() <= 24
    assert par["t_start_h"].max() == 23 - T.MIN_OFF_TIME_H     # endpoints reachable
    assert par["duration_h"].max() == 24


def test_h3_hourly_gamma(day):
    gamma = np.linspace(0.1, 0.8, 24)
    rep, _ = T.h3(day, _FixedRng(uniform=[gamma]))
    assert np.allclose(rep[0], day * np.repeat(gamma, SPH))


def test_h4_gamma_times_mean(day):
    gamma = np.linspace(0.8, 0.1, 24)
    rep, _ = T.h4(day, _FixedRng(uniform=[gamma]))
    assert np.allclose(rep[0], day.mean() * np.repeat(gamma, SPH))


def test_h5_daily_mean(day):
    rep, _ = T.h5(day)
    assert np.allclose(rep[0], 12.5)


def test_h6_reverses_the_day(day):
    rep, _ = T.h6(day)
    assert np.allclose(rep[0], day[::-1])
    assert rep[0][0] == 24.0 and rep[0][-1] == 1.0


@pytest.mark.parametrize("kind", T.ATTACKS)
def test_reported_energy_not_above_true(kind):
    """Daily energy: h1, h3, h4 report <= 0.8 of true; h2 drops a >= 4 h
    window; h5/h6 keep it (severity 0)."""
    rng = np.random.default_rng(1)
    x = rng.gamma(2.0, 0.3, size=(400, 96))
    rep, _ = T.apply_attack(kind, x, rng)
    e_true, e_rep = x.sum(axis=1), rep.sum(axis=1)
    assert np.all(e_rep <= e_true + 1e-9)
    if kind in ("h5", "h6"):
        assert np.allclose(e_rep, e_true)
        assert T.severity(e_true, e_rep) == pytest.approx(0.0, abs=1e-12)
    else:
        if kind != "h2":
            assert np.all(e_rep <= T.RANGE_HI * e_true + 1e-9)
        assert np.all(e_rep < e_true)
        assert T.severity(e_true, e_rep) > 0


def test_parameter_ranges_alpha_gamma():
    rng = np.random.default_rng(2)
    x = np.ones((3000, 96))
    _, p1 = T.h1(x, rng)
    _, p3 = T.h3(x, rng)
    for v in (p1["alpha"], p3["gamma"]):
        assert v.min() >= T.RANGE_LO and v.max() <= T.RANGE_HI


def test_severity_definition():
    assert T.severity([10.0, 10.0], [5.0, 7.0]) == pytest.approx(1 - 12 / 20)
    assert T.severity([0.0], [0.0]) == 0.0


@pytest.mark.parametrize("kind", T.ATTACKS)
def test_attack_class_load_shifting(kind, day):
    """Brief v2 X.3.1 item 4 / X.3.2 item 20: h5 and h6 keep the daily energy
    ("load-shifting"); h1-h4 reduce it ("energy-reducing")."""
    rep, _ = T.apply_attack(kind, day, np.random.default_rng(5))
    keeps = np.allclose(rep.sum(axis=-1), np.asarray(day).sum(axis=-1))
    assert (T.attack_class(kind) == "load-shifting") == (kind in T.LOAD_SHIFTING)
    assert keeps == (kind in T.LOAD_SHIFTING)


def test_unknown_attack_raises():
    with pytest.raises(ValueError, match="attack"):
        T.apply_attack("h7", np.ones(96), np.random.default_rng(0))
