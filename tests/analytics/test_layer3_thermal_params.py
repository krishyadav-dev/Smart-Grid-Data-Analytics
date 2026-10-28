"""config/layer3_thermal_params.yaml (Stage 3 run sheet §4, step C): the LSTM
size chosen on April 2024 origins and frozen with the training settings and seeds."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.analytics import forecast as F
from src.analytics import thermal as T

PARAMS = Path(__file__).resolve().parents[2] / "config" / "layer3_thermal_params.yaml"


def test_frozen_thermal_settings():
    d = yaml.safe_load(PARAMS.read_text(encoding="utf-8"))
    lstm, tr = d["lstm"], d["training"]
    assert lstm["variant"] in F.VARIANTS
    assert lstm["layers"] in F.GRID_LAYERS and lstm["units"] in F.GRID_UNITS
    assert lstm["dropout"] == F.DROPOUT
    assert (lstm["lookback_steps"], lstm["horizon_steps"]) == (F.LOOKBACK, F.HORIZON)
    assert (tr["lr"], tr["batch"], tr["max_epochs"], tr["patience"]) == (F.LR, F.BATCH, F.MAX_EPOCHS, F.PATIENCE)
    assert (tr["val_days"], tr["gap_days"], tr["data_start"]) == (F.VAL_DAYS, F.GAP_DAYS, "2024-01-02")
    assert d["seeds"] == list(F.SEEDS)
    assert d["grid"] == {"variants": list(F.VARIANTS), "layers": list(F.GRID_LAYERS), "units": list(F.GRID_UNITS)}
    assert "2024-04-01..2024-04-30" in d["tuned_on"]["window"]
    assert "2024-05-01..2024-12-30" in d["walk_forward"]["test_origins"]
    assert (d["thermal"]["dt_min"], d["thermal"]["substeps"]) == (T.DT_MIN, T.SUBSTEPS)
    assert F.n_parameters(F.build_model(1 + 2 + 7 + 13 + 3, lstm["layers"], lstm["units"])) == lstm["n_parameters"]
