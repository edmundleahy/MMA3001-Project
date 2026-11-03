"""Verification tests for models, metrics and the select/final workflow.

All data here is synthetic and is used only to check that the code behaves as
specified. None of it is project evaluation data.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    bootstrap_mae_gain,
    compare,
    errors_by_hour,
    run_final,
    run_select,
    select_n_lags,
)
from src.models import (
    LaggedLinearRegression,
    PersistenceForecaster,
    bias,
    condition_numbers,
    flop_estimate,
    mae,
    mae_skill,
    rmse,
)

T0 = pd.Timestamp("2024-03-01 00:00", tz="UTC")
SENSOR = "TEST"


# ---------- metrics, checked against hand calculations ----------

def test_metrics_match_hand_calculation():
    y, p = [1.0, 2.0, 4.0], [2.0, 2.0, 1.0]  # errors +1, 0, -3
    assert mae(y, p) == pytest.approx(4 / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt(10 / 3))
    assert bias(y, p) == pytest.approx(-2 / 3)


def test_skill_matches_hand_calculation():
    assert mae_skill(0.5, 1.0) == pytest.approx(0.5)
    assert mae_skill(1.5, 1.0) == pytest.approx(-0.5)
    with pytest.raises(ValueError):
        mae_skill(1.0, 0.0)


# ---------- models ----------

def test_persistence_returns_latest_observation():
    X = pd.DataFrame({"lag_0": [20.0, 21.5], "lag_1": [99.0, 99.0]})
    assert PersistenceForecaster().fit(X, None).predict(X).tolist() == [20.0, 21.5]


def noisy_frame(n=400, seed=1):
    """Random lags with a known linear target plus noise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(20, 2, (n, 3)), columns=["lag_0", "lag_1", "lag_2"])
    y = 1.5 + 0.7 * X["lag_0"] + 0.2 * X["lag_1"] - 0.1 * X["lag_2"]
    return X, y, y + rng.normal(0, 0.05, n)


def test_regression_recovers_known_coefficients():
    X, y, _ = noisy_frame()
    c = LaggedLinearRegression(3).fit(X, y).coefficients()
    assert c["intercept"] == pytest.approx(1.5)
    assert [c["lag_0"], c["lag_1"], c["lag_2"]] == pytest.approx([0.7, 0.2, -0.1])


def test_regression_matches_independent_least_squares_solution():
    X, _, noisy = noisy_frame()
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    expected = np.linalg.lstsq(A, noisy.to_numpy(), rcond=None)[0]
    c = LaggedLinearRegression(3).fit(X, noisy).coefficients()
    assert list(c.values()) == pytest.approx(expected.tolist())


def test_regression_uses_only_requested_lags():
    X, y, _ = noisy_frame()
    a = LaggedLinearRegression(1).fit(X, y).predict(X)
    b = LaggedLinearRegression(1).fit(X.assign(lag_2=0.0), y).predict(X.assign(lag_2=0.0))
    assert a.tolist() == b.tolist()


def test_regression_rejects_zero_lags():
    with pytest.raises(ValueError):
        LaggedLinearRegression(0)


def test_condition_number_detects_collinear_inputs():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    independent = np.column_stack([a, rng.normal(size=500)])
    collinear = np.column_stack([a, a + rng.normal(0, 1e-3, 500)])
    assert condition_numbers(collinear)["standardised"] > 100 * condition_numbers(independent)["standardised"]


def test_condition_number_handles_constant_column():
    X = np.column_stack([np.arange(5.0), np.ones(5)])
    assert np.isnan(condition_numbers(X)["standardised"])


def test_flop_estimate_matches_formula():
    f = flop_estimate(1000, 3)
    assert f["fit_flops"] == pytest.approx(2 * 1000 * 9 - 18)
    assert f["predict_flops_per_sample"] == 6


# ---------- selection and uncertainty ----------

def test_select_takes_fewest_lags_within_tolerance_of_best():
    t = pd.DataFrame({"n_lags": [1, 3, 6, 12], "regression_mae": [0.40, 0.349, 0.349, 0.348]})
    assert select_n_lags(t, tolerance=0.01) == 3


def test_select_keeps_more_lags_when_the_gain_exceeds_tolerance():
    t = pd.DataFrame({"n_lags": [1, 3, 6], "regression_mae": [0.40, 0.35, 0.30]})
    assert select_n_lags(t, tolerance=0.01) == 6


def test_select_ignores_missing_scores():
    t = pd.DataFrame({"n_lags": [1, 3], "regression_mae": [float("nan"), 0.35]})
    assert select_n_lags(t, tolerance=0.01) == 3


def test_bootstrap_is_repeatable_and_brackets_clear_gain():
    index = pd.date_range(T0, periods=24 * 20, freq="60min")
    rng = np.random.default_rng(0)
    y = np.zeros(len(index))
    worse, better = rng.normal(0, 1.0, len(index)), rng.normal(0, 0.2, len(index))
    a = bootstrap_mae_gain(index, y, worse, better)
    assert a == bootstrap_mae_gain(index, y, worse, better)
    assert a["ci95_low"] > 0 and a["ci95_low"] < a["mae_gain"] < a["ci95_high"]


def test_bootstrap_interval_spans_zero_for_identical_models():
    index = pd.date_range(T0, periods=24 * 5, freq="60min")
    pred = np.ones(len(index))
    out = bootstrap_mae_gain(index, np.zeros(len(index)), pred, pred)
    assert out["mae_gain"] == out["ci95_low"] == out["ci95_high"] == 0.0


def test_errors_by_hour_uses_melbourne_local_time():
    # 2024-07-15 00:00 UTC is 10:00 AEST; 2024-01-15 00:00 UTC is 11:00 AEDT.
    times = pd.DatetimeIndex(["2024-07-15 00:00", "2024-01-15 00:00"], tz="UTC")
    t = errors_by_hour(times, [0.0, 0.0], [1.0, 3.0], [2.0, 4.0])
    assert t["hour_local"].tolist() == [10, 11]
    assert t["persistence_mae"].tolist() == [1.0, 3.0]


# ---------- workflow ----------

def synthetic_csv(path, shift_after=None):
    """Write a daily-cycle series, optionally rewriting values after a given day."""
    index = pd.date_range(T0, periods=144 * 40, freq="10min")
    rng = np.random.default_rng(42)
    day = (index - T0) / pd.Timedelta(days=1)
    values = 22 + 2 * np.sin(2 * np.pi * day) + rng.normal(0, 0.05, len(index))
    if shift_after is not None:
        values = np.where(day >= shift_after, values + 50.0, values)
    pd.DataFrame({"sensorid": SENSOR, "variable": "Temperature",
                  "epoch_s": (index.astype("int64") // 10**9), "value": values}).to_csv(path, index=False)
    return path


def read_tables(folder):
    return {p.name: p.read_text(encoding="utf-8") for p in folder.glob("validation_*.csv")}


def test_compare_reports_empty_period_instead_of_crashing():
    train = pd.DataFrame({"lag_0": [1.0, 2.0, 3.0], "target": [2.0, 3.0, 4.0]})
    row = compare(train, train.iloc[0:0], 1)
    assert row["n_examples"] == 0
    assert np.isnan(row["regression_mae"]) and np.isnan(row["persistence_mae"])


def test_select_is_repeatable(tmp_path):
    csv = synthetic_csv(tmp_path / "r.csv")
    a = run_select(csv, SENSOR, tmp_path / "a", make_plots=False)
    b = run_select(csv, SENSOR, tmp_path / "b", make_plots=False)
    assert a == b
    assert read_tables(tmp_path / "a") == read_tables(tmp_path / "b")


def test_select_never_uses_test_period_values(tmp_path):
    # The test period starts about 32 days in. Rewriting it by +50 °C must not
    # change the selection or any sensitivity table.
    clean = run_select(synthetic_csv(tmp_path / "c.csv"), SENSOR, tmp_path / "c", make_plots=False)
    moved = run_select(synthetic_csv(tmp_path / "m.csv", shift_after=33), SENSOR, tmp_path / "m", make_plots=False)
    assert clean == moved
    assert read_tables(tmp_path / "c") == read_tables(tmp_path / "m")


def test_final_refuses_second_run_unless_reproducing(tmp_path):
    csv = synthetic_csv(tmp_path / "r.csv")
    out = tmp_path / "o"
    run_select(csv, SENSOR, out, make_plots=False)
    first = run_final(csv, SENSOR, out, out / "selection.json", make_plots=False)
    with pytest.raises(FileExistsError):
        run_final(csv, SENSOR, out, out / "selection.json", make_plots=False)
    again = run_final(csv, SENSOR, out, out / "selection.json", reproduce=True, make_plots=False)
    assert first == again
    assert json.loads((out / "test_metrics.json").read_text(encoding="utf-8")) == first


def test_final_beats_persistence_on_predictable_synthetic_cycle(tmp_path):
    # A sanity check that the pipeline can find real structure. It says nothing
    # about how the models perform on the real sensor.
    csv = synthetic_csv(tmp_path / "r.csv")
    out = tmp_path / "o"
    run_select(csv, SENSOR, out, make_plots=False)
    result = run_final(csv, SENSOR, out, out / "selection.json", make_plots=False)
    assert result["regression"]["mae"] < result["persistence"]["mae"]


def test_plots_are_written(tmp_path):
    csv = synthetic_csv(tmp_path / "r.csv")
    out = tmp_path / "o"
    run_select(csv, SENSOR, out)
    run_final(csv, SENSOR, out, out / "selection.json")
    for name in ("series_splits", "lag_sensitivity", "horizon_sensitivity", "test_overlay",
                 "test_metrics", "error_by_hour", "residuals", "error_over_time"):
        assert (out / f"fig_{name}.png").stat().st_size > 0
