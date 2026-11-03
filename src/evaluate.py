"""Model selection on validation data and one final evaluation on test data.

The workflow has two deliberately separate stages:

* :func:`run_select` uses the training and validation periods only. It picks the
  lag count and runs the sensitivity studies. No test-period target or feature
  reaches any model or metric here.
* :func:`run_final` applies the locked settings to the test period. It refuses
  to run twice unless asked to reproduce an existing result, which is what makes
  "the test set was used once" checkable.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .data_processing import LOCAL_TZ
from .features import (
    Split,
    chronological_split,
    horizon_to_steps,
    lag_columns,
    load_temperature,
    make_supervised,
    to_grid,
)
from .models import (
    LaggedLinearRegression,
    PersistenceForecaster,
    bias,
    condition_numbers,
    flop_estimate,
    mae,
    mae_skill,
    rmse,
)

#: Lag counts compared on validation. All use the same rows, built with the
#: largest count, so differences come from the features and not the examples.
LAG_OPTIONS = (1, 3, 6, 12)
#: A lag count counts as equal to the best if its validation MAE is within this
#: many °C. It is far below the 0.25 °C median step between recorded values.
#: Set after seeing validation results only, and before any test-period scoring.
LAG_TOLERANCE_C = 0.01
#: Horizons compared on validation, in minutes.
HORIZON_OPTIONS = (20, 60, 180)
#: Grid steps compared on validation, in minutes (all divide the 60 min horizon).
STEP_OPTIONS = (10, 20, 30)
#: Resamples and seed for the bootstrap, fixed so results repeat exactly.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 3001
#: Days shown in the forecast overlay: the first week of the test period, fixed
#: in advance so the window cannot be cherry-picked.
OVERLAY_DAYS = 7


@dataclass
class Prepared:
    """A sensor's data, ready for modelling."""

    readings: pd.Series
    grid: pd.Series
    frame: pd.DataFrame
    split: Split
    horizon_steps: int


def prepare(readings_csv: str | Path, sensorid: str, step_min: int, horizon_min: int,
            max_lags: int) -> Prepared:
    """Load a sensor and build its grid, examples and chronological split."""
    readings = load_temperature(readings_csv, sensorid)
    return _prepare_from(readings, step_min, horizon_min, max_lags)


def _prepare_from(readings: pd.Series, step_min: int, horizon_min: int,
                  max_lags: int) -> Prepared:
    """Build the grid, examples and split from already-loaded readings."""
    grid = to_grid(readings, step_min)
    h = horizon_to_steps(horizon_min, step_min)
    frame = make_supervised(grid, max_lags, h)
    return Prepared(readings, grid, frame, chronological_split(frame, grid.index, max_lags, h), h)


def score(y_true, y_pred) -> dict[str, float]:
    """Return the MAE, RMSE and bias of one set of forecasts."""
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred),
            "bias": bias(y_true, y_pred)}


def compare(train: pd.DataFrame, part: pd.DataFrame, n_lags: int) -> dict:
    """Fit on ``train`` and score regression and persistence on ``part``.

    If either period has no complete examples, the metrics are NaN and
    ``n_examples`` shows why, so a sparse setting is reported rather than hidden.
    """
    if train.empty or part.empty:
        names = ("persistence_mae", "persistence_rmse", "regression_mae",
                 "regression_rmse", "mae_skill")
        return {"n_examples": len(part), **dict.fromkeys(names, float("nan"))}
    model = LaggedLinearRegression(n_lags).fit(train, train["target"])
    reg = score(part["target"], model.predict(part))
    per = score(part["target"], PersistenceForecaster().predict(part))
    return {"n_examples": len(part),
            "persistence_mae": per["mae"], "persistence_rmse": per["rmse"],
            "regression_mae": reg["mae"], "regression_rmse": reg["rmse"],
            "mae_skill": mae_skill(reg["mae"], per["mae"])}


def lag_sensitivity(prep: Prepared, lag_options=LAG_OPTIONS) -> pd.DataFrame:
    """Compare lag counts on the validation period, using identical rows."""
    train, val = prep.frame[prep.split.train], prep.frame[prep.split.val]
    return pd.DataFrame([
        {"n_lags": n, **compare(train, val, n),
         "condition_standardised": condition_numbers(train[lag_columns(n)].to_numpy())["standardised"]}
        for n in lag_options
    ])


def select_n_lags(table: pd.DataFrame, tolerance: float = LAG_TOLERANCE_C) -> int:
    """Pick the fewest lags whose validation MAE is within ``tolerance`` of the best.

    Differences far below the sensor's recording step have no physical meaning,
    so a simpler, better-conditioned model is preferred when errors are that close.

    Args:
        table: Output of :func:`lag_sensitivity`.
        tolerance: Allowed excess over the best validation MAE, in °C.

    Returns:
        The selected lag count.
    """
    best = table["regression_mae"].min()
    return int(table.loc[table["regression_mae"] <= best + tolerance, "n_lags"].min())


def horizon_sensitivity(readings: pd.Series, step_min: int, n_lags: int,
                        horizons=HORIZON_OPTIONS) -> pd.DataFrame:
    """Compare forecast horizons on the validation period at a fixed lag count."""
    rows = []
    for horizon in horizons:
        p = _prepare_from(readings, step_min, horizon, n_lags)
        rows.append({"horizon_min": horizon,
                     **compare(p.frame[p.split.train], p.frame[p.split.val], n_lags)})
    return pd.DataFrame(rows)


def step_sensitivity(readings: pd.Series, horizon_min: int, base_step: int, base_lags: int,
                     steps=STEP_OPTIONS) -> pd.DataFrame:
    """Compare grid steps on validation, keeping the history length similar.

    The lag count is scaled by the step so each row looks back over about the
    same period. Rows report how many examples each step leaves.
    """
    rows = []
    for step in steps:
        n_lags = max(1, round(base_lags * base_step / step))
        p = _prepare_from(readings, step, horizon_min, n_lags)
        rows.append({"step_min": step, "n_lags": n_lags,
                     "train_examples": int(p.split.train.sum()),
                     "slot_fill": float(p.grid.notna().mean()),
                     **compare(p.frame[p.split.train], p.frame[p.split.val], n_lags)})
    return pd.DataFrame(rows)


def bootstrap_mae_gain(index: pd.DatetimeIndex, y_true, baseline_pred, model_pred,
                       n_resamples: int = BOOTSTRAP_RESAMPLES,
                       seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    """Estimate uncertainty in ``baseline MAE - model MAE`` by resampling days.

    Errors minutes apart are strongly correlated, so resampling single rows
    would overstate the evidence. Whole local calendar days are resampled
    instead, which keeps the correlation within each day.

    Args:
        index: Tz-aware issue times.
        y_true: Observed values.
        baseline_pred: Baseline forecasts.
        model_pred: Model forecasts.
        n_resamples: Bootstrap resamples.
        seed: Random seed, fixed so the result repeats exactly.

    Returns:
        The observed gain, the 2.5% and 97.5% percentiles of the resampled gains,
        and the number of days. A positive gain means the model has lower MAE.
    """
    y = np.asarray(y_true)
    gains = np.abs(y - np.asarray(baseline_pred)) - np.abs(y - np.asarray(model_pred))
    days = pd.Series(gains).groupby(np.asarray(index.tz_convert(LOCAL_TZ).date))
    sums, counts = days.sum().to_numpy(), days.size().to_numpy()
    picks = np.random.default_rng(seed).integers(0, len(sums), (n_resamples, len(sums)))
    resampled = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5])
    return {"mae_gain": float(gains.mean()), "ci95_low": float(low),
            "ci95_high": float(high), "n_days": int(len(sums))}


def errors_by_hour(target_times: pd.DatetimeIndex, y_true, persistence, regression) -> pd.DataFrame:
    """Return MAE per local hour of day of the forecast's target time."""
    hours = target_times.tz_convert(LOCAL_TZ).hour
    y = np.asarray(y_true)
    return (pd.DataFrame({"hour_local": hours,
                          "persistence_mae": np.abs(y - np.asarray(persistence)),
                          "regression_mae": np.abs(y - np.asarray(regression))})
            .groupby("hour_local").mean().reset_index())


def measure_performance(train: pd.DataFrame, n_lags: int, repeats: int = 20) -> dict:
    """Measure runtime and memory, and estimate FLOPS, for both models.

    Times are the median of ``repeats`` runs on this machine, so treat them as
    indicative rather than exact. Peak memory is the Python-level allocation
    during one fit, measured with ``tracemalloc``.
    """
    X, y = train, train["target"]
    fit_s, predict_s, persist_s = [], [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model = LaggedLinearRegression(n_lags).fit(X, y)
        t1 = time.perf_counter()
        model.predict(X)
        t2 = time.perf_counter()
        PersistenceForecaster().predict(X)
        t3 = time.perf_counter()
        fit_s.append(t1 - t0)
        predict_s.append(t2 - t1)
        persist_s.append(t3 - t2)
    tracemalloc.start()
    LaggedLinearRegression(n_lags).fit(X, y)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "n_train": len(X), "n_lags": n_lags, "repeats": repeats,
        "regression_fit_s_median": float(np.median(fit_s)),
        "regression_predict_all_train_s_median": float(np.median(predict_s)),
        "persistence_predict_all_train_s_median": float(np.median(persist_s)),
        "regression_fit_peak_python_alloc_mb": peak / 1e6,
        **flop_estimate(len(X), n_lags),
        "note": "Times are machine-dependent. FLOPS are textbook estimates, not measurements.",
    }


def data_summary(prep: Prepared, sensorid: str, step_min: int, horizon_min: int,
                 max_lags: int) -> dict:
    """Describe the data behind a run, including what was removed and why."""
    grid, split, frame = prep.grid, prep.split, prep.frame
    local = lambda ts: str(ts.tz_convert(LOCAL_TZ))
    values = np.unique(prep.readings.to_numpy())
    period = {"train": split.train, "validation": split.val, "test": split.test}
    return {
        "sensorid": sensorid, "step_min": step_min, "horizon_min": horizon_min,
        "max_lags_for_rows": max_lags,
        "unique_readings": int(len(prep.readings)),
        "first_reading_local": local(prep.readings.index[0]),
        "last_reading_local": local(prep.readings.index[-1]),
        "grid_slots": int(len(grid)),
        "grid_slots_filled_fraction": float(grid.notna().mean()),
        "examples_after_dropping_incomplete": int(len(frame)),
        "examples_purged_at_boundaries": split.purged,
        "examples_per_partition": {k: int(m.sum()) for k, m in period.items()},
        "last_train_slot_local": local(split.boundary_1),
        "last_validation_slot_local": local(split.boundary_2),
        "temperature_min_c": float(values.min()),
        "temperature_max_c": float(values.max()),
        "smallest_step_between_recorded_values_c": float(np.diff(values).min()),
        "median_step_between_recorded_values_c": float(np.median(np.diff(values))),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_select(readings_csv: str | Path, sensorid: str, out_dir: str | Path,
               step_min: int = 20, horizon_min: int = 60, make_plots: bool = True) -> dict:
    """Pick the lag count and run sensitivity studies on validation data only.

    Writes ``data_summary.json``, ``selection.json``, the three
    ``validation_*_sensitivity.csv`` tables and ``performance.json`` to
    ``out_dir``, plus figures if ``make_plots`` is true.

    Returns:
        The selection record that was written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    max_lags = max(LAG_OPTIONS)
    prep = prepare(readings_csv, sensorid, step_min, horizon_min, max_lags)
    _write_json(out / "data_summary.json",
                data_summary(prep, sensorid, step_min, horizon_min, max_lags))

    lags = lag_sensitivity(prep)
    lags.to_csv(out / "validation_lag_sensitivity.csv", index=False)
    n_lags = select_n_lags(lags)
    horizons = horizon_sensitivity(prep.readings, step_min, n_lags)
    horizons.to_csv(out / "validation_horizon_sensitivity.csv", index=False)
    steps = step_sensitivity(prep.readings, horizon_min, step_min, n_lags)
    steps.to_csv(out / "validation_step_sensitivity.csv", index=False)

    train = prep.frame[prep.split.train]
    model = LaggedLinearRegression(n_lags).fit(train, train["target"])
    selection = {
        "sensorid": sensorid, "step_min": step_min, "horizon_min": horizon_min,
        "n_lags": n_lags, "max_lags_for_rows": max_lags,
        "rule": f"Fewest lags with validation regression MAE within {LAG_TOLERANCE_C} °C of the best.",
        "coefficients_fit_on_train": model.coefficients(),
        "condition_number_on_train": condition_numbers(train[lag_columns(n_lags)].to_numpy()),
    }
    _write_json(out / "selection.json", selection)
    _write_json(out / "performance.json", measure_performance(train, n_lags))

    if make_plots:
        plots.series_with_splits(prep.grid, prep.split, sensorid, out / "fig_series_splits.png")
        plots.lag_sensitivity(lags, out / "fig_lag_sensitivity.png")
        plots.horizon_sensitivity(horizons, out / "fig_horizon_sensitivity.png")
    return selection


def run_final(readings_csv: str | Path, sensorid: str, out_dir: str | Path,
              selection_path: str | Path, reproduce: bool = False,
              make_plots: bool = True) -> dict:
    """Evaluate the locked settings on the test period, once.

    The model is fitted on the training period only, so the reported model is
    exactly the one that was assessed on validation.

    Args:
        readings_csv: Tidy readings table.
        sensorid: Sensor to evaluate. This may differ from the sensor the
            settings were selected on, which makes the run an independent check.
        out_dir: Output folder for this sensor.
        selection_path: The ``selection.json`` holding the locked settings.
        reproduce: Allow a rerun when a result already exists. The settings
            come from the same file, so a rerun cannot be used for tuning.
        make_plots: Whether to write figures.

    Returns:
        The final metrics record.

    Raises:
        FileExistsError: If a test result exists and ``reproduce`` is false.
    """
    out = Path(out_dir)
    result_path = out / "test_metrics.json"
    if result_path.exists() and not reproduce:
        raise FileExistsError(f"{result_path} already exists. The test set is used once; "
                              "pass --reproduce only to regenerate the same result.")
    out.mkdir(parents=True, exist_ok=True)
    locked = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    n_lags, step_min = locked["n_lags"], locked["step_min"]
    horizon_min, max_lags = locked["horizon_min"], locked["max_lags_for_rows"]

    prep = prepare(readings_csv, sensorid, step_min, horizon_min, max_lags)
    _write_json(out / "data_summary.json",
                data_summary(prep, sensorid, step_min, horizon_min, max_lags))
    frame, split = prep.frame, prep.split
    train, val, test = frame[split.train], frame[split.val], frame[split.test]
    model = LaggedLinearRegression(n_lags).fit(train, train["target"])

    table = pd.DataFrame([{"partition": name, **compare(train, part, n_lags)}
                          for name, part in (("train", train), ("validation", val), ("test", test))])
    table.to_csv(out / "metrics_by_partition.csv", index=False)

    y = test["target"]
    pred_reg = model.predict(test)
    pred_per = PersistenceForecaster().predict(test)
    step = prep.grid.index[1] - prep.grid.index[0]
    target_times = test.index + prep.horizon_steps * step
    by_hour = errors_by_hour(target_times, y, pred_per, pred_reg)
    by_hour.to_csv(out / "test_error_by_hour.csv", index=False)

    result = {
        "sensorid": sensorid, "settings_from": locked["sensorid"],
        "step_min": step_min, "horizon_min": horizon_min, "n_lags": n_lags,
        "n_test_examples": len(test),
        "test_first_issue_local": str(test.index[0].tz_convert(LOCAL_TZ)),
        "test_last_issue_local": str(test.index[-1].tz_convert(LOCAL_TZ)),
        "persistence": score(y, pred_per),
        "regression": score(y, pred_reg),
        "mae_skill": mae_skill(mae(y, pred_reg), mae(y, pred_per)),
        "mae_gain_day_bootstrap": bootstrap_mae_gain(test.index, y, pred_per, pred_reg),
        "coefficients_fit_on_train": model.coefficients(),
        "condition_number_on_train": condition_numbers(train[lag_columns(n_lags)].to_numpy()),
    }
    _write_json(result_path, result)

    if make_plots:
        plots.series_with_splits(prep.grid, split, sensorid, out / "fig_series_splits.png")
        plots.test_overlay(prep.grid, test.index, pred_per, pred_reg, target_times,
                           OVERLAY_DAYS, out / "fig_test_overlay.png")
        plots.test_metrics(result, out / "fig_test_metrics.png")
        plots.error_by_hour(by_hour, out / "fig_error_by_hour.png")
        plots.residuals(y, pred_per, pred_reg, out / "fig_residuals.png")
        plots.error_over_time(target_times, y, pred_per, pred_reg, out / "fig_error_over_time.png")
    return result
