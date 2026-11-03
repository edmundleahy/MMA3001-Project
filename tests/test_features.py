"""Verification tests for grid building, feature timing and leakage prevention."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    chronological_split,
    horizon_to_steps,
    lag_columns,
    load_temperature,
    make_supervised,
    to_grid,
)

T0 = pd.Timestamp("2024-03-01 00:00", tz="UTC")


def grid_of(values, step_min=20):
    """Build a regular grid from values; use np.nan for a missing slot."""
    index = pd.date_range(T0, periods=len(values), freq=f"{step_min}min")
    return pd.Series(values, index=index, dtype=float)


# ---------- to_grid ----------

def test_slot_holds_latest_reading_up_to_its_own_label():
    readings = pd.Series([1.0, 2.0, 3.0], index=[
        T0 + pd.Timedelta(minutes=5),
        T0 + pd.Timedelta(minutes=20),   # exactly on the slot end, still known at 00:20
        T0 + pd.Timedelta(minutes=20, seconds=1),  # one second later, must go to the next slot
    ])
    grid = to_grid(readings, 20)
    assert grid[T0 + pd.Timedelta(minutes=20)] == 2.0
    assert grid[T0 + pd.Timedelta(minutes=40)] == 3.0


def test_grid_keeps_gaps_as_nan_and_does_not_interpolate():
    readings = pd.Series([1.0, 9.0], index=[T0, T0 + pd.Timedelta(minutes=80)])
    grid = to_grid(readings, 20)
    assert grid.isna().sum() == 3
    assert grid.dropna().tolist() == [1.0, 9.0]


def test_grid_is_anchored_regardless_of_first_reading():
    a = to_grid(pd.Series([1.0], index=[T0 + pd.Timedelta(minutes=3)]), 20)
    b = to_grid(pd.Series([1.0], index=[T0 + pd.Timedelta(minutes=17)]), 20)
    assert a.index[0] == b.index[0] == T0 + pd.Timedelta(minutes=20)


@pytest.mark.parametrize("step", [0, -20])
def test_grid_rejects_invalid_step(step):
    with pytest.raises(ValueError):
        to_grid(pd.Series([1.0], index=[T0]), step)


def test_grid_rejects_empty_input():
    empty = pd.Series([], index=pd.DatetimeIndex([], tz="UTC"), dtype=float)
    with pytest.raises(ValueError):
        to_grid(empty, 20)


# ---------- horizon ----------

def test_horizon_converts_to_steps():
    assert horizon_to_steps(60, 20) == 3


@pytest.mark.parametrize("horizon", [0, -20, 50])
def test_horizon_must_be_positive_multiple_of_step(horizon):
    with pytest.raises(ValueError):
        horizon_to_steps(horizon, 20)


# ---------- make_supervised ----------

def test_target_is_exactly_one_horizon_after_issue_time():
    # With values 0, 1, 2, ... the value equals its slot number, so offsets are visible.
    frame = make_supervised(grid_of(np.arange(10.0)), n_lags=2, horizon_steps=3)
    assert (frame["lag_0"] == frame["lag_1"] + 1).all()
    assert (frame["target"] == frame["lag_0"] + 3).all()
    assert frame.index[0] == T0 + pd.Timedelta(minutes=20)  # first row with both lags
    assert frame.iloc[0].to_dict() == {"lag_0": 1.0, "lag_1": 0.0, "target": 4.0}


def test_gap_is_not_bridged_by_row_shift():
    values = np.arange(10.0)
    values[4] = np.nan
    frame = make_supervised(grid_of(values), n_lags=2, horizon_steps=2)
    assert (frame["target"] == frame["lag_0"] + 2).all()
    assert (frame["lag_0"] == frame["lag_1"] + 1).all()
    # Slot 4 cannot be a target (issue 2), a lag_0 (issue 4) or a lag_1 (issue 5).
    issue_slots = ((frame.index - T0) / pd.Timedelta(minutes=20)).astype(int).tolist()
    assert not {2, 4, 5} & set(issue_slots)


def test_features_do_not_change_when_future_values_change():
    base = np.arange(20.0)
    changed = base.copy()
    changed[12:] += 100.0  # the future, from slot 12 on, is rewritten
    a = make_supervised(grid_of(base), n_lags=3, horizon_steps=3)
    b = make_supervised(grid_of(changed), n_lags=3, horizon_steps=3)
    issued_before = a.index < grid_of(base).index[12]
    pd.testing.assert_frame_equal(a.loc[issued_before, lag_columns(3)],
                                  b.loc[issued_before, lag_columns(3)])


def test_make_supervised_rejects_irregular_grid():
    index = pd.DatetimeIndex([T0, T0 + pd.Timedelta(minutes=20), T0 + pd.Timedelta(minutes=50)])
    with pytest.raises(ValueError):
        make_supervised(pd.Series([1.0, 2.0, 3.0], index=index), 1, 1)


@pytest.mark.parametrize("lags, horizon", [(0, 1), (1, 0)])
def test_make_supervised_rejects_invalid_counts(lags, horizon):
    with pytest.raises(ValueError):
        make_supervised(grid_of(np.arange(5.0)), lags, horizon)


# ---------- chronological_split ----------

def split_of(n=100, n_lags=3, horizon=3):
    grid = grid_of(np.arange(float(n)))
    frame = make_supervised(grid, n_lags, horizon)
    return grid, frame, chronological_split(frame, grid.index, n_lags, horizon)


def test_partitions_are_ordered_and_disjoint():
    _, frame, s = split_of()
    assert not (s.train & s.val).any() and not (s.val & s.test).any() and not (s.train & s.test).any()
    assert frame.index[s.train].max() < frame.index[s.val].min()
    assert frame.index[s.val].max() < frame.index[s.test].min()


def test_no_slot_is_shared_between_partitions():
    # Values are slot numbers, so each row's values show exactly which slots it uses.
    _, frame, s = split_of()
    used = [set(frame.loc[m].to_numpy().ravel()) for m in (s.train, s.val, s.test)]
    assert not used[0] & used[1] and not used[1] & used[2] and not used[0] & used[2]


def test_examples_crossing_a_boundary_are_purged_and_counted():
    _, frame, s = split_of(n=100, n_lags=3, horizon=3)
    # Windows span 6 steps, so 5 issue times straddle each of the 2 boundaries.
    assert s.purged == 10
    assert s.train.sum() + s.val.sum() + s.test.sum() + s.purged == len(frame)


def test_boundaries_do_not_depend_on_lags_or_horizon():
    _, _, a = split_of(n_lags=1, horizon=1)
    _, _, b = split_of(n_lags=6, horizon=9)
    assert a.boundary_1 == b.boundary_1 and a.boundary_2 == b.boundary_2


# ---------- load_temperature ----------

def write_readings(path, rows):
    pd.DataFrame(rows, columns=["sensorid", "variable", "epoch_s", "value"]).to_csv(path, index=False)
    return path


def test_load_selects_sensor_and_variable_and_sorts(tmp_path):
    t = int(T0.timestamp())
    path = write_readings(tmp_path / "r.csv", [
        ("A", "Temperature", t + 600, 2.0),
        ("A", "Temperature", t, 1.0),
        ("A", "Humidity", t, 50.0),
        ("B", "Temperature", t, 9.0),
    ])
    s = load_temperature(path, "A")
    assert s.tolist() == [1.0, 2.0]
    assert s.index.is_monotonic_increasing and str(s.index.tz) == "UTC"


def test_load_excludes_readings_before_analysis_start(tmp_path):
    sensor = "6012002000326"
    early = int(pd.Timestamp("2023-12-11", tz="UTC").timestamp())
    late = int(pd.Timestamp("2025-10-01", tz="UTC").timestamp())
    path = write_readings(tmp_path / "r.csv", [(sensor, "Temperature", early, 1.0),
                                               (sensor, "Temperature", late, 2.0)])
    assert load_temperature(path, sensor).tolist() == [2.0]


def test_load_unknown_sensor_raises(tmp_path):
    path = write_readings(tmp_path / "r.csv", [("A", "Temperature", int(T0.timestamp()), 1.0)])
    with pytest.raises(ValueError):
        load_temperature(path, "missing")
