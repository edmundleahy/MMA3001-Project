"""Turn raw sensor readings into past-only forecasting examples.

The steps, in order, are:

1. :func:`load_temperature` reads one sensor's deduplicated readings.
2. :func:`to_grid` places them on a regular time grid. Empty slots stay NaN;
   nothing is interpolated, so no value is ever estimated from the future.
3. :func:`make_supervised` builds lag columns and the future target.
4. :func:`chronological_split` assigns examples to train, validation and test,
   removing any example whose time window would cross a partition boundary.

Every lag and target position is computed on the regular grid, so a row shift
always equals a true time shift, even across gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Readings before these instants are excluded, per sensor. Sensor ...326 has
#: isolated readings on 2023-12-11 and 2025-09-07 that are 636 and 4.1 days away
#: from its continuous record (measured in the audit), so they are not a usable
#: part of the series.
ANALYSIS_START = {"6012002000326": "2025-09-11 12:00:00+00:00"}

#: Earliest-to-latest fractions of the grid's time span for train and validation.
#: The test partition takes the remaining 20%.
SPLIT_FRACTIONS = (0.6, 0.8)


def load_temperature(
    csv_path: str | Path,
    sensorid: str,
    variable: str = "Temperature",
) -> pd.Series:
    """Load one sensor's readings from the tidy table written by the audit.

    Args:
        csv_path: Path to ``data/processed/readings.csv``.
        sensorid: Sensor ID, e.g. ``"6012002000869"``.
        variable: Exact variable name to load.

    Returns:
        Values in the variable's own unit, indexed by UTC instant and sorted by
        time, with any readings before :data:`ANALYSIS_START` removed.

    Raises:
        ValueError: If the sensor and variable have no readings.
    """
    df = pd.read_csv(csv_path, usecols=["sensorid", "variable", "epoch_s", "value"],
                     dtype={"sensorid": str})
    df = df[(df["sensorid"] == str(sensorid)) & (df["variable"] == variable)]
    if df.empty:
        raise ValueError(f"No {variable!r} readings for sensor {sensorid} in {csv_path}")
    index = pd.to_datetime(df["epoch_s"], unit="s", utc=True)
    series = pd.Series(df["value"].to_numpy(), index=index, name=variable).sort_index()
    start = ANALYSIS_START.get(str(sensorid))
    return series if start is None else series[series.index >= pd.Timestamp(start)]


def to_grid(readings: pd.Series, step_min: int) -> pd.Series:
    """Resample readings onto a regular grid without using future information.

    A slot labelled ``t`` holds the **latest reading in** ``(t - step, t]``. The
    slot is labelled at its end, so its value was already known at time ``t``.
    Labelling at the start of the slot would leak up to one step of the future
    into every feature.

    Args:
        readings: Values indexed by a sorted, tz-aware UTC DatetimeIndex.
        step_min: Grid spacing in minutes. Must be a positive integer.

    Returns:
        A regularly spaced series in which empty slots are NaN.

    Raises:
        ValueError: If ``step_min`` is not positive or there are no readings.
    """
    if step_min <= 0:
        raise ValueError("step_min must be positive")
    if readings.empty:
        raise ValueError("Cannot build a grid from zero readings")
    return readings.resample(f"{step_min}min", closed="right", label="right",
                             origin="epoch").last()


def horizon_to_steps(horizon_min: int, step_min: int) -> int:
    """Convert a horizon in minutes to a whole number of grid steps.

    Raises:
        ValueError: If the horizon is not a positive multiple of the step, since
            the target would otherwise not fall on the grid.
    """
    if horizon_min <= 0 or horizon_min % step_min:
        raise ValueError(f"Horizon {horizon_min} min must be a positive multiple "
                         f"of the {step_min} min step")
    return horizon_min // step_min


def lag_columns(n_lags: int) -> list[str]:
    """Return feature names: ``lag_0`` is T(t), ``lag_1`` is T(t - step), and so on."""
    return [f"lag_{i}" for i in range(n_lags)]


def make_supervised(grid: pd.Series, n_lags: int, horizon_steps: int) -> pd.DataFrame:
    """Build lagged features and the future target from a regular grid.

    For issue time ``t`` the features are ``T(t), T(t - step), ...`` and the
    target is ``T(t + horizon)``. Rows with any missing value are removed rather
    than filled, so a gap can never be bridged.

    Args:
        grid: Output of :func:`to_grid`.
        n_lags: Number of lag columns, at least 1.
        horizon_steps: Steps between the issue time and the target, at least 1.

    Returns:
        A frame indexed by issue time with columns ``lag_0 ... lag_{n-1}`` and
        ``target``.

    Raises:
        ValueError: If the arguments are invalid or the grid is irregular.
    """
    if n_lags < 1 or horizon_steps < 1:
        raise ValueError("n_lags and horizon_steps must both be at least 1")
    if len(grid) > 1 and grid.index.to_series().diff().dropna().nunique() != 1:
        raise ValueError("Grid must be regularly spaced; use to_grid() first")
    data = {name: grid.shift(i) for i, name in enumerate(lag_columns(n_lags))}
    data["target"] = grid.shift(-horizon_steps)
    return pd.DataFrame(data).dropna()


@dataclass
class Split:
    """Boolean masks over a supervised frame, plus the time boundaries.

    Attributes:
        train: Examples whose whole window lies in the training period.
        val: Examples whose whole window lies in the validation period.
        test: Examples whose whole window lies in the test period.
        boundary_1: Last grid slot belonging to training.
        boundary_2: Last grid slot belonging to validation.
        purged: Examples removed because their window crossed a boundary.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    boundary_1: pd.Timestamp
    boundary_2: pd.Timestamp
    purged: int


def chronological_split(
    frame: pd.DataFrame,
    grid_index: pd.DatetimeIndex,
    n_lags: int,
    horizon_steps: int,
) -> Split:
    """Split examples by time, purging windows that cross a boundary.

    The boundaries come from the grid's full time span, so they do not move when
    the lag count or horizon changes. An example's window runs from its oldest
    lag slot to its target slot, and the whole window must lie inside one
    partition. That is the embargo: no value is shared between partitions.

    Args:
        frame: Output of :func:`make_supervised`.
        grid_index: Index of the grid the frame was built from.
        n_lags: Lag count used to build ``frame``.
        horizon_steps: Horizon, in steps, used to build ``frame``.

    Returns:
        A :class:`Split` holding the masks and boundaries.
    """
    step = grid_index[1] - grid_index[0]
    span = grid_index[-1] - grid_index[0]
    b1 = grid_index[grid_index <= grid_index[0] + SPLIT_FRACTIONS[0] * span][-1]
    b2 = grid_index[grid_index <= grid_index[0] + SPLIT_FRACTIONS[1] * span][-1]
    first = frame.index - (n_lags - 1) * step
    last = frame.index + horizon_steps * step
    train = np.asarray(last <= b1)
    val = np.asarray((first > b1) & (last <= b2))
    test = np.asarray(first > b2)
    purged = int(len(frame) - train.sum() - val.sum() - test.sum())
    return Split(train, val, test, b1, b2, purged)
