"""Report figures. Every axis states its unit, and time axes state the time zone.

Colours are used consistently: observed data is dark neutral, persistence is
orange and regression is blue, in every figure. Line styles differ as well, so
the figures still read in greyscale or with colour-vision deficiency.
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from .data_processing import LOCAL_TZ  # noqa: E402

OBSERVED = "#0b0b0b"
PERSISTENCE = "#eb6834"
REGRESSION = "#2a78d6"
SECONDARY_INK = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#f0efec"
TZ_LABEL = f"local time, {LOCAL_TZ}"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": SECONDARY_INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.titlesize": 11,
    "axes.titlelocation": "left",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def _time_axis(ax) -> None:
    """Show tick labels in Melbourne local time, handling AEST and AEDT."""
    tz = ZoneInfo(LOCAL_TZ)
    locator = mdates.AutoDateLocator(tz=tz)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=tz))
    ax.set_xlabel(f"Date ({TZ_LABEL})")


def _save(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)


def series_with_splits(grid: pd.Series, split, sensorid: str, path: Path) -> None:
    """Plot the gridded series with the three periods marked. Gaps stay visible."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axvspan(split.boundary_1, split.boundary_2, color=BAND, zorder=0)
    ax.plot(grid.index, grid.to_numpy(), color=OBSERVED, lw=0.7)
    top = 1.01
    transform = ax.get_xaxis_transform()
    middle = split.boundary_1 + (split.boundary_2 - split.boundary_1) / 2
    ax.text(grid.index[0], top, "Train", transform=transform, color=SECONDARY_INK)
    ax.text(middle, top, "Validation", transform=transform, ha="center", color=SECONDARY_INK)
    ax.text(grid.index[-1], top, "Test", transform=transform, ha="right", color=SECONDARY_INK)
    ax.set_title(f"Sensor {sensorid}: observed temperature\n", pad=6)
    ax.set_ylabel("Temperature (°C)")
    _time_axis(ax)
    _save(fig, path)


def _two_lines(x, persistence, regression, xlabel: str, title: str, path: Path) -> None:
    """Draw MAE for both models against one numeric setting."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(x, persistence, color=PERSISTENCE, ls="--", marker="s", ms=6, lw=2, label="Persistence")
    ax.plot(x, regression, color=REGRESSION, marker="o", ms=6, lw=2, label="Linear regression")
    ax.set_xticks(list(x))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Validation MAE (°C)")
    ax.set_title(title)
    ax.legend()
    _save(fig, path)


def lag_sensitivity(table: pd.DataFrame, path: Path) -> None:
    """Plot validation MAE against the number of lags."""
    _two_lines(table["n_lags"], table["persistence_mae"], table["regression_mae"],
               "Number of lagged inputs", "Validation error by lag count", path)


def horizon_sensitivity(table: pd.DataFrame, path: Path) -> None:
    """Plot validation MAE against the forecast horizon."""
    _two_lines(table["horizon_min"], table["persistence_mae"], table["regression_mae"],
               "Forecast horizon (minutes)", "Validation error by forecast horizon", path)


def test_overlay(grid: pd.Series, issue_times: pd.DatetimeIndex, persistence, regression,
                 target_times: pd.DatetimeIndex, days: int, path: Path) -> None:
    """Overlay observed and forecast temperatures for the first days of the test period."""
    end = target_times[0] + pd.Timedelta(days=days)
    keep = np.asarray(target_times <= end)
    observed = grid[(grid.index >= target_times[0]) & (grid.index <= end)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(observed.index, observed.to_numpy(), color=OBSERVED, lw=1.4, label="Observed")
    ax.plot(target_times[keep], np.asarray(persistence)[keep], color=PERSISTENCE, ls="--", lw=1,
            label="Persistence forecast")
    ax.plot(target_times[keep], np.asarray(regression)[keep], color=REGRESSION, lw=1,
            label="Regression forecast")
    ax.set_title(f"Test period, first {days} days: observed and forecast temperature")
    ax.set_ylabel("Temperature (°C)")
    _time_axis(ax)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    _save(fig, path)


def test_metrics(result: dict, path: Path) -> None:
    """Compare both models' test MAE and RMSE as labelled bars."""
    names = ["MAE", "RMSE"]
    per = [result["persistence"]["mae"], result["persistence"]["rmse"]]
    reg = [result["regression"]["mae"], result["regression"]["rmse"]]
    x = np.arange(2)
    width = 0.32
    fig, ax = plt.subplots(figsize=(5, 3.4))
    a = ax.bar(x - width / 2 - 0.01, per, width, color=PERSISTENCE, label="Persistence")
    b = ax.bar(x + width / 2 + 0.01, reg, width, color=REGRESSION, label="Linear regression")
    ax.bar_label(a, fmt="%.3f", padding=2, color=SECONDARY_INK)
    ax.bar_label(b, fmt="%.3f", padding=2, color=SECONDARY_INK)
    ax.set_xticks(x, names)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Test error (°C)")
    ax.set_title(f"Test error, {result['n_test_examples']:,} forecasts")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    _save(fig, path)


def error_by_hour(table: pd.DataFrame, path: Path) -> None:
    """Plot test MAE against the local hour of the forecast's target time."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(table["hour_local"], table["persistence_mae"], color=PERSISTENCE, ls="--",
            marker="s", ms=4, lw=1.6, label="Persistence")
    ax.plot(table["hour_local"], table["regression_mae"], color=REGRESSION,
            marker="o", ms=4, lw=1.6, label="Linear regression")
    ax.set_xticks(range(0, 24, 3))
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"Hour of target time ({TZ_LABEL})")
    ax.set_ylabel("Test MAE (°C)")
    ax.set_title("Test error by time of day")
    ax.legend()
    _save(fig, path)


def residuals(y_true, persistence, regression, path: Path) -> None:
    """Plot both models' test error distributions as outlines on shared bins."""
    y = np.asarray(y_true)
    err_p, err_r = np.asarray(persistence) - y, np.asarray(regression) - y
    limit = max(np.abs(err_p).max(), np.abs(err_r).max())
    bins = np.linspace(-limit, limit, 41)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(err_p, bins=bins, histtype="step", color=PERSISTENCE, ls="--", lw=1.6, label="Persistence")
    ax.hist(err_r, bins=bins, histtype="step", color=REGRESSION, lw=1.6, label="Linear regression")
    ax.set_xlabel("Forecast minus observed (°C)")
    ax.set_ylabel("Forecasts")
    ax.set_title("Test error distribution")
    ax.legend()
    _save(fig, path)


def error_over_time(target_times: pd.DatetimeIndex, y_true, persistence, regression,
                    path: Path) -> None:
    """Plot daily test MAE to show whether error drifts through the period."""
    y = np.asarray(y_true)
    days = target_times.tz_convert(LOCAL_TZ).normalize()
    frame = pd.DataFrame({"p": np.abs(np.asarray(persistence) - y),
                          "r": np.abs(np.asarray(regression) - y)},
                         index=days).groupby(level=0).mean()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(frame.index, frame["p"], color=PERSISTENCE, ls="--", marker="s", ms=4, lw=1.6, label="Persistence")
    ax.plot(frame.index, frame["r"], color=REGRESSION, marker="o", ms=4, lw=1.6, label="Linear regression")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Daily test MAE (°C)")
    ax.set_title("Test error by day")
    _time_axis(ax)
    ax.legend()
    _save(fig, path)
