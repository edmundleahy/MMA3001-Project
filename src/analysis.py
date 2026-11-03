"""Data audit: coverage, sampling cadence, gaps and plausibility per sensor.

The audit is run *before* any modelling choice (sensor, resampling grid,
forecast horizon) is made, so those choices can be justified from measured
coverage rather than assumed from the archive's file name.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data_processing import LOCAL_TZ, DedupReport, ParseStats

#: Gap lengths (minutes) reported in the audit. They are reporting bins,
#: not cleaning thresholds: nothing is removed because of them.
GAP_THRESHOLDS_MIN = (60, 6 * 60, 24 * 60)


def sampling_intervals_min(epoch_s: pd.Series) -> pd.Series:
    """Return the time between consecutive unique readings, in minutes.

    ``epoch_s`` must belong to one sensor and one variable. Values are sorted
    first, so the result does not depend on input order.
    """
    t = np.sort(pd.unique(epoch_s.to_numpy()))
    return pd.Series(np.diff(t) / 60.0, name="interval_min")


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise each ``(sensorid, variable)`` in a deduplicated readings frame.

    Columns include first/last local time, number of readings, median and
    90th-percentile sampling interval, count of gaps longer than each of
    :data:`GAP_THRESHOLDS_MIN`, the longest gap, and value percentiles.
    """
    rows = []
    for (sid, var), grp in df.groupby(["sensorid", "variable"], sort=True):
        iv = sampling_intervals_min(grp["epoch_s"])
        v = grp["value"]
        rec = {
            "sensorid": sid,
            "variable": var,
            "unit": grp["unit"].iloc[0],
            "n_readings": len(grp),
            "first_local": grp["time_local"].min(),
            "last_local": grp["time_local"].max(),
            "span_days": (grp["epoch_s"].max() - grp["epoch_s"].min()) / 86400,
            "median_interval_min": iv.median() if len(iv) else np.nan,
            "p90_interval_min": iv.quantile(0.9) if len(iv) else np.nan,
            "longest_gap_days": iv.max() / 1440 if len(iv) else np.nan,
        }
        for thr in GAP_THRESHOLDS_MIN:
            rec[f"gaps_over_{thr}min"] = int((iv > thr).sum())
        rec.update({
            "value_min": v.min(),
            "value_p01": v.quantile(0.01),
            "value_median": v.median(),
            "value_p99": v.quantile(0.99),
            "value_max": v.max(),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count readings per sensor per local calendar month (wide table).

    This shows the *actual* event-time coverage, which may differ from the
    archive name ``MayToDec2024``.
    """
    month = df["time_local"].dt.tz_localize(None).dt.to_period("M").astype(str)
    return (
        df.assign(month=month)
        .pivot_table(index="month", columns="sensorid", values="value",
                     aggfunc="size", fill_value=0)
        .sort_index()
    )


def plot_series(df: pd.DataFrame, variable: str, path: Path) -> None:
    """Save one panel per sensor of ``variable`` against local time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[df["variable"] == variable]
    sensors = sorted(sub["sensorid"].unique())
    if not sensors:
        return
    unit = sub["unit"].iloc[0]
    fig, axes = plt.subplots(len(sensors), 1, figsize=(11, 2.2 * len(sensors)),
                             sharex=True, squeeze=False)
    for ax, sid in zip(axes[:, 0], sensors):
        s = sub[sub["sensorid"] == sid].sort_values("epoch_s")
        # Plot as points so gaps stay visible instead of being bridged by lines.
        ax.plot(s["time_local"], s["value"], ".", ms=1)
        ax.set_ylabel(f"{unit}")
        ax.set_title(f"Sensor {sid}", fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel(f"Event time ({LOCAL_TZ})")
    fig.suptitle(f"{variable} readings by sensor (raw, deduplicated)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_audit(
    df: pd.DataFrame,
    stats: ParseStats,
    dedup: DedupReport,
    createdate_range: tuple[str | None, str | None],
    out_dir: Path,
) -> dict:
    """Write audit tables, figure and a JSON summary into ``out_dir``.

    Files written: ``parse_summary.json``, ``variables_seen.csv``,
    ``coverage.csv``, ``monthly_counts.csv`` and one ``<variable>_by_sensor.png``
    per audited variable.

    Returns:
        The summary dictionary that was written to ``parse_summary.json``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    vars_seen = pd.DataFrame(
        [(n, u, c) for (n, u), c in stats.variables_seen.items()],
        columns=["variable", "unit", "n_readings_all_rows"],
    ).sort_values("n_readings_all_rows", ascending=False)
    vars_seen.to_csv(out_dir / "variables_seen.csv", index=False)

    cov = coverage_table(df)
    cov.to_csv(out_dir / "coverage.csv", index=False)
    monthly_counts(df).to_csv(out_dir / "monthly_counts.csv")

    for var in sorted(df["variable"].unique()):
        safe = var.lower().replace(" ", "_")
        plot_series(df, var, out_dir / f"{safe}_by_sensor.png")

    summary = {
        "parse": stats.as_dict(),
        "dedup": asdict(dedup),
        "event_time_utc_range": [
            str(df["time_utc"].min()) if len(df) else None,
            str(df["time_utc"].max()) if len(df) else None,
        ],
        "createdate_range_raw": list(createdate_range),
        "note": "Event time = jsondata values[].time. createdate is "
                "ingestion metadata and is reported only for comparison.",
    }
    (out_dir / "parse_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
