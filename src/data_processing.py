"""Streaming ingestion of the Monash environmental-sensor CSV (inside a ZIP).

The raw file has one CSV row per ingestion record. The measurements live in
the ``jsondata`` column, which is a JSON array *encoded as a string* inside the
CSV, e.g.::

    [{"values": [{"time": 1702273965, "value": 23.8}],
      "variable": {"id": 68, "name": "Temperature", "unit": "°C"}}]

Design decisions (each is deliberate and tested in ``tests/test_parser.py``):

* **Streaming.** The ZIP member (~353 MB uncompressed) is read row by row with
  ``zipfile`` + ``csv.DictReader``; it is never extracted to disk or loaded
  whole into memory.
* **Select variables by name**, never by the nested ``id`` or array position.
* **Event time, not ingestion time.** ``values[].time`` (Unix epoch seconds,
  a UTC instant) is the measurement time. ``createdate``/``processdate`` are
  pipeline metadata and are kept only for auditing.
* **No silent dropping.** Every record or reading that cannot be used is
  counted in :class:`ParseStats` under a named reason.
* **Deterministic duplicate rule.** See :func:`deduplicate`.
"""

from __future__ import annotations

import csv
import io
import json
import math
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

#: Timezone of the Monash Clayton campus, used only for calendar features
#: (hour of day etc.). Ordering and splitting always use UTC instants.
LOCAL_TZ = "Australia/Melbourne"

#: Epoch-second bounds outside which a timestamp is treated as implausible
#: (2015-01-01 to 2030-01-01 UTC). This catches millisecond epochs or zeros
#: being misread as seconds. The bounds are wide on purpose: they are a unit
#: sanity check, not a data-cleaning filter.
MIN_PLAUSIBLE_EPOCH = 1_420_070_400
MAX_PLAUSIBLE_EPOCH = 1_893_456_000

#: Columns of the tidy readings table produced by :func:`readings_to_frame`.
READING_COLUMNS = [
    "row_id",
    "sensorid",
    "variable",
    "unit",
    "epoch_s",
    "value",
    "createdate",
]


@dataclass
class ParseStats:
    """Counters describing everything the parser saw and why anything was skipped.

    Attributes:
        rows: CSV rows read.
        bad_json: rows whose ``jsondata`` was empty or not valid JSON.
        not_a_list: rows whose decoded ``jsondata`` was not a JSON array.
        bad_entry: variable entries missing ``variable.name`` or ``values``.
        bad_value: readings whose value was missing, non-numeric or NaN/inf.
        bad_time: readings whose time was missing, non-numeric or implausible.
        readings_kept: readings yielded for the requested variables.
        variables_seen: count of readings per ``(name, unit)`` across *all*
            variables, kept or not, so the audit can list what exists.
    """

    rows: int = 0
    bad_json: int = 0
    not_a_list: int = 0
    bad_entry: int = 0
    bad_value: int = 0
    bad_time: int = 0
    readings_kept: int = 0
    variables_seen: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        """Return the scalar counters as a plain dictionary (for JSON output)."""
        return {
            "rows": self.rows,
            "bad_json": self.bad_json,
            "not_a_list": self.not_a_list,
            "bad_entry": self.bad_entry,
            "bad_value": self.bad_value,
            "bad_time": self.bad_time,
            "readings_kept": self.readings_kept,
        }


def _to_float(x) -> float | None:
    """Convert ``x`` to a finite float, or return ``None`` if that is impossible.

    Booleans are rejected explicitly because ``float(True) == 1.0`` would
    otherwise turn a JSON ``true`` into a plausible-looking reading.
    """
    if x is None or isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def parse_jsondata(
    text: str | None,
    variables: Iterable[str] | None,
    stats: ParseStats | None = None,
) -> list[tuple[str, str, int, float]]:
    """Decode one ``jsondata`` string into readings for the requested variables.

    Args:
        text: The raw ``jsondata`` cell.
        variables: Variable names to keep (exact match on ``variable.name``,
            e.g. ``"Temperature"``). ``None`` keeps every variable.
        stats: Optional counters, updated in place.

    Returns:
        A list of ``(name, unit, epoch_s, value)`` tuples. ``epoch_s`` is an
        integer Unix time in seconds (UTC). Malformed input yields an empty
        list and increments the relevant counter; it never raises.
    """
    stats = stats if stats is not None else ParseStats()
    wanted = None if variables is None else set(variables)

    if not text or not text.strip():
        stats.bad_json += 1
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        stats.bad_json += 1
        return []
    if not isinstance(payload, list):
        stats.not_a_list += 1
        return []

    out: list[tuple[str, str, int, float]] = []
    for entry in payload:
        var = entry.get("variable") if isinstance(entry, dict) else None
        values = entry.get("values") if isinstance(entry, dict) else None
        name = var.get("name") if isinstance(var, dict) else None
        if not isinstance(name, str) or not isinstance(values, list):
            stats.bad_entry += 1
            continue
        unit = var.get("unit") if isinstance(var.get("unit"), str) else ""
        stats.variables_seen[(name, unit)] += len(values)
        if wanted is not None and name not in wanted:
            continue

        for v in values:
            if not isinstance(v, dict):
                stats.bad_value += 1
                continue
            t = _to_float(v.get("time"))
            if t is None or not (MIN_PLAUSIBLE_EPOCH <= t <= MAX_PLAUSIBLE_EPOCH):
                stats.bad_time += 1
                continue
            val = _to_float(v.get("value"))
            if val is None:
                stats.bad_value += 1
                continue
            out.append((name, unit, int(t), val))
            stats.readings_kept += 1
    return out


def find_csv_member(zf: zipfile.ZipFile) -> str:
    """Return the name of the single CSV inside the archive.

    Raises:
        ValueError: if the archive holds zero or several CSV files, so the
            caller must choose explicitly rather than us guessing.
    """
    csvs = [
        n for n in zf.namelist()
        if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")
    ]
    if len(csvs) != 1:
        raise ValueError(f"Expected exactly one CSV in archive, found: {csvs}")
    return csvs[0]


def iter_raw_rows(zip_path: str | Path, member: str | None = None) -> Iterator[dict]:
    """Yield raw CSV rows (as dicts) from a ZIP without extracting it.

    Args:
        zip_path: Path to the ``.zip`` file.
        member: CSV name inside the ZIP; detected automatically if omitted.
    """
    # jsondata cells can be long; lift the default 128 kB csv field limit.
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with zipfile.ZipFile(zip_path) as zf:
        member = member or find_csv_member(zf)
        with zf.open(member) as raw:
            # utf-8-sig strips a byte-order mark if present; newline="" is
            # required by the csv module for quoted fields containing newlines.
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            yield from csv.DictReader(text)


def stream_readings(
    zip_path: str | Path,
    variables: Iterable[str] | None = ("Temperature",),
    sensors: Iterable[str] | None = None,
    stats: ParseStats | None = None,
    limit_rows: int | None = None,
) -> Iterator[tuple]:
    """Stream tidy readings from the raw ZIP.

    Args:
        zip_path: Path to ``5EnvSensor_MayToDec2024_180kRows.zip``.
        variables: Variable names to keep; ``None`` keeps all.
        sensors: Optional sensor IDs (as strings) to keep.
        stats: Optional :class:`ParseStats`, updated in place.
        limit_rows: Stop after this many CSV rows (for quick inspection).

    Yields:
        Tuples in :data:`READING_COLUMNS` order.
    """
    stats = stats if stats is not None else ParseStats()
    sensor_set = None if sensors is None else {str(s) for s in sensors}
    for row in iter_raw_rows(zip_path):
        if limit_rows is not None and stats.rows >= limit_rows:
            break
        stats.rows += 1
        sid = (row.get("sensorid") or "").strip()
        if sensor_set is not None and sid not in sensor_set:
            continue
        for name, unit, t, val in parse_jsondata(row.get("jsondata"), variables, stats):
            yield (row.get("id"), sid, name, unit, t, val, row.get("createdate"))


def readings_to_frame(readings: Iterable[tuple]) -> pd.DataFrame:
    """Build a DataFrame from reading tuples and add timezone-aware timestamps.

    Adds ``time_utc`` (tz-aware UTC) and ``time_local`` (tz-aware
    Australia/Melbourne, so AEST/AEDT is handled by the tz database rather
    than a fixed +10/+11 offset).
    """
    df = pd.DataFrame(list(readings), columns=READING_COLUMNS)
    df["sensorid"] = df["sensorid"].astype(str)
    df["epoch_s"] = df["epoch_s"].astype("int64")
    df["value"] = df["value"].astype("float64")
    df["time_utc"] = pd.to_datetime(df["epoch_s"], unit="s", utc=True)
    df["time_local"] = df["time_utc"].dt.tz_convert(LOCAL_TZ)
    return df


@dataclass
class DedupReport:
    """Outcome of :func:`deduplicate`.

    Attributes:
        rows_in: readings before deduplication.
        rows_out: unique ``(sensorid, variable, epoch_s)`` keys after.
        exact_duplicates: surplus readings whose value matched another
            reading with the same key (pure repeats; safe to drop).
        conflicting_keys: keys that had two or more *different* values.
        max_conflict_spread: largest ``max - min`` among conflicting keys,
            in the variable's own unit (0 if none), to judge whether the
            conflicts matter.
    """

    rows_in: int
    rows_out: int
    exact_duplicates: int
    conflicting_keys: int
    max_conflict_spread: float


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, DedupReport]:
    """Collapse readings to one value per ``(sensorid, variable, epoch_s)``.

    Rule: identical repeats collapse to one; if a key has *different* values,
    the **median** is used and the key is counted as a conflict. The median
    is order-independent, so the result does not depend on CSV row order.
    The conflict count and spread are reported so the rule can be revisited
    if conflicts turn out to be common or large.

    Returns:
        ``(deduplicated_frame, DedupReport)``, sorted by sensor, variable and time.
    """
    key = ["sensorid", "variable", "epoch_s"]
    exact = int(df.duplicated(subset=key + ["value"]).sum())
    g = df.groupby(key, sort=True)
    spread = g["value"].agg(lambda s: s.max() - s.min())
    n_unique = g["value"].nunique()
    conflicts = spread[n_unique > 1]

    out = g.agg(
        unit=("unit", "first"),
        value=("value", "median"),
        n_raw=("value", "size"),
    ).reset_index()
    out["time_utc"] = pd.to_datetime(out["epoch_s"], unit="s", utc=True)
    out["time_local"] = out["time_utc"].dt.tz_convert(LOCAL_TZ)

    report = DedupReport(
        rows_in=len(df),
        rows_out=len(out),
        exact_duplicates=exact,
        conflicting_keys=int(len(conflicts)),
        max_conflict_spread=float(conflicts.max()) if len(conflicts) else 0.0,
    )
    return out, report
