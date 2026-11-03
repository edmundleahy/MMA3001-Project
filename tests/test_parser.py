"""Verification tests for ingestion: parsing, time zones and duplicates."""

import json
import zipfile

import pandas as pd
import pytest

from src.analysis import coverage_table, sampling_intervals_min
from src.data_processing import (
    ParseStats,
    deduplicate,
    find_csv_member,
    parse_jsondata,
    readings_to_frame,
    stream_readings,
)
from tests.conftest import T0, entry, make_row, write_zip


# ---------- parse_jsondata ----------

def test_selects_variable_by_name_not_id_or_position():
    js = json.dumps([entry("Battery", "V", [(T0, 3.6)], var_id=68),
                     entry("Temperature", "°C", [(T0, 21.3)], var_id=5)])
    assert parse_jsondata(js, ["Temperature"]) == [("Temperature", "°C", T0, 21.3)]


def test_keeps_multiple_values_for_one_variable():
    js = json.dumps([entry("Temperature", "°C", [(T0, 21.0), (T0 + 300, 21.1)])])
    out = parse_jsondata(js, ["Temperature"])
    assert [r[2] for r in out] == [T0, T0 + 300]


def test_none_keeps_all_variables_and_counts_seen():
    stats = ParseStats()
    js = json.dumps([entry("Temperature", "°C", [(T0, 21.0)]),
                     entry("Humidity", "HR%", [(T0, 40.0)])])
    assert len(parse_jsondata(js, None, stats)) == 2
    assert stats.variables_seen[("Humidity", "HR%")] == 1


@pytest.mark.parametrize("text, counter", [
    ("", "bad_json"),
    (None, "bad_json"),
    ("{broken", "bad_json"),
    ('{"a": 1}', "not_a_list"),
])
def test_malformed_json_is_counted_not_raised(text, counter):
    stats = ParseStats()
    assert parse_jsondata(text, ["Temperature"], stats) == []
    assert getattr(stats, counter) == 1


def test_bad_entries_values_and_times_are_counted():
    js = json.dumps([
        {"variable": {"name": "Temperature"}},                       # no values
        {"values": [], "variable": {"unit": "°C"}},                    # no name
        {"values": [{"time": T0, "value": "abc"},                      # non-numeric
                    {"time": T0, "value": None},                       # missing
                    {"time": T0, "value": True},                       # boolean
                    {"time": T0 * 1000, "value": 20.0},                # ms epoch
                    {"value": 20.0},                                   # no time
                    {"time": T0, "value": "20.5"}],                    # numeric string OK
         "variable": {"name": "Temperature", "unit": "°C"}},
    ])
    stats = ParseStats()
    out = parse_jsondata(js, ["Temperature"], stats)
    assert out == [("Temperature", "°C", T0, 20.5)]
    assert stats.bad_entry == 2
    assert stats.bad_value == 3
    assert stats.bad_time == 2


# ---------- time zones ----------

def test_utc_to_melbourne_handles_aest_and_aedt():
    jan = int(pd.Timestamp("2024-01-15 00:00", tz="UTC").timestamp())
    jul = int(pd.Timestamp("2024-07-15 00:00", tz="UTC").timestamp())
    df = readings_to_frame([(1, "A", "Temperature", "°C", jan, 20.0, ""),
                            (2, "A", "Temperature", "°C", jul, 20.0, "")])
    assert str(df["time_local"].dt.tz) == "Australia/Melbourne"
    assert df["time_local"].iloc[0].hour == 11  # AEDT, UTC+11
    assert df["time_local"].iloc[1].hour == 10  # AEST, UTC+10


def test_dst_end_repeated_local_hour_keeps_utc_order():
    # 2024-04-07: clocks go back 03:00 AEDT -> 02:00 AEST, so 02:30 occurs twice.
    first = int(pd.Timestamp("2024-04-06 15:30", tz="UTC").timestamp())   # 02:30 AEDT
    second = int(pd.Timestamp("2024-04-06 16:30", tz="UTC").timestamp())  # 02:30 AEST
    df = readings_to_frame([(1, "A", "Temperature", "°C", first, 1.0, ""),
                            (2, "A", "Temperature", "°C", second, 2.0, "")])
    local = df["time_local"]
    assert local.iloc[0].strftime("%H:%M") == local.iloc[1].strftime("%H:%M") == "02:30"
    assert df["time_utc"].is_monotonic_increasing
    assert (df["time_utc"].iloc[1] - df["time_utc"].iloc[0]) == pd.Timedelta(hours=1)


# ---------- streaming from ZIP ----------

def test_stream_from_zip_end_to_end(sample_zip):
    stats = ParseStats()
    rows = list(stream_readings(sample_zip, ["Temperature"], stats=stats))
    assert stats.rows == 7
    assert stats.bad_json == 2
    assert len(rows) == 6
    assert {r[1] for r in rows} == {"A", "B"}
    assert all(r[2] == "Temperature" for r in rows)


def test_stream_sensor_filter_and_row_limit(sample_zip):
    assert {r[1] for r in stream_readings(sample_zip, sensors=["B"])} == {"B"}
    stats = ParseStats()
    list(stream_readings(sample_zip, stats=stats, limit_rows=2))
    assert stats.rows == 2


def test_utf8_without_bom_also_reads(tmp_path):
    z = write_zip(tmp_path / "nobom.zip",
                  [make_row(1, "A", [entry("Temperature", "°C", [(T0, 19.0)])])], bom=False)
    rows = list(stream_readings(z))
    assert rows[0][1] == "A" and rows[0][3] == "°C"


def test_find_csv_member_rejects_ambiguous_archive(tmp_path):
    p = tmp_path / "two.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("a.csv", "x")
        zf.writestr("b.csv", "x")
    with zipfile.ZipFile(p) as zf, pytest.raises(ValueError):
        find_csv_member(zf)


# ---------- deduplication ----------

def test_dedup_exact_and_conflicting(sample_zip):
    df = readings_to_frame(stream_readings(sample_zip, ["Temperature"]))
    out, rep = deduplicate(df)
    assert rep.rows_in == 6 and rep.rows_out == 4
    assert rep.exact_duplicates == 1
    assert rep.conflicting_keys == 1
    assert rep.max_conflict_spread == pytest.approx(0.2)
    b = out[(out.sensorid == "B") & (out.epoch_s == T0 + 600)]
    assert b["value"].item() == pytest.approx(22.5)  # median of 22.4 and 22.6
    assert b["n_raw"].item() == 2


def test_dedup_is_independent_of_row_order(sample_zip):
    df = readings_to_frame(stream_readings(sample_zip, ["Temperature"]))
    a, _ = deduplicate(df)
    b, _ = deduplicate(df.iloc[::-1].reset_index(drop=True))
    pd.testing.assert_frame_equal(a, b)


# ---------- audit helpers ----------

def test_sampling_intervals_sorted_and_unique():
    iv = sampling_intervals_min(pd.Series([T0 + 1200, T0, T0 + 600, T0 + 600]))
    assert iv.tolist() == [10.0, 10.0]


def test_coverage_table_counts_gaps(sample_zip):
    df, _ = deduplicate(readings_to_frame(stream_readings(sample_zip, ["Temperature"])))
    cov = coverage_table(df).set_index("sensorid")
    assert cov.loc["A", "n_readings"] == 2
    assert cov.loc["A", "median_interval_min"] == 10.0
    assert cov.loc["A", "gaps_over_60min"] == 0
