"""Shared pytest fixtures.

The fixture ZIP is **synthetic** and exists only to unit-test the parser.
It is never used as evaluation data.
"""

import csv
import io
import json
import zipfile

import pytest

COLUMNS = ["id", "sensorid", "devicetype", "jsondata", "status",
           "processing_errors", "createdate", "processdate"]


def entry(name, unit, pairs, var_id=1):
    """Build one jsondata variable entry from ``[(time, value), ...]``."""
    return {"values": [{"time": t, "value": v} for t, v in pairs],
            "variable": {"id": var_id, "name": name, "unit": unit}}


def make_row(rid, sid, entries, createdate="2024-02-01 00:00:00"):
    """Build one raw CSV row; ``entries`` may be a list or a raw string."""
    js = entries if isinstance(entries, str) else json.dumps(entries, ensure_ascii=False)
    return {"id": rid, "sensorid": sid, "devicetype": "env", "jsondata": js,
            "status": "Processed", "processing_errors": "",
            "createdate": createdate, "processdate": createdate}


def write_zip(path, rows, member="sample.csv", bom=True):
    """Write rows to a CSV inside a ZIP, like the real archive."""
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=COLUMNS)
    w.writeheader()
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8-sig" if bom else "utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, data)
    return path


T0 = 1_717_200_000  # 2024-06-01 00:00:00 UTC


@pytest.fixture
def sample_zip(tmp_path):
    """Small archive with two sensors, a duplicate, a conflict and bad rows."""
    rows = [
        make_row(1, "A", [entry("Temperature", "°C", [(T0, 20.0)], 68),
                          entry("Humidity", "HR%", [(T0, 50.0)], 69)]),
        # Temperature is second in the array and has a different id: must still be found.
        make_row(2, "A", [entry("Battery", "V", [(T0 + 600, 3.6)], 1),
                          entry("Temperature", "°C", [(T0 + 600, 20.5)], 99)]),
        make_row(3, "A", [entry("Temperature", "°C", [(T0 + 600, 20.5)])]),  # exact duplicate
        make_row(4, "B", [entry("Temperature", "°C", [(T0, 22.0), (T0 + 600, 22.4)])]),
        make_row(5, "B", [entry("Temperature", "°C", [(T0 + 600, 22.6)])]),  # conflict
        make_row(6, "B", "{not valid json"),
        make_row(7, "B", ""),
    ]
    return write_zip(tmp_path / "env.zip", rows)
