"""Command-line entry point.

Run from the repository root, e.g. (Windows PowerShell)::

    python -m src.main audit --zip data/raw/5EnvSensor_MayToDec2024_180kRows.zip

Sub-commands:

* ``audit`` - stream the raw ZIP, extract the chosen variables, deduplicate,
  write coverage tables/figures to ``results/audit/`` and a tidy CSV to
  ``data/processed/``.
* ``select`` - pick the lag count and run sensitivity studies, using training
  and validation data only.
* ``final`` - evaluate the locked settings on the test period, once.
* ``profile`` - run ``select`` under cProfile and save the slowest functions.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

from .analysis import write_audit
from .data_processing import ParseStats, deduplicate, readings_to_frame, stream_readings
from .evaluate import run_final, run_select

DEFAULT_ZIP = Path("data/raw/5EnvSensor_MayToDec2024_180kRows.zip")
DEFAULT_READINGS = Path("data/processed/readings.csv")
DEFAULT_SENSOR = "6012002000869"
FORECAST_DIR = Path("results/forecast")


def cmd_audit(args: argparse.Namespace) -> None:
    """Run the data audit (see module docstring)."""
    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(
            f"Raw ZIP not found: {zip_path}\n"
            "Copy the authorised dataset into data/raw/ (it is git-ignored)."
        )

    stats = ParseStats()
    t0 = time.perf_counter()
    created = []  # (min, max) createdate tracked without storing every row

    def tracked():
        lo = hi = None
        for r in stream_readings(zip_path, variables=args.variables,
                                 sensors=args.sensors, stats=stats,
                                 limit_rows=args.limit_rows):
            c = r[6]
            if c:
                lo = c if lo is None or c < lo else lo
                hi = c if hi is None or c > hi else hi
            yield r
        created.extend([lo, hi])

    raw = readings_to_frame(tracked())
    t_parse = time.perf_counter() - t0
    df, dedup = deduplicate(raw)

    out_dir = Path(args.out)
    summary = write_audit(df, stats, dedup, (created[0], created[1]), out_dir)

    proc = Path(args.processed)
    proc.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["time_local"]).to_csv(proc, index=False)

    print(f"Parsed {stats.rows:,} CSV rows in {t_parse:.1f} s")
    print(f"Parse counters: {summary['parse']}")
    print(f"Dedup: {summary['dedup']}")
    print(f"Event-time range (UTC): {summary['event_time_utc_range']}")
    print(f"createdate range (raw text): {summary['createdate_range_raw']}")
    print(f"Audit written to {out_dir.resolve()}")
    print(f"Tidy readings written to {proc.resolve()}")


def _require_readings(path: Path) -> None:
    """Stop with a clear message if the audit has not produced the tidy table."""
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `python -m src.main audit` first.")


def cmd_select(args: argparse.Namespace) -> None:
    """Choose settings on validation data and print the outcome."""
    _require_readings(Path(args.readings))
    out = FORECAST_DIR / args.sensor
    selection = run_select(args.readings, args.sensor, out, args.step, args.horizon)
    print(f"Selected n_lags={selection['n_lags']}. Files written to {out.resolve()}")


def cmd_final(args: argparse.Namespace) -> None:
    """Evaluate locked settings on the test period and print the metrics."""
    _require_readings(Path(args.readings))
    selection = FORECAST_DIR / (args.settings_from or args.sensor) / "selection.json"
    if not selection.exists():
        raise SystemExit(f"{selection} not found. Run `python -m src.main select` first.")
    out = FORECAST_DIR / args.sensor
    try:
        r = run_final(args.readings, args.sensor, out, selection, reproduce=args.reproduce)
    except FileExistsError as err:
        raise SystemExit(str(err)) from err
    print(f"Test MAE (°C): persistence {r['persistence']['mae']:.3f}, "
          f"regression {r['regression']['mae']:.3f}. Files written to {out.resolve()}")


def cmd_profile(args: argparse.Namespace) -> None:
    """Profile the validation-only stage and save the 25 slowest functions."""
    _require_readings(Path(args.readings))
    out = FORECAST_DIR / args.sensor
    profiler = cProfile.Profile()
    profiler.runcall(run_select, args.readings, args.sensor, out, args.step, args.horizon)
    text = io.StringIO()
    pstats.Stats(profiler, stream=text).sort_stats("cumulative").print_stats(25)
    path = out / "profile_select.txt"
    path.write_text(text.getvalue(), encoding="utf-8")
    print(f"Profile written to {path.resolve()}")


def _add_forecast_args(parser: argparse.ArgumentParser) -> None:
    """Add the arguments shared by every modelling sub-command."""
    parser.add_argument("--readings", default=str(DEFAULT_READINGS), help="Tidy readings CSV")
    parser.add_argument("--sensor", default=DEFAULT_SENSOR, help="Sensor ID")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for all sub-commands."""
    p = argparse.ArgumentParser(prog="python -m src.main",
                                description="MMA3001 indoor-temperature forecasting")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="Audit raw environmental data coverage")
    a.add_argument("--zip", default=str(DEFAULT_ZIP), help="Path to raw env-sensor ZIP")
    a.add_argument("--variables", nargs="+",
                   default=["Temperature", "Humidity", "Carbon dioxide"],
                   help="Variable names to extract (exact names, quote multi-word names)")
    a.add_argument("--sensors", nargs="+", default=None, help="Restrict to these sensor IDs")
    a.add_argument("--limit-rows", type=int, default=None,
                   help="Only read the first N CSV rows (quick check)")
    a.add_argument("--out", default="results/audit", help="Audit output folder")
    a.add_argument("--processed", default="data/processed/readings.csv",
                   help="Where to write the tidy deduplicated readings")
    a.set_defaults(func=cmd_audit)

    for name, func, text in (("select", cmd_select, "Choose settings on validation data"),
                             ("profile", cmd_profile, "Profile the select stage")):
        s = sub.add_parser(name, help=text)
        _add_forecast_args(s)
        s.add_argument("--step", type=int, default=20, help="Grid step in minutes")
        s.add_argument("--horizon", type=int, default=60, help="Forecast horizon in minutes")
        s.set_defaults(func=func)

    f = sub.add_parser("final", help="Evaluate locked settings on the test period, once")
    _add_forecast_args(f)
    f.add_argument("--settings-from", default=None,
                   help="Sensor whose selection.json holds the locked settings (default: --sensor)")
    f.add_argument("--reproduce", action="store_true",
                   help="Regenerate an existing test result with the same locked settings")
    f.set_defaults(func=cmd_final)
    return p


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the chosen sub-command."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
