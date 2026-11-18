"""Command line: python -m tempforecast {fetch,evaluate,forecast}."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import data, evaluate, forecast


def _fetch(args: argparse.Namespace) -> int:
    print("Downloading the Marrakech archive from Open-Meteo…")
    df = data.fetch()
    print(f"  {len(df)} days, {df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}")
    if not args.archive_only:
        recent = data.fetch_recent()
        print(f"  + recent days up to {recent.index.max():%Y-%m-%d} (provisional)")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    df = data.load()
    print(f"Training on days before {args.test_start}, testing on {args.test_start} → {df.index.max():%Y-%m-%d}\n")
    scores = evaluate.evaluate(df, args.test_start)
    table = evaluate.summary(scores, horizon=1)
    print("Tomorrow's forecast, mean absolute error (°C), lower is better:")
    print(table.to_string(float_format=lambda v: f"{v:.2f}"))
    if args.output:
        scores.to_csv(args.output, index=False)
        print(f"\nAll scores saved to {args.output}")
    if args.coverage:
        coverage = evaluate.interval_coverage(df, args.test_start)
        coverage.to_csv(args.coverage, index=False)
        print(f"Uncertainty band coverage saved to {args.coverage}")
    return 0


def _forecast(args: argparse.Namespace) -> int:
    df = data.load()
    trained = forecast.train(df)
    table = forecast.next_days(trained, df)
    last = df.index.max()
    note = " (provisional)" if bool(df["provisional"].iloc[-1]) else ""
    print(f"Marrakech · forecast from the last observed day: {last:%A %d %B %Y}{note}\n")
    for day, row in table.iterrows():
        print(
            f"  {day:%a %d %b}   max {row.tmax:5.1f} °C  [{row.tmax_low:.1f}–{row.tmax_high:.1f}]"
            f"   min {row.tmin:5.1f} °C  [{row.tmin_low:.1f}–{row.tmin_high:.1f}]"
        )
    print("\nBrackets: 80% of real temperatures fall in these ranges.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tempforecast", description="Marrakech temperature forecasting")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="download the latest weather history")
    fetch.add_argument("--archive-only", action="store_true", help="skip the provisional recent days")
    fetch.set_defaults(run=_fetch)

    ev = commands.add_parser("evaluate", help="compare every model on the test years")
    ev.add_argument("--test-start", default=evaluate.TEST_START)
    ev.add_argument("--output", help="save all scores to this CSV")
    ev.add_argument("--coverage", help="also check the 80%% band and save the result to this CSV")
    ev.set_defaults(run=_evaluate)

    fc = commands.add_parser("forecast", help="forecast the next 7 days")
    fc.set_defaults(run=_forecast)

    args = parser.parse_args(argv)
    pd.set_option("display.width", 120)
    try:
        return args.run(args)
    except OSError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
