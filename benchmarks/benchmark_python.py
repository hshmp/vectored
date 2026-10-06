"""Comparable local DataFrame workloads for Pandas and Polars.

Run with the same machine, row count, and trial count as the Dart driver.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any

PATTERN = "category-3"
EDIT_COUNT = 40


def measure(
    fn: Callable[[], Any],
    setup: Callable[[], None],
    *,
    warmups: int,
    trials: int,
    iterations: int,
) -> tuple[float, float]:
    for _ in range(warmups):
        setup()
        fn()

    samples_ms: list[float] = []
    for _ in range(trials):
        setup()
        start = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        samples_ms.append(
            (time.perf_counter_ns() - start) / iterations / 1_000_000
        )
    return statistics.mean(samples_ms), statistics.stdev(samples_ms) if trials > 1 else 0.0


def emit(
    runtime: str,
    library: str,
    operation: str,
    rows: int,
    stats: tuple[float, float],
) -> None:
    print(
        f"{runtime},{library},{operation},{rows},"
        f"{stats[0]:.8g},{stats[1]:.8g}"
    )


def run_pandas(args: argparse.Namespace) -> None:
    import pandas as pd

    labels = [f"event-{row} category-{row % 7}" for row in range(args.rows)]
    values = [row % 1000 for row in range(args.rows)]
    frame = pd.DataFrame({"label": labels, "value": values})
    holder: dict[str, Any] = {"frame": frame}

    def fresh_frame() -> None:
        holder["frame"] = frame

    def mask() -> Any:
        return holder["frame"]["label"].str.contains(PATTERN, regex=False)

    def filtered() -> Any:
        current = holder["frame"]
        return current.loc[current["label"].str.contains(PATTERN, regex=False)]

    def total() -> int:
        return int(holder["frame"]["value"].sum())

    def edited_query() -> float:
        current = holder["frame"]
        checksum = 0.0
        for edit in range(EDIT_COUNT):
            tail = current.iloc[1:]
            inserted = pd.DataFrame(
                {"label": [f"new-{edit}"], "value": [args.rows + edit]}
            )
            current = pd.concat((tail, inserted), ignore_index=True)
            values_column = current["value"]
            checksum += float(values_column.mean())
        holder["frame"] = current
        return checksum

    def fragmented_setup() -> None:
        current = frame
        for edit in range(EDIT_COUNT * 5):
            inserted = pd.DataFrame(
                {
                    "label": [f"new-{edit}"],
                    "value": [args.rows + edit],
                }
            )
            current = pd.concat(
                (current.iloc[1:], inserted),
                ignore_index=True,
            )
        holder["frame"] = current

    expected_mask = [PATTERN in label for label in labels]
    expected_mask_count = sum(expected_mask)
    if mask().tolist() != expected_mask:
        raise RuntimeError("Pandas contains result differs from reference")
    selected = filtered()
    selected_sum = sum(values[i] for i, match in enumerate(expected_mask) if match)
    if (
        len(selected) != expected_mask_count
        or int(selected["value"].sum()) != selected_sum
        or total() != sum(values)
    ):
        raise RuntimeError("Pandas filter or sum result differs from reference")

    for operation, fn, iterations in (
        ("contains-mask", lambda: int(mask().sum()), 1),
        ("filter-materialized", lambda: len(filtered()), 1),
        ("sum", total, 100),
        ("edit-query-40", edited_query, 1),
    ):
        setup = fresh_frame if operation != "edit-query-40" else fresh_frame
        stats = measure(
            fn,
            setup,
            warmups=args.warmups,
            trials=args.trials,
            iterations=iterations,
        )
        emit("python", "pandas", operation, args.rows, stats)
    stats = measure(
        lambda: holder["frame"].copy(deep=True),
        fragmented_setup,
        warmups=args.warmups,
        trials=args.trials,
        iterations=1,
    )
    emit("python", "pandas", "compact-200-edits", args.rows, stats)


def run_polars(args: argparse.Namespace) -> None:
    import polars as pl

    labels = [f"event-{row} category-{row % 7}" for row in range(args.rows)]
    values = [row % 1000 for row in range(args.rows)]
    frame = pl.DataFrame({"label": labels, "value": values})
    holder: dict[str, Any] = {"frame": frame}

    def fresh_frame() -> None:
        holder["frame"] = frame

    def mask_count() -> int:
        return int(
            holder["frame"]
            .select(
                pl.col("label")
                .str.contains(PATTERN, literal=True)
                .sum()
            )
            .item()
        )

    def filtered() -> Any:
        return holder["frame"].filter(
            pl.col("label").str.contains(PATTERN, literal=True)
        )

    def total() -> int:
        return int(holder["frame"].select(pl.col("value").sum()).item())

    def edited_query() -> float:
        current = holder["frame"]
        checksum = 0.0
        for edit in range(EDIT_COUNT):
            inserted = pl.DataFrame(
                {"label": [f"new-{edit}"], "value": [args.rows + edit]}
            )
            current = current.slice(1).vstack(inserted)
            result = current.select(
                pl.col("value").mean()
            ).item()
            checksum += float(result)
        holder["frame"] = current
        return checksum

    def fragmented_setup() -> None:
        current = frame
        for edit in range(EDIT_COUNT * 5):
            inserted = pl.DataFrame(
                {
                    "label": [f"new-{edit}"],
                    "value": [args.rows + edit],
                }
            )
            current = current.slice(1).vstack(inserted)
        holder["frame"] = current

    expected_mask = [PATTERN in label for label in labels]
    expected_mask_count = sum(expected_mask)
    actual_mask = (
        frame.select(
            pl.col("label").str.contains(PATTERN, literal=True)
        )
        .to_series()
        .to_list()
    )
    if actual_mask != expected_mask:
        raise RuntimeError("Polars contains result differs from reference")
    selected = filtered()
    selected_sum = sum(values[i] for i, match in enumerate(expected_mask) if match)
    if (
        selected.height != expected_mask_count
        or selected["value"].sum() != selected_sum
        or total() != sum(values)
    ):
        raise RuntimeError("Polars filter or sum result differs from reference")

    for operation, fn, iterations in (
        ("contains-mask", mask_count, 1),
        ("filter-materialized", lambda: filtered().height, 1),
        ("sum", total, 100),
        ("edit-query-40", edited_query, 1),
    ):
        stats = measure(
            fn,
            fresh_frame,
            warmups=args.warmups,
            trials=args.trials,
            iterations=iterations,
        )
        emit("python", "polars", operation, args.rows, stats)
    stats = measure(
        lambda: holder["frame"].rechunk(),
        fragmented_setup,
        warmups=args.warmups,
        trials=args.trials,
        iterations=1,
    )
    emit("python", "polars", "compact-200-edits", args.rows, stats)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    if min(args.rows, args.trials, args.warmups) <= 0:
        parser.error("rows, trials, and warmups must be positive")

    available = {
        name: importlib.util.find_spec(name) is not None
        for name in ("pandas", "polars")
    }
    missing = [name for name, present in available.items() if not present]
    versions = []
    for name, present in available.items():
        if present:
            module = __import__(name)
            versions.append(f"{name}={module.__version__}")
    if missing:
        print(
            "missing optional benchmark dependencies: " + ", ".join(missing),
            file=sys.stderr,
        )
    print(
        f"Python={sys.version.split()[0]} rows={args.rows} "
        f"trials={args.trials} warmups={args.warmups} "
        + " ".join(versions),
        file=sys.stderr,
    )

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        ["runtime", "library", "operation", "rows", "mean_ms", "stddev_ms"]
    )
    if available["pandas"]:
        run_pandas(args)
    if available["polars"]:
        run_polars(args)
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
