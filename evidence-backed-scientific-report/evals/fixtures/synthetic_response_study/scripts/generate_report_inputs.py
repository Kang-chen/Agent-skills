from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import random
from typing import Sequence


def summarize_rows(input_path: Path) -> tuple[dict[str, float], ...]:
    grouped: dict[str, list[float]] = {}
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"sample_id", "group", "response_score"}
        if reader.fieldnames is None or set(reader.fieldnames) != required_columns:
            raise ValueError("Input columns do not match the registered fixture")
        seen_sample_ids: set[str] = set()
        for row in reader:
            sample_id = row["sample_id"]
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id: {sample_id}")
            seen_sample_ids.add(sample_id)
            grouped.setdefault(row["group"], []).append(
                float(row["response_score"])
            )
    if set(grouped) != {"Control", "Treatment"}:
        raise ValueError("Expected Control and Treatment groups")
    summaries: list[dict[str, float]] = []
    for group in ("Control", "Treatment"):
        values = grouped[group]
        # Contract 固定 dict API；group label 作为序列化字段随 numeric summary 返回。
        summaries.append({
            "group": group,  # type: ignore[dict-item]
            "n": len(values),  # type: ignore[dict-item]
            "mean_response_score": sum(values) / len(values),
        })
    return tuple(summaries)


def _write_summary_tables(
    summaries: tuple[dict[str, float], ...],
    output_root: Path,
) -> None:
    table_path = output_root / "tables" / "group_summary.csv"
    plot_path = output_root / "plot_data" / "response_by_group.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (table_path, plot_path):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("group", "n", "mean_response_score"))
            for row in summaries:
                writer.writerow((
                    row["group"],
                    int(row["n"]),
                    f'{row["mean_response_score"]:.4f}',
                ))


def _render_figure(
    summaries: tuple[dict[str, float], ...],
    output_root: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    random.seed(42)
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "pdf.compression": 6,
        "path.simplify": False,
    })
    figure_directory = output_root / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    groups = [str(row["group"]) for row in summaries]
    means = [float(row["mean_response_score"]) for row in summaries]
    figure, axis = plt.subplots(figsize=(5.0, 4.0), dpi=120)
    axis.bar(groups, means, color=("#4C78A8", "#F58518"), width=0.62)
    axis.set_xlabel("Group")
    axis.set_ylabel("Mean response score")
    axis.set_title("Synthetic response by group")
    axis.set_ylim(0.0, 2.0)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    figure.tight_layout()
    stable_time = datetime(1970, 1, 1, tzinfo=timezone.utc)
    figure.savefig(
        figure_directory / "response_by_group.png",
        dpi=120,
        metadata={
            "Software": "evidence-backed-scientific-report-eval",
            "Creation Time": "1970-01-01T00:00:00Z",
        },
    )
    figure.savefig(
        figure_directory / "response_by_group.pdf",
        metadata={
            "Creator": "evidence-backed-scientific-report-eval",
            "Producer": "evidence-backed-scientific-report-eval",
            "CreationDate": stable_time,
            "ModDate": stable_time,
        },
    )
    plt.close(figure)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate registered synthetic report inputs"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--render-figure", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summaries = summarize_rows(args.input)
    _write_summary_tables(summaries, args.output_root)
    if args.render_figure:
        _render_figure(summaries, args.output_root)
    print(f"Generated synthetic report inputs: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
