from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_by_threads(data: pd.DataFrame, workload: str, metric: str, output_dir: Path) -> None:
    subset = data[data["workload_name"] == workload]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for version, group in subset.groupby("version"):
        group = group.sort_values("threads")
        ax.plot(group["threads"], group[metric], marker="o", label=version)

    ax.set_xlabel("Threads")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{workload}: {metric.replace('_', ' ')}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / workload / f"{metric}_by_threads.png", dpi=170)
    plt.close(fig)


def workload_size(name: str) -> tuple[str, int] | None:
    match = re.match(r"paper_(2d|3d)_(\d+)", str(name))
    if not match:
        return None
    return match.group(1), int(match.group(2))


def plot_scaling(data: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for _, row in data.iterrows():
        parsed = workload_size(row["workload_name"])
        if not parsed:
            continue
        dimension, n_points = parsed
        if row["version"] == "sequential" or (row["version"] in {"manual", "auto"} and row["threads"] == 8):
            rows.append(
                {
                    "dimension": dimension,
                    "n_points": n_points,
                    "series": row["version"] if row["version"] == "sequential" else f"{row['version']} t=8",
                    "runtime_seconds": row["runtime_seconds"],
                }
            )

    scaling = pd.DataFrame(rows)
    if scaling.empty:
        return

    for dimension, subset in scaling.groupby("dimension"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for series, group in subset.groupby("series"):
            group = group.sort_values("n_points")
            ax.plot(group["n_points"], group["runtime_seconds"], marker="o", label=series)
        ax.set_xlabel("Dataset size N")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title(f"Paper-style {dimension.upper()} scaling")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"scaling_{dimension}.png", dpi=170)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create readable plots for multi-workload benchmark CSVs.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for workload in sorted(data["workload_name"].dropna().unique()):
        workload_dir = output_dir / workload
        workload_dir.mkdir(parents=True, exist_ok=True)
        for metric in ["runtime_seconds", "speedup", "efficiency"]:
            plot_metric_by_threads(data, workload, metric, output_dir)

    plot_scaling(data, output_dir)
    print(f"Wrote suite plots to {output_dir}")


if __name__ == "__main__":
    main()
