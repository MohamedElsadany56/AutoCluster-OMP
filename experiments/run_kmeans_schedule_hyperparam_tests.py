from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autocluster_omp.benchmark.runner import benchmark_sources
from autocluster_omp.generator.openmp_generator import generate_from_file
from autocluster_omp.reports.csv_report import export_csv


@dataclass(frozen=True)
class ScheduleConfig:
    schedule: str
    chunk_size: int | None = None

    @property
    def label(self) -> str:
        return self.schedule if self.chunk_size is None else f"{self.schedule}_{self.chunk_size}"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    dataset: Path
    labels: Path | None
    features: int
    clusters: int


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_schedule_list(value: str) -> list[ScheduleConfig]:
    schedules: list[ScheduleConfig] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            schedule, chunk = item.split(":", 1)
            schedules.append(ScheduleConfig(schedule.strip(), int(chunk.strip())))
        else:
            schedules.append(ScheduleConfig(item))
    return schedules


def ensure_covtype_labels() -> Path | None:
    dataset = PROJECT_ROOT / "datasets" / "covtype.csv"
    labels = PROJECT_ROOT / "datasets" / "covtype_labels.csv"
    if labels.exists() or not dataset.exists():
        return labels if labels.exists() else None

    frame = pd.read_csv(dataset, usecols=["Cover_Type"])
    labels.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(labels, index=False, header=False)
    return labels


def ensure_paper_datasets() -> None:
    required = PROJECT_ROOT / "datasets" / "paper_gaussian" / "paper_2d_100000.csv"
    if required.exists():
        return
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "experiments" / "generate_paper_datasets.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )


def dataset_configs(mode: str) -> list[DatasetConfig]:
    configs: list[DatasetConfig] = []
    if mode in {"smoke", "paper-2d", "paper-all", "all"}:
        ensure_paper_datasets()
        paper_dir = PROJECT_ROOT / "datasets" / "paper_gaussian"
        sizes = [100_000] if mode == "smoke" else [100_000, 200_000, 500_000]
        for size in sizes:
            configs.append(
                DatasetConfig(
                    name=f"paper_2d_{size}",
                    dataset=paper_dir / f"paper_2d_{size}.csv",
                    labels=paper_dir / f"paper_2d_{size}_cluster_labels.csv",
                    features=2,
                    clusters=8,
                )
            )

    if mode in {"paper-3d", "paper-all", "all"}:
        ensure_paper_datasets()
        paper_dir = PROJECT_ROOT / "datasets" / "paper_gaussian"
        for size in [100_000, 200_000, 400_000, 800_000, 1_000_000]:
            configs.append(
                DatasetConfig(
                    name=f"paper_3d_{size}",
                    dataset=paper_dir / f"paper_3d_{size}.csv",
                    labels=paper_dir / f"paper_3d_{size}_cluster_labels.csv",
                    features=3,
                    clusters=4,
                )
            )

    if mode in {"covtype", "all"}:
        configs.append(
            DatasetConfig(
                name="covtype",
                dataset=PROJECT_ROOT / "datasets" / "covtype.csv",
                labels=ensure_covtype_labels(),
                features=54,
                clusters=7,
            )
        )

    return configs


def write_auto_source(schedule: ScheduleConfig, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"kmeans_auto_{schedule.label}.c"
    source, _ = generate_from_file(
        str(PROJECT_ROOT / "c_kernels" / "kmeans" / "kmeans_seq.c"),
        "kmeans",
        schedule.schedule,
        schedule.chunk_size,
    )
    shared_dir = (PROJECT_ROOT / "c_kernels" / "shared").as_posix()
    source = source.replace('#include "../shared/checksum_utils.h"', f'#include "{shared_dir}/checksum_utils.h"')
    source = source.replace('#include "../shared/csv_loader.h"', f'#include "{shared_dir}/csv_loader.h"')
    source = source.replace('#include "../shared/dataset_utils.h"', f'#include "{shared_dir}/dataset_utils.h"')
    source = source.replace('#include "../shared/timing_utils.h"', f'#include "{shared_dir}/timing_utils.h"')
    output.write_text(source, encoding="utf-8")
    return output


def plot_schedule_comparison(data: pd.DataFrame, output_dir: Path) -> None:
    auto = data[data["version"] == "auto"].copy()
    if auto.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["runtime_seconds", "speedup", "efficiency"]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for (workload, schedule), group in auto.groupby(["workload_name", "schedule"]):
            group = group.sort_values("threads")
            ax.plot(group["threads"], group[metric], marker="o", label=f"{workload} / {schedule}")
        ax.set_xlabel("Threads")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Auto OpenMP {metric.replace('_', ' ')} by schedule")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output_dir / f"auto_schedule_{metric}.png", dpi=170)
        plt.close(fig)


def plot_hyperparam_summary(data: pd.DataFrame, output_dir: Path) -> None:
    auto8 = data[(data["version"] == "auto") & (data["threads"] == data["threads"].max())].copy()
    if auto8.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [
        f"{row.workload_name}\n{row.schedule}\nk={int(row.clusters)}, it={int(row.iterations)}, r={int(row.repeat)}"
        for row in auto8.itertuples()
    ]
    ax.bar(range(len(auto8)), auto8["runtime_seconds"])
    ax.set_xticks(range(len(auto8)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(f"Auto OpenMP runtime at {int(data['threads'].max())} threads")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "auto_hyperparam_runtime_summary.png", dpi=170)
    plt.close(fig)


def write_manifest(path: Path, args: argparse.Namespace, schedules: list[ScheduleConfig], datasets: list[DatasetConfig]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "value"])
        writer.writerow(["created_at", datetime.now().isoformat(timespec="seconds")])
        writer.writerow(["mode", args.mode])
        writer.writerow(["threads", args.threads])
        writer.writerow(["iterations", args.iterations])
        writer.writerow(["repeats", args.repeats])
        writer.writerow(["cluster_values", args.cluster_values or "dataset defaults"])
        writer.writerow(["schedules", ",".join(schedule.label for schedule in schedules)])
        writer.writerow(["datasets", ",".join(dataset.name for dataset in datasets)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run K-Means schedule and hyperparameter benchmark tests and save CSVs/plots."
    )
    parser.add_argument("--mode", choices=["smoke", "covtype", "paper-2d", "paper-3d", "paper-all", "all"], default="smoke")
    parser.add_argument("--schedules", default="static,dynamic:64,guided:64,auto")
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--iterations", default="10,30")
    parser.add_argument("--repeats", default="1")
    parser.add_argument("--cluster-values", help="Comma-separated cluster counts. Defaults to each dataset's paper/real value.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--auto-only", action="store_true", help="Drop manual OpenMP rows from saved CSVs.")
    args = parser.parse_args()

    schedules = parse_schedule_list(args.schedules)
    threads = parse_int_list(args.threads)
    iterations = parse_int_list(args.iterations)
    repeats = parse_int_list(args.repeats)
    cluster_values = parse_int_list(args.cluster_values) if args.cluster_values else None
    datasets = dataset_configs(args.mode)

    if not datasets:
        raise SystemExit("No datasets selected.")

    run_name = datetime.now().strftime("kmeans_schedule_hyperparam_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "sweeps" / run_name
    raw_dir = output_dir / "raw"
    plot_dir = output_dir / "plots"
    generated_dir = output_dir / "generated_sources"
    build_root = output_dir / "build"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for schedule in schedules:
        auto_source = write_auto_source(schedule, generated_dir)
        for dataset in datasets:
            clusters_to_test = cluster_values or [dataset.clusters]
            for clusters in clusters_to_test:
                for max_iter in iterations:
                    for repeat in repeats:
                        workload_name = f"{dataset.name}_k{clusters}_it{max_iter}_r{repeat}"
                        print(f"Benchmarking {workload_name} with {schedule.label}")
                        results = benchmark_sources(
                            algorithm="kmeans",
                            sequential=str(PROJECT_ROOT / "c_kernels" / "kmeans" / "kmeans_seq.c"),
                            manual=str(PROJECT_ROOT / "c_kernels" / "kmeans" / "kmeans_manual_omp.c"),
                            auto=str(auto_source),
                            threads=threads,
                            schedule=schedule.label,
                            dataset=dataset.dataset,
                            labels=dataset.labels,
                            clusters=clusters,
                            features=dataset.features,
                            iterations=max_iter,
                            repeat=repeat,
                            workload_name=workload_name,
                            build_dir=build_root / schedule.label / workload_name,
                        )
                        if args.auto_only:
                            results = [item for item in results if item.version != "manual"]
                        csv_path = raw_dir / f"{workload_name}_{schedule.label}.csv"
                        export_csv(results, csv_path)
                        all_rows.extend(results)

    aggregate_csv = output_dir / "kmeans_schedule_hyperparam_results.csv"
    export_csv(all_rows, aggregate_csv)

    data = pd.read_csv(aggregate_csv)
    plot_schedule_comparison(data, plot_dir)
    plot_hyperparam_summary(data, plot_dir)
    write_manifest(output_dir / "manifest.csv", args, schedules, datasets)

    print(f"Wrote aggregate results: {aggregate_csv}")
    print(f"Wrote per-run CSVs: {raw_dir}")
    print(f"Wrote plots: {plot_dir}")
    print(f"Wrote generated OpenMP sources: {generated_dir}")


if __name__ == "__main__":
    main()
