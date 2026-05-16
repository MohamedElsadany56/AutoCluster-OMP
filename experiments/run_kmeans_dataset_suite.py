from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autocluster_omp.benchmark.runner import benchmark_sources
from autocluster_omp.reports.csv_report import export_csv


def parse_threads(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def paper_jobs(suite: str) -> list[dict]:
    jobs = []
    if suite in {"paper-2d", "paper-all"}:
        for n_points in [100_000, 200_000, 500_000]:
            jobs.append(
                {
                    "name": f"paper_2d_{n_points}",
                    "dataset": PROJECT_ROOT / "datasets" / "paper_gaussian" / f"paper_2d_{n_points}.csv",
                    "labels": PROJECT_ROOT / "datasets" / "paper_gaussian" / f"paper_2d_{n_points}_cluster_labels.csv",
                    "clusters": 8,
                    "features": 2,
                }
            )
    if suite in {"paper-3d", "paper-all"}:
        for n_points in [100_000, 200_000, 400_000, 800_000, 1_000_000]:
            jobs.append(
                {
                    "name": f"paper_3d_{n_points}",
                    "dataset": PROJECT_ROOT / "datasets" / "paper_gaussian" / f"paper_3d_{n_points}.csv",
                    "labels": PROJECT_ROOT / "datasets" / "paper_gaussian" / f"paper_3d_{n_points}_cluster_labels.csv",
                    "clusters": 4,
                    "features": 3,
                }
            )
    return jobs


def covtype_job() -> dict:
    labels = PROJECT_ROOT / "datasets" / "covtype_labels.csv"
    return {
        "name": "covtype",
        "dataset": PROJECT_ROOT / "datasets" / "covtype.csv",
        "labels": labels if labels.exists() else None,
        "clusters": 7,
        "features": 54,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run K-Means benchmarks for real and paper-style datasets.")
    parser.add_argument("--suite", choices=["covtype", "paper-2d", "paper-3d", "paper-all"], required=True)
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    jobs = [covtype_job()] if args.suite == "covtype" else paper_jobs(args.suite)
    results = []
    for job in jobs:
        print(f"Benchmarking {job['name']}")
        results.extend(
            benchmark_sources(
                algorithm="kmeans",
                sequential=str(PROJECT_ROOT / "c_kernels" / "kmeans" / "kmeans_seq.c"),
                manual=str(PROJECT_ROOT / "c_kernels" / "kmeans" / "kmeans_manual_omp.c"),
                auto=str(PROJECT_ROOT / "c_kernels" / "kmeans" / "kmeans_auto_omp.c"),
                threads=parse_threads(args.threads),
                dataset=job["dataset"],
                labels=job["labels"],
                clusters=job["clusters"],
                features=job["features"],
                iterations=args.iterations,
                repeat=args.repeat,
                workload_name=job["name"],
                build_dir=PROJECT_ROOT / "results" / "processed" / "build" / job["name"],
            )
        )

    output = export_csv(results, args.output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
