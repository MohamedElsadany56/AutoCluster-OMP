from pathlib import Path

from autocluster_omp.benchmark.schedule_experiment import run_schedule_experiment
from autocluster_omp.benchmark.workload_experiment import defines_from_config, load_workload_config
from autocluster_omp.config import PROJECT_ROOT
from autocluster_omp.reports.csv_report import export_csv


if __name__ == "__main__":
    config = load_workload_config(PROJECT_ROOT / "experiments" / "configs" / "schedule_comparison.json")
    results = run_schedule_experiment(
        algorithm="kmeans",
        sequential=str(PROJECT_ROOT / "c_kernels" / "kmeans" / "kmeans_seq.c"),
        auto=str(PROJECT_ROOT / "c_kernels" / "kmeans" / "kmeans_auto_omp.c"),
        manual=str(PROJECT_ROOT / "c_kernels" / "kmeans" / "kmeans_manual_omp.c"),
        threads=config["threads"],
        schedules=config["schedules"],
        defines=defines_from_config(config),
    )
    export_csv(results, Path("results/raw/kmeans_schedule_results.csv"))
