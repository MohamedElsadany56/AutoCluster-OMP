from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from autocluster_omp.analyzer import make_workload
from autocluster_omp.benchmark import benchmark_sources, defines_from_config, load_workload_config
from autocluster_omp.config import PROJECT_ROOT, SUPPORTED_ALGORITHMS
from autocluster_omp.generator import write_code, analyze_source, generate_from_file
from autocluster_omp.plotting import plot_efficiency, plot_runtime, plot_speedup
from autocluster_omp.reports import export_csv, export_analysis_json


console = Console()


def _check_algorithm(value: str) -> str:
    if value not in SUPPORTED_ALGORITHMS:
        raise argparse.ArgumentTypeError(f"Unsupported algorithm: {value}")
    return value


def command_analyze(args: argparse.Namespace) -> None:
    source = Path(args.source).read_text(encoding="utf-8")
    analyses = analyze_source(source, args.algorithm, args.schedule)
    workload_info = None
    if args.n_points and args.n_clusters and args.n_features and args.iterations:
        workload = make_workload(args.algorithm, args.n_points, args.n_clusters, args.n_features, args.iterations)
        workload_info = {
            "algorithm": workload.algorithm,
            "n_points": workload.n_points,
            "n_clusters": workload.n_clusters,
            "n_features": workload.n_features,
            "iterations": workload.iterations,
            "estimated_work": workload.estimated_work,
        }
    if args.export_json:
        export_analysis_json(analyses, args.export_json, workload_info)
    table = Table(title="Loop Analysis")
    table.add_column("Lines")
    table.add_column("Classification")
    table.add_column("Decision")
    table.add_column("Safety")
    for item in analyses:
        table.add_row(
            f"{item.loop.start_line}-{item.loop.end_line}",
            item.classification,
            item.decision,
            item.safety,
        )
    console.print(table)


def command_generate(args: argparse.Namespace) -> None:
    generated, analyses = generate_from_file(args.source, args.algorithm, args.schedule, args.chunk_size)
    write_code(generated, args.output)
    if args.export_json:
        export_analysis_json(analyses, args.export_json)
    console.print(f"Wrote generated OpenMP source to {args.output}")


def command_benchmark(args: argparse.Namespace) -> None:
    results = benchmark_sources(
        algorithm=args.algorithm,
        sequential=args.sequential,
        auto=args.auto,
        manual=args.manual,
        threads=args.threads,
        schedule=args.schedule,
        dataset=args.dataset,
        labels=args.labels,
        clusters=args.clusters,
        features=args.features,
        iterations=args.iterations,
        points=args.points,
        repeat=args.repeat,
        fuzziness=args.fuzziness,
    )
    if args.export_csv:
        export_csv(results, args.export_csv)
        console.print(f"Wrote benchmark CSV to {args.export_csv}")
    for result in results:
        console.print(
            f"{result.algorithm} {result.version} t={result.threads}: "
            f"{result.runtime_seconds:.6f}s speedup={result.speedup:.3f} correctness={result.correctness}"
        )


def command_experiment(args: argparse.Namespace) -> None:
    config_path = PROJECT_ROOT / "experiments" / "configs" / f"{args.preset}.json"
    config = load_workload_config(config_path)
    if args.algorithm == "kmeans":
        sequential = PROJECT_ROOT / "c_kernels" / "kmeans" / "kmeans_seq.c"
        manual = PROJECT_ROOT / "c_kernels" / "kmeans" / "kmeans_manual_omp.c"
        auto = PROJECT_ROOT / "c_kernels" / "kmeans" / "kmeans_auto_omp.c"
    else:
        sequential = PROJECT_ROOT / "c_kernels" / "fuzzy_cmeans" / "fcm_seq.c"
        manual = PROJECT_ROOT / "c_kernels" / "fuzzy_cmeans" / "fcm_manual_omp.c"
        auto = PROJECT_ROOT / "c_kernels" / "fuzzy_cmeans" / "fcm_auto_omp.c"

    results = benchmark_sources(
        algorithm=args.algorithm,
        sequential=str(sequential),
        auto=str(auto),
        manual=str(manual),
        threads=config["threads"],
        schedule=config.get("schedule", "static"),
        workload_name=config["name"],
        defines=defines_from_config(config),
    )
    output = PROJECT_ROOT / "results" / "raw" / f"{args.algorithm}_{args.preset}_results.csv"
    export_csv(results, output)
    console.print(f"Wrote experiment results to {output}")


def command_plot(args: argparse.Namespace) -> None:
    plot_runtime(args.csv, args.output_dir)
    plot_speedup(args.csv, args.output_dir)
    plot_efficiency(args.csv, args.output_dir)
    console.print(f"Wrote plots to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autocluster-omp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("source")
    analyze.add_argument("--algorithm", type=_check_algorithm, required=True)
    analyze.add_argument("--schedule", default="static")
    analyze.add_argument("--export-json")
    analyze.add_argument("--n-points", type=int)
    analyze.add_argument("--n-clusters", type=int)
    analyze.add_argument("--n-features", type=int)
    analyze.add_argument("--iterations", type=int)
    analyze.set_defaults(func=command_analyze)

    generate = subparsers.add_parser("generate")
    generate.add_argument("source")
    generate.add_argument("-o", "--output", required=True)
    generate.add_argument("--algorithm", type=_check_algorithm, required=True)
    generate.add_argument("--schedule", default="static")
    generate.add_argument("--chunk-size", type=int)
    generate.add_argument("--export-json")
    generate.set_defaults(func=command_generate)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--algorithm", type=_check_algorithm, required=True)
    benchmark.add_argument("--sequential", required=True)
    benchmark.add_argument("--auto", required=True)
    benchmark.add_argument("--manual", required=True)
    benchmark.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    benchmark.add_argument("--schedule", default="static")
    benchmark.add_argument("--export-csv")
    benchmark.add_argument("--dataset")
    benchmark.add_argument("--labels")
    benchmark.add_argument("--clusters", type=int)
    benchmark.add_argument("--features", type=int)
    benchmark.add_argument("--iterations", type=int)
    benchmark.add_argument("--points", type=int)
    benchmark.add_argument("--fuzziness", type=float, default=2.0)
    benchmark.add_argument("--repeat", type=int, default=1)
    benchmark.set_defaults(func=command_benchmark)

    experiment = subparsers.add_parser("experiment")
    experiment.add_argument("--algorithm", type=_check_algorithm, required=True)
    experiment.add_argument("--preset", choices=["small", "medium", "large"], default="small")
    experiment.set_defaults(func=command_experiment)

    plot = subparsers.add_parser("plot")
    plot.add_argument("csv")
    plot.add_argument("--output-dir", required=True)
    plot.set_defaults(func=command_plot)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
