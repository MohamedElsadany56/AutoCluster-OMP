from pathlib import Path
import os
import re
import subprocess

from autocluster_omp.benchmark.correctness import correctness_label
from autocluster_omp.benchmark.metrics import calculate_efficiency, calculate_speedup
from autocluster_omp.compiler.gcc_runner import compile_c_source
from autocluster_omp.config import DEFAULT_BUILD_DIR
from autocluster_omp.models import BenchmarkResult


def parse_program_output(output: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for key in ("RuntimeSeconds", "Checksum", "Objective"):
        match = re.search(rf"{key}:\s*([-+0-9.eE]+)", output)
        if match:
            parsed[key] = float(match.group(1))
    return parsed


def run_binary(binary: Path, threads: int, runtime_args: list[str] | None = None) -> dict[str, float]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    command = [str(binary), *(runtime_args or [])]
    completed = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Program failed: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return parse_program_output(completed.stdout)


def build_runtime_args(
    *,
    algorithm: str,
    dataset: str | Path | None = None,
    clusters: int | None = None,
    features: int | None = None,
    iterations: int | None = None,
    repeat: int = 1,
    fuzziness: float | None = None,
) -> list[str]:
    if not dataset:
        return []

    missing = [
        name
        for name, value in (
            ("clusters", clusters),
            ("features", features),
            ("iterations", iterations),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"--dataset requires: {', '.join('--' + item for item in missing)}")

    args = [
        "--dataset",
        str(dataset),
        "--clusters",
        str(clusters),
        "--features",
        str(features),
        "--iterations",
        str(iterations),
        "--repeat",
        str(repeat),
    ]
    if algorithm in {"fuzzy-cmeans", "fcm"}:
        args.extend(["--fuzziness", str(2.0 if fuzziness is None else fuzziness)])
    return args


def validate_dataset_inputs(dataset: str | Path | None, labels: str | Path | None = None) -> None:
    if dataset and not Path(dataset).exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset}")
    if labels and not Path(labels).exists():
        raise FileNotFoundError(f"Labels file does not exist: {labels}")


def benchmark_sources(
    algorithm: str,
    sequential: str,
    auto: str,
    manual: str,
    threads: list[int],
    schedule: str = "static",
    workload_name: str = "default",
    defines: dict[str, int | float | str] | None = None,
    build_dir: str | Path = DEFAULT_BUILD_DIR,
    dataset: str | Path | None = None,
    labels: str | Path | None = None,
    clusters: int | None = None,
    features: int | None = None,
    iterations: int | None = None,
    points: int | None = None,
    repeat: int = 1,
    fuzziness: float | None = None,
) -> list[BenchmarkResult]:
    validate_dataset_inputs(dataset, labels)
    runtime_args = build_runtime_args(
        algorithm=algorithm,
        dataset=dataset,
        clusters=clusters,
        features=features,
        iterations=iterations,
        repeat=repeat,
        fuzziness=fuzziness,
    )

    effective_defines = dict(defines or {})
    if dataset:
        effective_defines["ENABLE_CSV_LOADING"] = 1
    else:
        if points is not None:
            effective_defines["N_POINTS"] = points
        if clusters is not None:
            effective_defines["N_CLUSTERS"] = clusters
        if features is not None:
            effective_defines["N_FEATURES"] = features
        if iterations is not None:
            effective_defines["MAX_ITER"] = iterations
        if fuzziness is not None and algorithm in {"fuzzy-cmeans", "fcm"}:
            effective_defines["FUZZINESS"] = fuzziness

    build = Path(build_dir)
    suffix = ".exe" if os.name == "nt" else ""
    binaries = {
        "sequential": compile_c_source(sequential, build / f"{algorithm}_seq{suffix}", False, effective_defines),
        "manual": compile_c_source(manual, build / f"{algorithm}_manual{suffix}", True, effective_defines),
        "auto": compile_c_source(auto, build / f"{algorithm}_auto{suffix}", True, effective_defines),
    }

    seq_output = run_binary(binaries["sequential"], 1, runtime_args)
    seq_time = seq_output.get("RuntimeSeconds", 0.0)
    seq_checksum = seq_output.get("Checksum")
    results = [
        BenchmarkResult(
            algorithm=algorithm,
            version="sequential",
            threads=1,
            runtime_seconds=seq_time,
            speedup=1.0,
            efficiency=1.0,
            checksum=seq_checksum,
            correctness="reference",
            schedule=schedule,
            workload_name=workload_name,
            dataset=str(dataset) if dataset else None,
            labels=str(labels) if labels else None,
            clusters=clusters,
            features=features,
            iterations=iterations,
            repeat=repeat,
            fuzziness=fuzziness if algorithm in {"fuzzy-cmeans", "fcm"} else None,
            objective=seq_output.get("Objective"),
        )
    ]

    for version in ("manual", "auto"):
        for thread_count in threads:
            output = run_binary(binaries[version], thread_count, runtime_args)
            runtime = output.get("RuntimeSeconds", 0.0)
            speedup = calculate_speedup(seq_time, runtime)
            results.append(
                BenchmarkResult(
                    algorithm=algorithm,
                    version=version,
                    threads=thread_count,
                    runtime_seconds=runtime,
                    speedup=speedup,
                    efficiency=calculate_efficiency(speedup, thread_count),
                    checksum=output.get("Checksum"),
                    correctness=correctness_label(seq_checksum, output.get("Checksum"), tolerance=1e-3),
                    schedule=schedule,
                    workload_name=workload_name,
                    dataset=str(dataset) if dataset else None,
                    labels=str(labels) if labels else None,
                    clusters=clusters,
                    features=features,
                    iterations=iterations,
                    repeat=repeat,
                    fuzziness=fuzziness if algorithm in {"fuzzy-cmeans", "fcm"} else None,
                    objective=output.get("Objective"),
                )
            )
    return results
