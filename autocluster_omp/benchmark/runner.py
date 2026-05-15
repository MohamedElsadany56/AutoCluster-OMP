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


def run_binary(binary: Path, threads: int) -> dict[str, float]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    completed = subprocess.run([str(binary)], capture_output=True, text=True, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Program failed: {binary}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return parse_program_output(completed.stdout)


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
) -> list[BenchmarkResult]:
    build = Path(build_dir)
    suffix = ".exe" if os.name == "nt" else ""
    binaries = {
        "sequential": compile_c_source(sequential, build / f"{algorithm}_seq{suffix}", False, defines),
        "manual": compile_c_source(manual, build / f"{algorithm}_manual{suffix}", True, defines),
        "auto": compile_c_source(auto, build / f"{algorithm}_auto{suffix}", True, defines),
    }

    seq_output = run_binary(binaries["sequential"], 1)
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
        )
    ]

    for version in ("manual", "auto"):
        for thread_count in threads:
            output = run_binary(binaries[version], thread_count)
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
                )
            )
    return results
