import json
from pathlib import Path

from autocluster_omp.benchmark.runner import benchmark_sources


def load_workload_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def defines_from_config(config: dict) -> dict[str, int | float]:
    defines: dict[str, int | float] = {
        "N_POINTS": config["n_points"],
        "N_CLUSTERS": config["n_clusters"],
        "N_FEATURES": config["n_features"],
        "MAX_ITER": config["iterations"],
    }
    if "fuzziness" in config:
        defines["FUZZINESS"] = config["fuzziness"]
    return defines


def run_workload_experiment(
    config_path: str | Path,
    algorithm: str,
    sequential: str,
    auto: str,
    manual: str,
) -> list:
    config = load_workload_config(config_path)
    return benchmark_sources(
        algorithm=algorithm,
        sequential=sequential,
        auto=auto,
        manual=manual,
        threads=config["threads"],
        schedule=config.get("schedule", "static"),
        workload_name=config["name"],
        defines=defines_from_config(config),
    )
