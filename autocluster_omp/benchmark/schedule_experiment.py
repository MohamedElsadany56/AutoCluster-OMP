from autocluster_omp.benchmark.runner import benchmark_sources


def run_schedule_experiment(
    algorithm: str,
    sequential: str,
    auto: str,
    manual: str,
    threads: list[int],
    schedules: list[dict],
    defines: dict | None = None,
) -> list:
    results = []
    for schedule in schedules:
        results.extend(
            benchmark_sources(
                algorithm=algorithm,
                sequential=sequential,
                auto=auto,
                manual=manual,
                threads=threads,
                schedule=schedule["name"],
                workload_name="schedule_comparison",
                defines=defines,
            )
        )
    return results
