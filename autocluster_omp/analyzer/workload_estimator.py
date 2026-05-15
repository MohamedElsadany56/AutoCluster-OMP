from autocluster_omp.models import WorkloadConfig


def estimate_workload(workload: WorkloadConfig) -> int:
    return workload.estimated_work


def make_workload(
    algorithm: str,
    n_points: int,
    n_clusters: int,
    n_features: int,
    iterations: int,
    fuzziness: float = 2.0,
) -> WorkloadConfig:
    return WorkloadConfig(
        algorithm=algorithm,
        n_points=n_points,
        n_clusters=n_clusters,
        n_features=n_features,
        iterations=iterations,
        fuzziness=fuzziness,
    )


def should_parallelize(workload: WorkloadConfig, threshold: int) -> bool:
    if threshold <= 0:
        return True

    return workload.estimated_work >= threshold


def explain_workload_decision(workload: WorkloadConfig, threshold: int) -> str:
    if should_parallelize(workload, threshold):
        return (
            f"Estimated work = {workload.estimated_work:,}. "
            f"This is above threshold {threshold:,}, so OpenMP generation is enabled."
        )

    return (
        f"Estimated work = {workload.estimated_work:,}. "
        f"This is below threshold {threshold:,}, so OpenMP generation is skipped."
    )
