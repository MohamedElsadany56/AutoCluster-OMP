import re
from autocluster_omp.models import LoopInfo, WorkloadConfig


def brace_balance(text: str) -> int:
    return text.count("{") - text.count("}")


def detect_for_loops(source: str) -> list[LoopInfo]:
    lines = source.splitlines(keepends=True)
    loops: list[LoopInfo] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if re.search(r"\bfor\s*\(", line):
            start = i
            header = line.strip()
            balance = brace_balance(line)
            j = i

            while balance <= 0 and j + 1 < len(lines):
                j += 1
                balance += brace_balance(lines[j])
                if "{" in lines[j]:
                    break

            while balance > 0 and j + 1 < len(lines):
                j += 1
                balance += brace_balance(lines[j])

            loops.append(
                LoopInfo(
                    start_line=start + 1,
                    end_line=j + 1,
                    header=header,
                    body="".join(lines[start:j + 1]),
                )
            )

            i = j + 1
        else:
            i += 1

    return loops


def check_loop_safety(loop: LoopInfo, classification: str) -> tuple[str, str]:
    """Return a coarse safety label and human-readable reason."""
    body = loop.body

    if classification == "kmeans_assignment_loop" and "labels[i]" in body:
        return "safe", "Each iteration writes labels[i], so iterations update disjoint memory."

    if classification == "fuzzy_membership_update_loop" and "membership[i * k + c]" in body:
        return "safe", "The outer i-loop owns one membership row; writes are disjoint by i and c."

    unsafe_markers = (
        "centroid_sums[",
        "centroid_counts[",
        "counts[",
        "new_centroids[",
    )
    if any(marker in body for marker in unsafe_markers) and ("+=" in body or "++" in body):
        return (
            "unsafe",
            "The loop updates shared centroid accumulation buffers and can race without reductions or local buffers.",
        )

    if "+=" in body:
        return "requires_review", "The loop contains accumulation syntax; inspect whether it is a true reduction."

    return "unknown", "No supported safety rule matched this loop."


def classify_loop(loop: LoopInfo) -> str:
    """Classify controlled C loop patterns used by the prototype kernels."""
    body = loop.body
    compact = " ".join(body.split())

    if "labels[i] = best_cluster" in compact:
        return "kmeans_assignment_loop"

    if (
        "membership[i * k + c] =" in compact
        and resemblant_for_i(loop.header)
        and ("distance_to_centroid" in compact or "1.0 / denominator" in compact)
    ):
        return "fuzzy_membership_update_loop"

    shared_writes = (
        "centroid_sums[",
        "centroid_counts[",
        "new_centroids[",
        "counts[",
    )
    if any(token in body for token in shared_writes) and ("+=" in body or "++" in body):
        return "centroid_update_or_shared_write_loop"

    if "+=" in body or "*=" in body:
        return "possible_reduction_loop"

    return "unknown_loop"


def resemblant_for_i(header: str) -> bool:
    return "for" in header and " i " in f" {header} "


SAFETY_RULES = {
    "kmeans_assignment_loop": (
        "safe",
        "K-Means assignment writes one label per point. Iterations are independent when centroids are read-only.",
    ),
    "fuzzy_membership_update_loop": (
        "safe",
        "FCM membership update writes one membership row per point. Centroids are read-only during the update.",
    ),
    "centroid_update_or_shared_write_loop": (
        "unsafe",
        "Centroid update loops typically accumulate into shared sums or counts and need reductions or local buffers.",
    ),
    "possible_reduction_loop": (
        "requires_review",
        "The loop looks like a reduction. The prototype reports it but does not transform it automatically.",
    ),
    "unknown_loop": (
        "unknown",
        "The loop does not match a supported clustering pattern.",
    ),
}


def explain_classification(classification: str) -> tuple[str, str]:
    return SAFETY_RULES.get(classification, SAFETY_RULES["unknown_loop"])


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
