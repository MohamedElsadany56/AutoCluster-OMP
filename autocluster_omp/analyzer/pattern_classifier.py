from autocluster_omp.models import LoopInfo


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
