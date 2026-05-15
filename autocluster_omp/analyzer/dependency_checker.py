from autocluster_omp.models import LoopInfo


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
