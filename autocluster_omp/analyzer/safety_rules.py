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
