def checksums_match(reference: float | None, candidate: float | None, tolerance: float = 1e-6) -> bool:
    if reference is None or candidate is None:
        return False
    return abs(reference - candidate) <= tolerance


def correctness_label(reference: float | None, candidate: float | None, tolerance: float = 1e-6) -> str:
    return "pass" if checksums_match(reference, candidate, tolerance) else "fail"
