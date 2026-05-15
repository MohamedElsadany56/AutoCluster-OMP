def build_parallel_for_pragma(schedule: str, chunk_size: int | None = None) -> str:
    allowed = {"static", "dynamic", "guided", "auto"}

    if schedule not in allowed:
        raise ValueError(f"Unsupported schedule: {schedule}")

    if schedule in {"dynamic", "guided"} and chunk_size is not None:
        return f"#pragma omp parallel for schedule({schedule}, {chunk_size})"

    return f"#pragma omp parallel for schedule({schedule})"