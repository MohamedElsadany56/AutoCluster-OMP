def calculate_speedup(sequential_time: float, parallel_time: float) -> float:
    if parallel_time <= 0:
        return 0.0
    return sequential_time / parallel_time


def calculate_efficiency(speedup: float, threads: int) -> float:
    if threads <= 0:
        return 0.0
    return speedup / threads
