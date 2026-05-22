from autocluster_omp.benchmark import calculate_efficiency, calculate_speedup


def test_metrics_calculate_speedup_and_efficiency():
    speedup = calculate_speedup(10.0, 2.5)
    assert speedup == 4.0
    assert calculate_efficiency(speedup, 8) == 0.5
