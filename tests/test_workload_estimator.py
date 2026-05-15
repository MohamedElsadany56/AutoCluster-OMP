from autocluster_omp.analyzer.workload_estimator import estimate_workload
from autocluster_omp.models import WorkloadConfig


def test_workload_estimator_calculates_kmeans_work():
    workload = WorkloadConfig("kmeans", 10, 3, 2, 4)
    assert estimate_workload(workload) == 240


def test_workload_estimator_scales_fcm_by_cluster_ratio_work():
    workload = WorkloadConfig("fuzzy-cmeans", 10, 3, 2, 4)
    assert estimate_workload(workload) == 720
