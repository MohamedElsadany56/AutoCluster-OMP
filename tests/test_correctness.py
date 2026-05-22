from autocluster_omp.benchmark import checksums_match, correctness_label


def test_correctness_passes_identical_checksums():
    assert checksums_match(1.0, 1.0)
    assert correctness_label(1.0, 1.0) == "pass"


def test_correctness_fails_different_checksums():
    assert not checksums_match(1.0, 2.0)
    assert correctness_label(1.0, 2.0) == "fail"
