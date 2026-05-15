from autocluster_omp.generator.openmp_generator import generate_openmp_source
from autocluster_omp.generator.pragma_builder import build_parallel_for_pragma


def test_pragma_builder_generates_schedules():
    assert build_parallel_for_pragma("static") == "#pragma omp parallel for schedule(static)"
    assert build_parallel_for_pragma("dynamic", 64) == "#pragma omp parallel for schedule(dynamic, 64)"
    assert build_parallel_for_pragma("guided", 64) == "#pragma omp parallel for schedule(guided, 64)"
    assert build_parallel_for_pragma("auto") == "#pragma omp parallel for schedule(auto)"


def test_openmp_generator_inserts_pragma():
    source = """
void assign(void) {
    for (int i = 0; i < n; i++) {
        int best_cluster = 0;
        labels[i] = best_cluster;
    }
}
"""
    generated, analyses = generate_openmp_source(source, "kmeans", "static")
    assert "#pragma omp parallel for schedule(static)" in generated
    assert analyses[0].decision == "parallelize"
