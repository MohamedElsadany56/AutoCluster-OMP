from autocluster_omp.analyzer.loop_detector import detect_for_loops


def test_loop_detector_detects_loops():
    source = """
void f(void) {
    for (int i = 0; i < n; i++) {
        labels[i] = 0;
    }
}
"""
    loops = detect_for_loops(source)
    assert len(loops) == 1
    assert "for (int i" in loops[0].header
    assert "labels[i]" in loops[0].body
