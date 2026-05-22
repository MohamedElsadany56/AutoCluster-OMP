from autocluster_omp.analyzer import detect_for_loops, classify_loop


def test_classifier_detects_kmeans_assignment_loop():
    source = """
for (int i = 0; i < n; i++) {
    int best_cluster = 0;
    labels[i] = best_cluster;
}
"""
    assert classify_loop(detect_for_loops(source)[0]) == "kmeans_assignment_loop"


def test_classifier_detects_fcm_membership_update_loop():
    source = """
for (int i = 0; i < n; i++) {
    for (int c = 0; c < k; c++) {
        double distance_c = distance_to_centroid(data, centroids, i, c, d);
        double denominator = distance_c + 1.0;
        membership[i * k + c] = 1.0 / denominator;
    }
}
"""
    assert classify_loop(detect_for_loops(source)[0]) == "fuzzy_membership_update_loop"
