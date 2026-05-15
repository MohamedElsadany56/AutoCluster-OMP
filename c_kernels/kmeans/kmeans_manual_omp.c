#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../shared/checksum_utils.h"
#include "../shared/dataset_utils.h"
#include "../shared/timing_utils.h"

#ifndef N_POINTS
#define N_POINTS 50000
#endif
#ifndef N_CLUSTERS
#define N_CLUSTERS 5
#endif
#ifndef N_FEATURES
#define N_FEATURES 3
#endif
#ifndef MAX_ITER
#define MAX_ITER 20
#endif

void generate_dataset(double *data, int n, int d) {
    for (int i = 0; i < n; i++) {
        for (int f = 0; f < d; f++) {
            data[i * d + f] = deterministic_value(i, f);
        }
    }
}

void initialize_centroids(const double *data, double *centroids, int k, int d) {
    for (int c = 0; c < k; c++) {
        for (int f = 0; f < d; f++) {
            centroids[c * d + f] = data[c * d + f];
        }
    }
}

void assign_clusters(const double *data, const double *centroids, int *labels, int n, int k, int d) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int best_cluster = 0;
        double best_distance = DBL_MAX;
        for (int c = 0; c < k; c++) {
            double distance = 0.0;
            for (int f = 0; f < d; f++) {
                double diff = data[i * d + f] - centroids[c * d + f];
                distance += diff * diff;
            }
            if (distance < best_distance) {
                best_distance = distance;
                best_cluster = c;
            }
        }
        labels[i] = best_cluster;
    }
}

void update_centroids(const double *data, const int *labels, double *centroids, int n, int k, int d) {
    double *centroid_sums = (double *)calloc((size_t)k * d, sizeof(double));
    int *centroid_counts = (int *)calloc((size_t)k, sizeof(int));
    if (!centroid_sums || !centroid_counts) {
        fprintf(stderr, "Allocation failed in update_centroids\n");
        exit(1);
    }
    for (int i = 0; i < n; i++) {
        int label = labels[i];
        centroid_counts[label]++;
        for (int f = 0; f < d; f++) {
            centroid_sums[label * d + f] += data[i * d + f];
        }
    }
    for (int c = 0; c < k; c++) {
        if (centroid_counts[c] == 0) {
            continue;
        }
        for (int f = 0; f < d; f++) {
            centroids[c * d + f] = centroid_sums[c * d + f] / (double)centroid_counts[c];
        }
    }
    free(centroid_sums);
    free(centroid_counts);
}

double checksum_labels(const int *labels, int n) {
    return checksum_int_array(labels, n);
}

int main(void) {
    int n = N_POINTS;
    int k = N_CLUSTERS;
    int d = N_FEATURES;
    double *data = (double *)malloc((size_t)n * d * sizeof(double));
    double *centroids = (double *)malloc((size_t)k * d * sizeof(double));
    int *labels = (int *)calloc((size_t)n, sizeof(int));
    if (!data || !centroids || !labels) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    generate_dataset(data, n, d);
    initialize_centroids(data, centroids, k, d);
    double start = current_time_seconds();
    for (int iter = 0; iter < MAX_ITER; iter++) {
        assign_clusters(data, centroids, labels, n, k, d);
        update_centroids(data, labels, centroids, n, k, d);
    }
    double end = current_time_seconds();
    printf("RuntimeSeconds: %.9f\n", end - start);
    printf("Checksum: %.9f\n", checksum_labels(labels, n));
    printf("N_POINTS: %d\n", N_POINTS);
    printf("N_CLUSTERS: %d\n", N_CLUSTERS);
    printf("N_FEATURES: %d\n", N_FEATURES);
    printf("MAX_ITER: %d\n", MAX_ITER);
    free(data);
    free(centroids);
    free(labels);
    return 0;
}
