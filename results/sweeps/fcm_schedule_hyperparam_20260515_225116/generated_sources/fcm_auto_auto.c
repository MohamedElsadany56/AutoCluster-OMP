#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "D:/AutoCluster-OMP/c_kernels/shared/checksum_utils.h"
#include "D:/AutoCluster-OMP/c_kernels/shared/csv_loader.h"
#include "D:/AutoCluster-OMP/c_kernels/shared/dataset_utils.h"
#include "D:/AutoCluster-OMP/c_kernels/shared/timing_utils.h"

#ifndef N_POINTS
#define N_POINTS 20000
#endif
#ifndef N_CLUSTERS
#define N_CLUSTERS 5
#endif
#ifndef N_FEATURES
#define N_FEATURES 3
#endif
#ifndef MAX_ITER
#define MAX_ITER 15
#endif
#ifndef FUZZINESS
#define FUZZINESS 2.0
#endif

void generate_dataset(double *data, int n, int d) {
    for (int i = 0; i < n; i++) {
        for (int f = 0; f < d; f++) {
            data[i * d + f] = deterministic_value(i, f);
        }
    }
}

void initialize_membership(double *membership, int n, int k) {
    for (int i = 0; i < n; i++) {
        double total = 0.0;
        for (int c = 0; c < k; c++) {
            double value = (double)(((i + 1) * (c + 3)) % 17 + 1);
            membership[i * k + c] = value;
            total += value;
        }
        for (int c = 0; c < k; c++) {
            membership[i * k + c] /= total;
        }
    }
}

void normalize_membership(double *membership, int n, int k) {
    for (int i = 0; i < n; i++) {
        double total = 0.0;
        for (int c = 0; c < k; c++) {
            total += membership[i * k + c];
        }
        if (total == 0.0) {
            double uniform = 1.0 / (double)k;
            for (int c = 0; c < k; c++) {
                membership[i * k + c] = uniform;
            }
        } else {
            for (int c = 0; c < k; c++) {
                membership[i * k + c] /= total;
            }
        }
    }
}

double distance_to_centroid(const double *data, const double *centroids, int i, int c, int d) {
    double distance = 0.0;
    for (int f = 0; f < d; f++) {
        double diff = data[i * d + f] - centroids[c * d + f];
        distance += diff * diff;
    }
    return sqrt(distance) + 1e-12;
}

void update_centroids_fcm(const double *data, const double *membership, double *centroids, int n, int k, int d, double m) {
    for (int c = 0; c < k; c++) {
        double denominator = 0.0;
        for (int f = 0; f < d; f++) {
            centroids[c * d + f] = 0.0;
        }
        for (int i = 0; i < n; i++) {
            double weight = pow(membership[i * k + c], m);
            denominator += weight;
            for (int f = 0; f < d; f++) {
                centroids[c * d + f] += weight * data[i * d + f];
            }
        }
        if (denominator > 0.0) {
            for (int f = 0; f < d; f++) {
                centroids[c * d + f] /= denominator;
            }
        }
    }
}

void update_membership(const double *data, const double *centroids, double *membership, int n, int k, int d, double m) {
    double exponent = 2.0 / (m - 1.0);
    #pragma omp parallel for schedule(auto)
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < k; c++) {
            double distance_c = distance_to_centroid(data, centroids, i, c, d);
            double denominator = 0.0;
            for (int other = 0; other < k; other++) {
                double distance_other = distance_to_centroid(data, centroids, i, other, d);
                denominator += pow(distance_c / distance_other, exponent);
            }
            membership[i * k + c] = 1.0 / denominator;
        }
    }
}

double objective_function(const double *data, const double *centroids, const double *membership, int n, int k, int d, double m) {
    double objective = 0.0;
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < k; c++) {
            double distance = distance_to_centroid(data, centroids, i, c, d);
            objective += pow(membership[i * k + c], m) * distance * distance;
        }
    }
    return objective;
}

double checksum_membership(const double *membership, int n, int k) {
    return checksum_double_array(membership, n * k);
}

int main(int argc, char **argv) {
    RuntimeConfig config = {NULL, NULL, N_CLUSTERS, N_FEATURES, MAX_ITER, 1, FUZZINESS};
    parse_common_args(argc, argv, &config, 1);

    int n = N_POINTS;
    int k = config.n_clusters;
    int d = config.n_features;
    double m = config.fuzziness;
    double *data = NULL;
    CsvDataset dataset = {NULL, 0, 0};

    if (config.dataset_path) {
        if (!load_csv_dataset(config.dataset_path, d, &dataset)) {
            return 1;
        }
        data = repeat_points(dataset.points, dataset.n_points, dataset.n_features, config.repeat, &n);
        free(dataset.points);
        if (!data) {
            return 1;
        }
    } else {
        data = (double *)malloc((size_t)n * d * sizeof(double));
        if (!data) {
            fprintf(stderr, "Allocation failed\n");
            return 1;
        }
        generate_dataset(data, n, d);
    }

    if (k > n) {
        fprintf(stderr, "N_CLUSTERS must be less than or equal to N_POINTS\n");
        free(data);
        return 1;
    }

    double *membership = (double *)malloc((size_t)n * k * sizeof(double));
    double *centroids = (double *)calloc((size_t)k * d, sizeof(double));
    if (!membership || !centroids) {
        fprintf(stderr, "Allocation failed\n");
        free(data);
        free(membership);
        free(centroids);
        return 1;
    }
    initialize_membership(membership, n, k);

    double start = current_time_seconds();
    for (int iter = 0; iter < config.max_iter; iter++) {
        update_centroids_fcm(data, membership, centroids, n, k, d, m);
        update_membership(data, centroids, membership, n, k, d, m);
        normalize_membership(membership, n, k);
    }
    double objective = objective_function(data, centroids, membership, n, k, d, m);
    double end = current_time_seconds();

    printf("RuntimeSeconds: %.9f\n", end - start);
    printf("Checksum: %.9f\n", checksum_membership(membership, n, k));
    printf("N_POINTS: %d\n", n);
    printf("N_CLUSTERS: %d\n", k);
    printf("N_FEATURES: %d\n", d);
    printf("MAX_ITER: %d\n", config.max_iter);
    printf("DATASET: %s\n", config.dataset_path ? config.dataset_path : "synthetic");
    printf("REPEAT: %d\n", config.repeat);
    printf("Objective: %.9f\n", objective);
    printf("FUZZINESS: %.9f\n", m);

    free(data);
    free(membership);
    free(centroids);
    return 0;
}
