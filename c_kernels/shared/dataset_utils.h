#ifndef AUTOCLUSTER_DATASET_UTILS_H
#define AUTOCLUSTER_DATASET_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct RuntimeConfig {
    const char *dataset_path;
    int n_clusters;
    int n_features;
    int max_iter;
    int repeat;
    double fuzziness;
} RuntimeConfig;

static double deterministic_value(int point, int feature) {
    int cluster_hint = point % 8;
    double base = (double)cluster_hint * 7.5;
    double wave = (double)((point * 17 + feature * 31) % 100) / 100.0;
    return base + wave + (double)feature * 0.25;
}

static int parse_int_arg(const char *value, const char *name) {
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (!value || *value == '\0' || !end || *end != '\0' || parsed <= 0) {
        fprintf(stderr, "Invalid positive integer for %s: %s\n", name, value ? value : "");
        exit(1);
    }
    return (int)parsed;
}

static double parse_double_arg(const char *value, const char *name) {
    char *end = NULL;
    double parsed = strtod(value, &end);
    if (!value || *value == '\0' || !end || *end != '\0' || parsed <= 0.0) {
        fprintf(stderr, "Invalid positive number for %s: %s\n", name, value ? value : "");
        exit(1);
    }
    return parsed;
}

static void parse_common_args(int argc, char **argv, RuntimeConfig *config, int allow_fuzziness) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dataset") == 0 && i + 1 < argc) {
            config->dataset_path = argv[++i];
        } else if (strcmp(argv[i], "--clusters") == 0 && i + 1 < argc) {
            config->n_clusters = parse_int_arg(argv[++i], "--clusters");
        } else if (strcmp(argv[i], "--features") == 0 && i + 1 < argc) {
            config->n_features = parse_int_arg(argv[++i], "--features");
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            config->max_iter = parse_int_arg(argv[++i], "--iterations");
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            config->repeat = parse_int_arg(argv[++i], "--repeat");
        } else if (allow_fuzziness && strcmp(argv[i], "--fuzziness") == 0 && i + 1 < argc) {
            config->fuzziness = parse_double_arg(argv[++i], "--fuzziness");
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            exit(1);
        }
    }
}

#endif
