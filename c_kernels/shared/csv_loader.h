#ifndef AUTOCLUSTER_CSV_LOADER_H
#define AUTOCLUSTER_CSV_LOADER_H

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct CsvDataset {
    double *points;
    int n_points;
    int n_features;
} CsvDataset;

static char *csv_read_line(FILE *handle) {
    size_t capacity = 256;
    size_t length = 0;
    char *line = (char *)malloc(capacity);
    int ch;

    if (!line) {
        return NULL;
    }

    while ((ch = fgetc(handle)) != EOF) {
        if (length + 1 >= capacity) {
            size_t next_capacity = capacity * 2;
            char *next = (char *)realloc(line, next_capacity);
            if (!next) {
                free(line);
                return NULL;
            }
            line = next;
            capacity = next_capacity;
        }
        line[length++] = (char)ch;
        if (ch == '\n') {
            break;
        }
    }

    if (length == 0 && ch == EOF) {
        free(line);
        return NULL;
    }

    line[length] = '\0';
    return line;
}

static int csv_line_is_empty(const char *line) {
    for (const char *p = line; *p; p++) {
        if (!isspace((unsigned char)*p) && *p != ',') {
            return 0;
        }
    }
    return 1;
}

static int csv_parse_row(const char *line, int n_features, double *out) {
    const char *p = line;

    for (int f = 0; f < n_features; f++) {
        char *end = NULL;

        while (isspace((unsigned char)*p)) {
            p++;
        }

        errno = 0;
        double value = strtod(p, &end);
        if (end == p || errno == ERANGE) {
            return 0;
        }

        if (out) {
            out[f] = value;
        }

        p = end;
        while (isspace((unsigned char)*p)) {
            p++;
        }

        if (f < n_features - 1) {
            if (*p != ',') {
                return 0;
            }
            p++;
        }
    }

    return 1;
}

static int load_csv_dataset(const char *path, int n_features, CsvDataset *dataset) {
    FILE *handle = fopen(path, "r");
    char *line = NULL;
    int saw_data_or_header = 0;
    int skipped_header = 0;
    int rows = 0;
    int capacity = 1024;
    double *points = NULL;

    if (!handle) {
        fprintf(stderr, "Failed to open dataset: %s\n", path);
        return 0;
    }

    points = (double *)malloc((size_t)capacity * n_features * sizeof(double));
    if (!points) {
        fclose(handle);
        fprintf(stderr, "Allocation failed while loading dataset\n");
        return 0;
    }

    while ((line = csv_read_line(handle)) != NULL) {
        if (csv_line_is_empty(line)) {
            free(line);
            continue;
        }

        if (rows >= capacity) {
            int next_capacity = capacity * 2;
            double *next = (double *)realloc(points, (size_t)next_capacity * n_features * sizeof(double));
            if (!next) {
                free(line);
                free(points);
                fclose(handle);
                fprintf(stderr, "Allocation failed while growing dataset\n");
                return 0;
            }
            points = next;
            capacity = next_capacity;
        }

        if (!csv_parse_row(line, n_features, &points[(size_t)rows * n_features])) {
            if (!saw_data_or_header && !skipped_header) {
                skipped_header = 1;
                saw_data_or_header = 1;
                free(line);
                continue;
            }
            fprintf(stderr, "Failed to parse numeric CSV row %d in %s\n", rows + 1, path);
            free(line);
            free(points);
            fclose(handle);
            return 0;
        }

        rows++;
        saw_data_or_header = 1;
        free(line);
    }

    fclose(handle);

    if (rows == 0) {
        free(points);
        fprintf(stderr, "Dataset contains no numeric rows: %s\n", path);
        return 0;
    }

    dataset->points = points;
    dataset->n_points = rows;
    dataset->n_features = n_features;
    return 1;
}

static double *repeat_points(const double *points, int n_points, int n_features, int repeat, int *effective_points) {
    int safe_repeat = repeat > 0 ? repeat : 1;
    size_t row_values = (size_t)n_points * n_features;
    size_t total_values = row_values * (size_t)safe_repeat;
    double *repeated = (double *)malloc(total_values * sizeof(double));

    if (!repeated) {
        fprintf(stderr, "Allocation failed while repeating dataset\n");
        return NULL;
    }

    for (int r = 0; r < safe_repeat; r++) {
        memcpy(&repeated[(size_t)r * row_values], points, row_values * sizeof(double));
    }

    *effective_points = n_points * safe_repeat;
    return repeated;
}

#endif
