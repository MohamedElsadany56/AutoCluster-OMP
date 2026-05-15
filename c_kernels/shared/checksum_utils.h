#ifndef AUTOCLUSTER_CHECKSUM_UTILS_H
#define AUTOCLUSTER_CHECKSUM_UTILS_H

static double checksum_int_array(const int *values, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)(values[i] + 1) * (double)(i % 97 + 1);
    }
    return sum;
}

static double checksum_double_array(const double *values, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += values[i] * (double)(i % 97 + 1);
    }
    return sum;
}

#endif
