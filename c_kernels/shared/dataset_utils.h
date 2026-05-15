#ifndef AUTOCLUSTER_DATASET_UTILS_H
#define AUTOCLUSTER_DATASET_UTILS_H

static double deterministic_value(int point, int feature) {
    int cluster_hint = point % 8;
    double base = (double)cluster_hint * 7.5;
    double wave = (double)((point * 17 + feature * 31) % 100) / 100.0;
    return base + wave + (double)feature * 0.25;
}

#endif
