#ifndef AUTOCLUSTER_TIMING_UTILS_H
#define AUTOCLUSTER_TIMING_UTILS_H

#ifdef _OPENMP
#include <omp.h>
#else
#include <time.h>
#endif

static double current_time_seconds(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

#endif
