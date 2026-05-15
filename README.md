# AutoCluster-OMP

AutoCluster-OMP is a lightweight research prototype for source-to-source OpenMP auto-parallelization of sequential C implementations of K-Means and Fuzzy C-Means clustering.

The tool uses explainable pattern matching rather than a full C compiler frontend. It detects clustering loop patterns, classifies loop safety, inserts OpenMP pragmas for safe assignment or membership-update loops, compiles generated code, benchmarks sequential/manual/auto versions, validates checksums, and exports CSV/JSON reports and plots.

## Research Motivation

Many K-Means and fuzzy clustering papers demonstrate manual OpenMP parallelization. AutoCluster-OMP explores a practical gap: a developer-facing workflow that recognizes common clustering kernels and automatically generates OpenMP C code while documenting unsafe loops and scalability limits.

## Supported Algorithms

- K-Means assignment loops: `labels[i] = best_cluster`
- Fuzzy C-Means membership update loops: `membership[i * k + c] = ...`
- Centroid update loops are detected and reported as unsafe unless rewritten with reductions or thread-local buffers.

## Installation

```bash
pip install -e .
```

GCC is required for compilation. On Windows, use MinGW, MSYS2, or WSL.

## CLI Usage

```bash
python -m autocluster_omp.cli analyze c_kernels/kmeans/kmeans_seq.c --algorithm kmeans --export-json results/reports/kmeans_analysis.json
python -m autocluster_omp.cli generate c_kernels/kmeans/kmeans_seq.c -o c_kernels/kmeans/kmeans_auto_omp.c --algorithm kmeans --schedule static
python -m autocluster_omp.cli benchmark --algorithm kmeans --sequential c_kernels/kmeans/kmeans_seq.c --auto c_kernels/kmeans/kmeans_auto_omp.c --manual c_kernels/kmeans/kmeans_manual_omp.c --threads 1 2 4 8 --export-csv results/raw/kmeans_results.csv
python -m autocluster_omp.cli generate c_kernels/fuzzy_cmeans/fcm_seq.c -o c_kernels/fuzzy_cmeans/fcm_auto_omp.c --algorithm fuzzy-cmeans --schedule static
python -m autocluster_omp.cli benchmark --algorithm fuzzy-cmeans --sequential c_kernels/fuzzy_cmeans/fcm_seq.c --auto c_kernels/fuzzy_cmeans/fcm_auto_omp.c --manual c_kernels/fuzzy_cmeans/fcm_manual_omp.c --threads 1 2 4 8 --export-csv results/raw/fcm_results.csv
python -m autocluster_omp.cli plot results/raw/kmeans_results.csv --output-dir results/figures
```

## Expected Output

C kernels print `RuntimeSeconds`, `Checksum`, `N_POINTS`, `N_CLUSTERS`, `N_FEATURES`, and `MAX_ITER`. Fuzzy C-Means also prints `Objective`.

## Limitations

AutoCluster-OMP is intentionally pattern-based. It is not a C compiler, does not fully solve alias analysis, and only supports controlled clustering code patterns. Clang AST integration, richer reduction transformations, GPU generation, and broader algorithm coverage are future work.
