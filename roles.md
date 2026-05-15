
# AutoCluster-OMP Team Contributions and Presentation Plan

## Project Title

**AutoCluster-OMP: A Lightweight Source-to-Source OpenMP Auto-Parallelizer for K-Means and Fuzzy C-Means Clustering**

---

## Project Overview

AutoCluster-OMP is a research prototype for automatic parallelization in clustering workloads. The tool analyzes sequential C implementations of K-Means and Fuzzy C-Means, detects safe parallel loop patterns, generates OpenMP code automatically, compiles sequential/manual/auto-generated versions, benchmarks them, validates correctness, and exports reproducible results.

The project focuses on two clustering algorithms:

- **K-Means:** hard clustering, where each point belongs to one cluster.
- **Fuzzy C-Means:** soft clustering, where each point has membership degrees across clusters.

The main technical goal is not only to manually parallelize the algorithms, but to build a tool that can automatically detect and transform common clustering loop patterns into OpenMP code.

---

## Team Members

| Name | ID | Email | Section |
|---|---:|---|---|
| Mohamed Yasser Goma | 221001823 | M.yasser2223@nu.edu.eg | 3A |
| Abdullah Ismail | 221001008 | a.yasser2208@nu.edu.eg | 3A |
| Mohamed Khaled | 221000741 | M.Khaled2241@nu.edu.eg | 3A |
| Mohamed Saleh | 221001519 | M.Mahmoud2219@nu.edu.eg | 3A |
| Ahmed Sayed | 211000425 | A.sayed2125@nu.edu.eg | 3B |

---

## Project Pipeline

```text
Sequential C Clustering Code
        ↓
Loop Detection
        ↓
Pattern Classification
        ↓
Safety and Workload Check
        ↓
OpenMP Code Generation
        ↓
Compilation
        ↓
Benchmarking
        ↓
Correctness Validation
        ↓
CSV / JSON Reports
        ↓
Plots and Final Results
````

---

## Main Project Modules

```text
AutoCluster-OMP/
│
├── autocluster_omp/
│   ├── cli.py
│   ├── models.py
│   ├── analyzer/
│   ├── generator/
│   ├── compiler/
│   ├── benchmark/
│   ├── reports/
│   ├── plotting/
│   └── utils/
│
├── c_kernels/
│   ├── kmeans/
│   ├── fuzzy_cmeans/
│   └── shared/
│
├── experiments/
│   └── configs/
│
├── results/
│   ├── raw/
│   ├── reports/
│   └── figures/
│
├── docs/
├── tests/
├── paper/
├── presentation/
├── README.md
├── requirements.txt
└── pyproject.toml
```
```
# Team Contributions

## 1. Mohamed Yasser Goma

### Role

**Core Source-to-Source Transformation and OpenMP Generation Lead**

Mohamed Yasser is responsible for the central transformation pipeline that connects the analyzer with OpenMP code generation. His work focuses on transforming loop-analysis results into valid OpenMP C code.

### Assigned Modules

```text
autocluster_omp/cli.py
autocluster_omp/models.py

autocluster_omp/generator/openmp_generator.py
autocluster_omp/generator/pragma_builder.py
autocluster_omp/generator/code_writer.py
autocluster_omp/generator/kmeans_transformer.py
autocluster_omp/generator/fcm_transformer.py
```

### Assigned Functions

```text
main()
handle_analyze()
handle_generate()

generate_openmp_source()
build_parallel_for_pragma()
write_generated_code()

transform_kmeans_source()
transform_fcm_source()
```

### Main Responsibilities

```text
- Build the source-to-source transformation flow.
- Connect loop analysis results with OpenMP generation.
- Insert OpenMP pragmas before safe loops.
- Preserve C code indentation and structure.
- Support OpenMP schedules: static, dynamic, guided, and auto.
- Generate kmeans_auto_omp.c and fcm_auto_omp.c.
- Ensure CLI generation commands work correctly.
```

### Parallel-Computing Logic Owned

```text
- OpenMP pragma generation.
- Schedule selection.
- Mapping safe loop classifications into OpenMP transformations.
- Ensuring generated code is compilable and valid for OpenMP.
```

---

## 2. Abdullah Ismail

### Role

**Loop Analysis, Safety Classification, and Workload-Aware Parallelization Lead**

Abdullah is responsible for the analysis side of the tool. His work determines which loops are safe to parallelize, which loops are unsafe, and whether the workload is large enough to justify OpenMP generation.

### Assigned Modules

```text
autocluster_omp/analyzer/loop_detector.py
autocluster_omp/analyzer/pattern_classifier.py
autocluster_omp/analyzer/dependency_checker.py
autocluster_omp/analyzer/workload_estimator.py
autocluster_omp/analyzer/safety_rules.py
```

### Assigned Functions

```text
detect_for_loops()
classify_loop()

is_kmeans_assignment_loop()
is_fcm_membership_update_loop()
is_centroid_update_loop()

is_unique_index_write()
has_shared_centroid_update()
has_reduction_pattern()

should_parallelize()
explain_workload_decision()
```

### Main Responsibilities

```text
- Detect C for-loops.
- Classify K-Means assignment loops.
- Classify Fuzzy C-Means membership-update loops.
- Detect unsafe centroid-update loops.
- Detect possible reduction loops.
- Add workload-aware decisions based on estimated computation size.
- Return clear explanations for each loop decision.
```

### Parallel-Computing Logic Owned

```text
- Loop independence analysis.
- Race-condition detection.
- Shared-write detection.
- Workload threshold decision.
- Safe vs unsafe OpenMP transformation rules.
```

---

## 3. Mohamed Khaled

### Role

**Parallel Clustering Kernels and Thread-Local Centroid Update Lead**

Mohamed Khaled is responsible for the C implementations of K-Means and Fuzzy C-Means. His contribution includes both sequential baselines and manual OpenMP implementations, including safe parallel centroid-update logic.

### Assigned Files

```text
c_kernels/kmeans/kmeans_seq.c
c_kernels/kmeans/kmeans_manual_omp.c
c_kernels/kmeans/kmeans_auto_omp.c
c_kernels/kmeans/kmeans_template.c

c_kernels/fuzzy_cmeans/fcm_seq.c
c_kernels/fuzzy_cmeans/fcm_manual_omp.c
c_kernels/fuzzy_cmeans/fcm_auto_omp.c
c_kernels/fuzzy_cmeans/fcm_template.c

c_kernels/shared/dataset_utils.h
c_kernels/shared/timing_utils.h
c_kernels/shared/checksum_utils.h
```

### Assigned Functions

```text
generate_dataset()

initialize_centroids()
assign_clusters()
update_centroids()
update_centroids_parallel_local_buffers()
checksum_labels()

initialize_membership()
normalize_membership()
distance_to_centroid()
update_centroids_fcm()
update_centroids_fcm_parallel_local_buffers()
update_membership()
objective_function()
checksum_membership()
```

### Main Responsibilities

```text
- Implement sequential K-Means.
- Implement manual OpenMP K-Means.
- Implement sequential Fuzzy C-Means.
- Implement manual OpenMP Fuzzy C-Means.
- Implement safe parallel centroid update using thread-local arrays.
- Ensure all C programs compile with GCC.
- Ensure all C programs print RuntimeSeconds and Checksum.
```

### Parallel-Computing Logic Owned

```text
- Manual OpenMP implementation for K-Means.
- Manual OpenMP implementation for Fuzzy C-Means.
- Thread-local centroid accumulation.
- Safe merging of local thread buffers.
- Avoiding race conditions in centroid-update loops.
```

### Important Parallel Functions

```text
update_centroids_parallel_local_buffers()
update_centroids_fcm_parallel_local_buffers()
```

These functions should follow this pattern:

```c
#pragma omp parallel
{
    // local arrays per thread

    #pragma omp for
    for (...) {
        // update local arrays only
    }

    #pragma omp critical
    {
        // merge local arrays into global arrays
    }
}
```

---

## 4. Mohamed Saleh

### Role

**Benchmarking, Parallel Performance Metrics, and Scheduling Experiments Lead**

Mohamed Saleh is responsible for measuring and explaining the performance of sequential, manual OpenMP, and auto-generated OpenMP versions.

### Assigned Modules

```text
autocluster_omp/compiler/command_builder.py
autocluster_omp/compiler/gcc_runner.py

autocluster_omp/benchmark/runner.py
autocluster_omp/benchmark/metrics.py
autocluster_omp/benchmark/correctness.py
autocluster_omp/benchmark/schedule_experiment.py
autocluster_omp/benchmark/workload_experiment.py

autocluster_omp/reports/csv_report.py
autocluster_omp/reports/json_report.py
autocluster_omp/reports/markdown_report.py
autocluster_omp/reports/diagnostics.py
```

### Assigned Functions

```text
build_gcc_command()
compile_c()
has_gcc()

run_binary()
benchmark_algorithm()
benchmark_manual_vs_auto()
run_schedule_experiment()
run_workload_experiment()

calculate_speedup()
calculate_efficiency()
calculate_parallel_overhead()
calculate_scalability_drop()

checksums_match()
correctness_label()

export_benchmark_csv()
export_analysis_json()
generate_markdown_report()
generate_scalability_notes()
```

### Main Responsibilities

```text
- Compile sequential, manual OpenMP, and auto OpenMP versions.
- Run benchmarks using 1, 2, 4, and 8 threads.
- Compare sequential vs manual OpenMP vs auto OpenMP.
- Calculate speedup and efficiency.
- Calculate OpenMP overhead for one-thread runs.
- Run schedule comparison experiments.
- Run small, medium, and large workload experiments.
- Export CSV and JSON results.
- Generate diagnostic notes explaining performance behavior.
```

### Parallel-Computing Logic Owned

```text
- Speedup calculation.
- Parallel efficiency calculation.
- OpenMP overhead analysis.
- Schedule comparison: static vs dynamic vs guided vs auto.
- Scalability analysis across thread counts.
```

### Important Parallel-Performance Functions

```text
calculate_speedup()
calculate_efficiency()
calculate_parallel_overhead()
calculate_scalability_drop()
run_schedule_experiment()
run_workload_experiment()
benchmark_manual_vs_auto()
```

These functions answer:

```text
- Does the generated OpenMP code improve runtime?
- How close is auto OpenMP to manual OpenMP?
- Which schedule performs better?
- Why does efficiency drop at higher thread counts?
- When is the workload large enough to benefit from OpenMP?
```

---

## 5. Ahmed Sayed

### Role

**Experiment Automation, Plotting, Testing, and Reproducibility Lead**

Ahmed is responsible for automating experiments, generating plots, writing tests, and ensuring the project can be reproduced from a clean setup.

### Assigned Files

```text
experiments/run_kmeans_experiment.py
experiments/run_fcm_experiment.py
experiments/run_schedule_experiment.py
experiments/run_all_experiments.py

experiments/configs/small.json
experiments/configs/medium.json
experiments/configs/large.json
experiments/configs/schedule_comparison.json

autocluster_omp/plotting/plot_runtime.py
autocluster_omp/plotting/plot_speedup.py
autocluster_omp/plotting/plot_efficiency.py

tests/test_loop_detector.py
tests/test_pattern_classifier.py
tests/test_workload_estimator.py
tests/test_openmp_generator.py
tests/test_correctness.py
tests/test_metrics.py
tests/test_schedule_experiment.py
```

### Assigned Functions

```text
run_kmeans_experiment()
run_fcm_experiment()
run_schedule_experiment()
run_all_experiments()

load_experiment_config()
run_experiment_from_config()
collect_results()

plot_runtime()
plot_speedup()
plot_efficiency()

test_detect_for_loops()
test_classify_kmeans_assignment_loop()
test_classify_fcm_membership_update_loop()
test_workload_estimator()
test_openmp_generator()
test_correctness_check()
test_speedup_efficiency_metrics()
test_schedule_experiment()
```

### Main Responsibilities

```text
- Create experiment configuration files.
- Automate K-Means experiments.
- Automate Fuzzy C-Means experiments.
- Automate OpenMP schedule experiments.
- Generate plots from CSV results.
- Write pytest tests for the core pipeline.
- Verify reproducibility from a clean setup.
```

### Parallel-Computing Logic Owned

```text
- Workload scaling experiments.
- Thread-count scaling experiments.
- Schedule-policy experiments.
- Runtime/speedup/efficiency visualization.
- Reproducibility testing of parallel results.
```

### Important Experiment Functions

```text
run_experiment_from_config()
run_all_experiments()
plot_speedup()
plot_efficiency()
test_schedule_experiment()
```

These functions provide evidence for:

```text
- Small vs medium vs large workloads.
- 1, 2, 4, and 8 thread scaling.
- Static vs dynamic vs guided scheduling.
- Correctness across generated OpenMP runs.
```

---

# Shared Team Responsibilities

All team members participate in:

```text
- Related work discussion.
- Research gap discussion.
- Final paper writing.
- Presentation preparation.
- Demo preparation.
- Final testing and debugging.
```

---

# Integration Flow

```text
Mohamed Khaled
    builds correct sequential/manual clustering C kernels
        ↓
Abdullah Ismail
    detects and classifies the important loops
        ↓
Mohamed Yasser Goma
    generates OpenMP code from the analysis result
        ↓
Mohamed Saleh
    compiles, benchmarks, validates, and exports results
        ↓
Ahmed Sayed
    automates experiments, plots results, and tests reproducibility
```

---

# Minimum Required Working Pipeline

## K-Means Pipeline

```bash
python -m autocluster_omp.cli analyze c_kernels/kmeans/kmeans_seq.c --algorithm kmeans

python -m autocluster_omp.cli generate c_kernels/kmeans/kmeans_seq.c \
  -o c_kernels/kmeans/kmeans_auto_omp.c \
  --algorithm kmeans \
  --schedule static

python -m autocluster_omp.cli benchmark \
  --algorithm kmeans \
  --sequential c_kernels/kmeans/kmeans_seq.c \
  --manual c_kernels/kmeans/kmeans_manual_omp.c \
  --auto c_kernels/kmeans/kmeans_auto_omp.c \
  --threads 1 2 4 8 \
  --export-csv results/raw/kmeans_results.csv
```

## Fuzzy C-Means Pipeline

```bash
python -m autocluster_omp.cli analyze c_kernels/fuzzy_cmeans/fcm_seq.c --algorithm fuzzy-cmeans

python -m autocluster_omp.cli generate c_kernels/fuzzy_cmeans/fcm_seq.c \
  -o c_kernels/fuzzy_cmeans/fcm_auto_omp.c \
  --algorithm fuzzy-cmeans \
  --schedule static

python -m autocluster_omp.cli benchmark \
  --algorithm fuzzy-cmeans \
  --sequential c_kernels/fuzzy_cmeans/fcm_seq.c \
  --manual c_kernels/fuzzy_cmeans/fcm_manual_omp.c \
  --auto c_kernels/fuzzy_cmeans/fcm_auto_omp.c \
  --threads 1 2 4 8 \
  --export-csv results/raw/fcm_results.csv
```


---

# Short Demo Flow

```text
1. Show the sequential K-Means assignment loop.
2. Run the analyze command.
3. Show that the tool classifies it as kmeans_assignment_loop.
4. Run the generate command.
5. Show the inserted OpenMP pragma.
6. Run the benchmark command.
7. Show runtime, speedup, efficiency, and correctness.
8. Repeat the same idea briefly for Fuzzy C-Means.
9. Show one speedup chart.
```

---

# Recommended Storytelling Flow

```text
We started from a simple observation:
K-Means and Fuzzy C-Means spend most of their time in repeated distance computations.

Previous papers show that these algorithms can be parallelized, but the programmer usually does it manually.

Our idea is to automate part of this process.

So we built AutoCluster-OMP:
a lightweight tool that reads sequential C clustering code, detects known safe loop patterns, inserts OpenMP pragmas, compiles the generated code, benchmarks it, and checks correctness.

K-Means represents hard clustering.
Fuzzy C-Means represents soft clustering.

Our results show that the generated OpenMP version can achieve meaningful speedup while preserving correctness.
```

```
```
