# Architecture

AutoCluster-OMP follows a simple source-to-source pipeline:

```text
Sequential C code
-> loop detection
-> pattern classification
-> dependency and safety checks
-> OpenMP pragma generation
-> C compilation
-> benchmarking and checksum validation
-> CSV, JSON, Markdown, and plot outputs
```

The analyzer scans C source lines and matches `for` loops using brace balancing. The classifier recognizes K-Means assignment loops, Fuzzy C-Means membership update loops, shared centroid accumulation loops, possible reductions, and unknown loops.

The generator inserts `#pragma omp parallel for schedule(...)` before loops classified as safe and parallelizable. Unsafe loops are reported but not transformed.

The benchmark runner compiles sequential, manual OpenMP, and auto-generated OpenMP kernels with GCC, runs them with selected thread counts, parses runtime and checksum output, and computes speedup and efficiency.
