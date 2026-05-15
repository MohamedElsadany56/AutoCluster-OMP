# Related Work Gaps

AutoCluster-OMP is motivated by these gaps:

- Existing K-Means and fuzzy clustering papers often manually parallelize kernels but do not automatically generate OpenMP code.
- Existing work rarely provides one lightweight framework for both hard clustering and soft clustering loop patterns.
- Source-to-source transformation from sequential clustering C code to OpenMP C code is not commonly packaged as a developer-facing workflow.
- Loop safety classification is usually implicit, not exported as a report.
- Race conditions in centroid updates are handled manually rather than surfaced automatically.
- Workload size is known to affect speedup, but workload-aware decisions are often not integrated into tooling.
- Reproducible CSV/JSON benchmark reports are not standardized across clustering studies.
- Automatic explanations of OpenMP overhead, memory bandwidth, synchronization, and sequential bottlenecks are limited.
