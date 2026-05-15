# Usage

Install the package from the project root:

```bash
pip install -e .
```

Analyze:

```bash
python -m autocluster_omp.cli analyze c_kernels/kmeans/kmeans_seq.c --algorithm kmeans --export-json results/reports/kmeans_analysis.json
```

Generate:

```bash
python -m autocluster_omp.cli generate c_kernels/kmeans/kmeans_seq.c -o c_kernels/kmeans/kmeans_auto_omp.c --algorithm kmeans --schedule static
```

Benchmark:

```bash
python -m autocluster_omp.cli benchmark --algorithm kmeans --sequential c_kernels/kmeans/kmeans_seq.c --auto c_kernels/kmeans/kmeans_auto_omp.c --manual c_kernels/kmeans/kmeans_manual_omp.c --threads 1 2 4 8 --export-csv results/raw/kmeans_results.csv
```

Plot:

```bash
python -m autocluster_omp.cli plot results/raw/kmeans_results.csv --output-dir results/figures
```
