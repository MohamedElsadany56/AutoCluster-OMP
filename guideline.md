# AutoCluster-OMP Command Guide

Run all commands from the project root:

```powershell
cd D:\AutoCluster-OMP
```

## 1. Verify Python Tests

```powershell
pytest -q
```

## 2. Generate Auto OpenMP Sources

### K-Means

```powershell
python -m autocluster_omp.cli generate c_kernels/kmeans/kmeans_seq.c `
  -o c_kernels/kmeans/kmeans_auto_omp.c `
  --algorithm kmeans `
  --schedule static
```

### Fuzzy C-Means

```powershell
python -m autocluster_omp.cli generate c_kernels/fuzzy_cmeans/fcm_seq.c `
  -o c_kernels/fuzzy_cmeans/fcm_auto_omp.c `
  --algorithm fuzzy-cmeans `
  --schedule static
```

## 3. Compile C Kernels Manually

These commands are useful for direct binary testing outside the Python benchmark runner.

### K-Means Sequential

```powershell
gcc c_kernels/kmeans/kmeans_seq.c -O2 -lm `
  -o results/processed/build/kmeans_seq.exe
```

### K-Means Manual OpenMP

```powershell
gcc c_kernels/kmeans/kmeans_manual_omp.c -O2 -fopenmp -lm `
  -o results/processed/build/kmeans_manual.exe
```

### K-Means Auto OpenMP

```powershell
gcc c_kernels/kmeans/kmeans_auto_omp.c -O2 -fopenmp -lm `
  -o results/processed/build/kmeans_auto.exe
```

### FCM Sequential

```powershell
gcc c_kernels/fuzzy_cmeans/fcm_seq.c -O2 -lm `
  -o results/processed/build/fcm_seq.exe
```

### FCM Manual OpenMP

```powershell
gcc c_kernels/fuzzy_cmeans/fcm_manual_omp.c -O2 -fopenmp -lm `
  -o results/processed/build/fcm_manual.exe
```

### FCM Auto OpenMP

```powershell
gcc c_kernels/fuzzy_cmeans/fcm_auto_omp.c -O2 -fopenmp -lm `
  -o results/processed/build/fcm_auto.exe
```

## 4. Direct Binary Runs

### K-Means Synthetic

```powershell
results/processed/build/kmeans_seq.exe
```

### K-Means CSV Dataset

```powershell
results/processed/build/kmeans_seq.exe `
  --dataset datasets/gaussian/10k_15c_20f_1.csv `
  --clusters 15 `
  --features 20 `
  --iterations 30 `
  --repeat 1
```

### K-Means CSV Dataset With Output Labels

```powershell
results/processed/build/kmeans_auto.exe `
  --dataset datasets/paper_gaussian/paper_2d_500000.csv `
  --clusters 8 `
  --features 2 `
  --iterations 30 `
  --output-labels results/processed/paper_2d_500k_auto_labels.csv
```

### FCM Synthetic

```powershell
results/processed/build/fcm_seq.exe
```

### FCM CSV Dataset

```powershell
results/processed/build/fcm_seq.exe `
  --dataset datasets/gaussian/10k_15c_20f_1.csv `
  --clusters 15 `
  --features 20 `
  --iterations 30 `
  --fuzziness 2.0 `
  --repeat 1
```

## 5. Benchmark K-Means

### Gaussian Dataset

```powershell
python -m autocluster_omp.cli benchmark `
  --algorithm kmeans `
  --sequential c_kernels/kmeans/kmeans_seq.c `
  --manual c_kernels/kmeans/kmeans_manual_omp.c `
  --auto c_kernels/kmeans/kmeans_auto_omp.c `
  --dataset datasets/gaussian/10k_15c_20f_1.csv `
  --labels datasets/gaussian/10k_15c_20f_1_cluster_labels.csv `
  --clusters 15 `
  --features 20 `
  --iterations 30 `
  --repeat 10 `
  --threads 1 2 4 8 `
  --export-csv results/raw/kmeans_10k_15c_20f_1_results.csv
```

### Covtype Dataset

```powershell
python -m autocluster_omp.cli benchmark `
  --algorithm kmeans `
  --sequential c_kernels/kmeans/kmeans_seq.c `
  --manual c_kernels/kmeans/kmeans_manual_omp.c `
  --auto c_kernels/kmeans/kmeans_auto_omp.c `
  --dataset datasets/covtype.csv `
  --labels datasets/covtype_labels.csv `
  --clusters 7 `
  --features 54 `
  --iterations 30 `
  --threads 1 2 4 8 `
  --export-csv results/raw/kmeans_covtype_results.csv
```

## 6. Benchmark Fuzzy C-Means

### Gaussian Dataset

```powershell
python -m autocluster_omp.cli benchmark `
  --algorithm fuzzy-cmeans `
  --sequential c_kernels/fuzzy_cmeans/fcm_seq.c `
  --manual c_kernels/fuzzy_cmeans/fcm_manual_omp.c `
  --auto c_kernels/fuzzy_cmeans/fcm_auto_omp.c `
  --dataset datasets/gaussian/10k_15c_20f_1.csv `
  --labels datasets/gaussian/10k_15c_20f_1_cluster_labels.csv `
  --clusters 15 `
  --features 20 `
  --iterations 30 `
  --fuzziness 2.0 `
  --repeat 10 `
  --threads 1 2 4 8 `
  --export-csv results/raw/fcm_10k_15c_20f_1_results.csv
```

### Covtype Dataset

```powershell
python -m autocluster_omp.cli benchmark `
  --algorithm fuzzy-cmeans `
  --sequential c_kernels/fuzzy_cmeans/fcm_seq.c `
  --manual c_kernels/fuzzy_cmeans/fcm_manual_omp.c `
  --auto c_kernels/fuzzy_cmeans/fcm_auto_omp.c `
  --dataset datasets/covtype.csv `
  --labels datasets/covtype_labels.csv `
  --clusters 7 `
  --features 54 `
  --iterations 10 `
  --fuzziness 2.0 `
  --threads 1 2 4 8 `
  --export-csv results/raw/fcm_covtype_results.csv
```

## 7. Benchmark Different OpenMP Schedules

The schedule is inserted during generation. Generate one auto source per schedule, then benchmark that generated source.

### K-Means Static

```powershell
python -m autocluster_omp.cli generate c_kernels/kmeans/kmeans_seq.c `
  -o c_kernels/kmeans/kmeans_auto_static.c `
  --algorithm kmeans `
  --schedule static
```

### K-Means Dynamic With Chunk Size

```powershell
python -m autocluster_omp.cli generate c_kernels/kmeans/kmeans_seq.c `
  -o c_kernels/kmeans/kmeans_auto_dynamic_64.c `
  --algorithm kmeans `
  --schedule dynamic `
  --chunk-size 64
```

### K-Means Guided With Chunk Size

```powershell
python -m autocluster_omp.cli generate c_kernels/kmeans/kmeans_seq.c `
  -o c_kernels/kmeans/kmeans_auto_guided_64.c `
  --algorithm kmeans `
  --schedule guided `
  --chunk-size 64
```

### FCM Static

```powershell
python -m autocluster_omp.cli generate c_kernels/fuzzy_cmeans/fcm_seq.c `
  -o c_kernels/fuzzy_cmeans/fcm_auto_static.c `
  --algorithm fuzzy-cmeans `
  --schedule static
```

### FCM Dynamic With Chunk Size

```powershell
python -m autocluster_omp.cli generate c_kernels/fuzzy_cmeans/fcm_seq.c `
  -o c_kernels/fuzzy_cmeans/fcm_auto_dynamic_64.c `
  --algorithm fuzzy-cmeans `
  --schedule dynamic `
  --chunk-size 64
```

### FCM Guided With Chunk Size

```powershell
python -m autocluster_omp.cli generate c_kernels/fuzzy_cmeans/fcm_seq.c `
  -o c_kernels/fuzzy_cmeans/fcm_auto_guided_64.c `
  --algorithm fuzzy-cmeans `
  --schedule guided `
  --chunk-size 64
```

## 8. Generate Paper-Style Datasets

This creates the 2D and 3D Gaussian datasets used for scaling comparisons.

```powershell
python experiments/generate_paper_datasets.py --overwrite
```

Generated datasets:

```text
datasets/paper_gaussian/paper_2d_100000.csv
datasets/paper_gaussian/paper_2d_200000.csv
datasets/paper_gaussian/paper_2d_500000.csv
datasets/paper_gaussian/paper_3d_100000.csv
datasets/paper_gaussian/paper_3d_200000.csv
datasets/paper_gaussian/paper_3d_400000.csv
datasets/paper_gaussian/paper_3d_800000.csv
datasets/paper_gaussian/paper_3d_1000000.csv
```

## 9. Run K-Means Dataset Suite

### Covtype

```powershell
python experiments/run_kmeans_dataset_suite.py `
  --suite covtype `
  --threads 1,2,4,8 `
  --iterations 30 `
  --repeat 1 `
  --output results/raw/kmeans_covtype_results.csv
```

### Paper-Style 2D and 3D Datasets

```powershell
python experiments/run_kmeans_dataset_suite.py `
  --suite paper-all `
  --threads 1,2,4,8 `
  --iterations 30 `
  --repeat 1 `
  --output results/raw/kmeans_paper_gaussian_results.csv
```

## 10. Run K-Means Schedule and Hyperparameter Sweep

### Quick Smoke Sweep

```powershell
python experiments/run_kmeans_schedule_hyperparam_tests.py `
  --mode smoke `
  --schedules static,dynamic:64,guided:64,auto `
  --threads 1,2,4,8 `
  --iterations 10 `
  --repeats 1
```

### Covtype Sweep

```powershell
python experiments/run_kmeans_schedule_hyperparam_tests.py `
  --mode covtype `
  --schedules static,dynamic:64,dynamic:256,guided:64,guided:256,auto `
  --threads 1,2,4,8 `
  --iterations 30 `
  --repeats 1 `
  --cluster-values 7
```

### Paper-Style Full Sweep

```powershell
python experiments/run_kmeans_schedule_hyperparam_tests.py `
  --mode paper-all `
  --schedules static,dynamic:64,dynamic:256,guided:64,guided:256,auto `
  --threads 1,2,4,8 `
  --iterations 30 `
  --repeats 1
```

### K-Means Hyperparameter Sweep

```powershell
python experiments/run_kmeans_schedule_hyperparam_tests.py `
  --mode paper-2d `
  --schedules static,dynamic:64,guided:64,auto `
  --threads 1,2,4,8 `
  --iterations 10,30,50 `
  --repeats 1 `
  --cluster-values 4,8,11
```

## 11. Run FCM Schedule and Hyperparameter Sweep

### Quick Smoke Sweep

```powershell
python experiments/run_fcm_schedule_hyperparam_tests.py `
  --mode smoke `
  --schedules static,dynamic:64,guided:64,auto `
  --threads 1,2,4,8 `
  --iterations 2 `
  --fuzziness-values 2.0 `
  --repeats 1
```

### Covtype Sweep

```powershell
python experiments/run_fcm_schedule_hyperparam_tests.py `
  --mode covtype `
  --schedules static,dynamic:64,guided:64,auto `
  --threads 1,2,4,8 `
  --iterations 10 `
  --fuzziness-values 2.0 `
  --cluster-values 7 `
  --repeats 1
```

### Paper-Style FCM Sweep

```powershell
python experiments/run_fcm_schedule_hyperparam_tests.py `
  --mode paper-2d `
  --schedules static,dynamic:64,dynamic:256,guided:64,guided:256,auto `
  --threads 1,2,4,8 `
  --iterations 5,15,30 `
  --fuzziness-values 1.5,2.0,2.5 `
  --repeats 1
```

## 12. Plot Benchmark CSVs

### Generic Plot Command

```powershell
python -m autocluster_omp.cli plot `
  results/raw/kmeans_covtype_results.csv `
  --output-dir results/figures/covtype_kmeans
```

### Multi-Workload Suite Plots

```powershell
python experiments/plot_benchmark_suite.py `
  --csv results/raw/kmeans_paper_gaussian_results.csv `
  --output-dir results/figures/paper_gaussian_kmeans_suite
```

## 13. Plot Cluster Data

### Covtype Source Labels

```powershell
python experiments/plot_cluster_data.py `
  --dataset datasets/covtype.csv `
  --features 54 `
  --label-column Cover_Type `
  --sample 50000 `
  --output results/figures/covtype_kmeans/covtype_cover_type_pca.png `
  --title "Covtype Cover_Type labels"
```

### Paper 2D Source Labels

```powershell
python experiments/plot_cluster_data.py `
  --dataset datasets/paper_gaussian/paper_2d_500000.csv `
  --labels datasets/paper_gaussian/paper_2d_500000_cluster_labels.csv `
  --features 2 `
  --sample 50000 `
  --output results/figures/paper_gaussian_kmeans/paper_2d_500k_source_labels.png `
  --title "Paper-style 2D Gaussian source labels"
```

### K-Means Auto Output Labels

First produce labels:

```powershell
results/processed/build/kmeans_auto.exe `
  --dataset datasets/paper_gaussian/paper_2d_500000.csv `
  --clusters 8 `
  --features 2 `
  --iterations 30 `
  --output-labels results/processed/paper_2d_500k_auto_labels.csv
```

Then plot them:

```powershell
python experiments/plot_cluster_data.py `
  --dataset datasets/paper_gaussian/paper_2d_500000.csv `
  --labels results/processed/paper_2d_500k_auto_labels.csv `
  --features 2 `
  --sample 50000 `
  --output results/figures/paper_gaussian_kmeans/paper_2d_500k_auto_clusters.png `
  --title "Auto OpenMP K-Means clusters, 500k 2D"
```

## 14. Output Locations

Single benchmark CSVs:

```text
results/raw/
```

Generic plots:

```text
results/figures/
```

Schedule and hyperparameter sweeps:

```text
results/sweeps/
```

Each sweep directory contains:

```text
*_schedule_hyperparam_results.csv
manifest.csv
raw/
plots/
generated_sources/
build/
```

## 15. Notes

- K-Means covtype uses `--clusters 7` and `--features 54`.
- FCM is much more expensive than K-Means. Start with low `--iterations` for covtype.
- The label files are metadata and plotting inputs. Correctness validation is still checksum-based: sequential checksum must match manual/auto OpenMP checksum.
- OpenMP schedule changes require regenerating the auto source before benchmarking.
- For large sweeps, prefer running one dataset family first, then expanding to `--mode all`.
