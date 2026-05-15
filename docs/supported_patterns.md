# Supported Patterns

## K-Means Assignment Loop

Supported loops write one label per point:

```c
for (int i = 0; i < n; i++) {
    ...
    labels[i] = best_cluster;
}
```

This is safe because centroids are read-only during assignment and each iteration writes a distinct `labels[i]`.

## Fuzzy C-Means Membership Update Loop

Supported loops write one membership row per point:

```c
for (int i = 0; i < n; i++) {
    for (int c = 0; c < k; c++) {
        ...
        membership[i * k + c] = ...;
    }
}
```

This is safe because each outer iteration owns point `i`.

## Unsafe Centroid Update Loop

Centroid accumulation patterns such as `centroid_sums[...] += ...` and `centroid_counts[...]++` are reported as unsafe because concurrent iterations may update the same cluster accumulator.
