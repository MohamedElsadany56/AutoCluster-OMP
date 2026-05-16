from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PAPER_2D_SIZES = [100_000, 200_000, 500_000]
PAPER_3D_SIZES = [100_000, 200_000, 400_000, 800_000, 1_000_000]


def make_centers(n_clusters: int, n_features: int) -> np.ndarray:
    if n_features == 2:
        angles = np.linspace(0.0, 2.0 * np.pi, n_clusters, endpoint=False)
        radii = np.where(np.arange(n_clusters) % 2 == 0, 18.0, 10.0)
        return np.column_stack([np.cos(angles) * radii, np.sin(angles) * radii])

    grid = np.linspace(-18.0, 18.0, int(np.ceil(n_clusters ** (1.0 / 3.0))) + 1)
    centers = []
    for x in grid:
        for y in grid:
            for z in grid:
                centers.append([x, y, z])
                if len(centers) == n_clusters:
                    return np.asarray(centers, dtype=np.float64)
    return np.asarray(centers[:n_clusters], dtype=np.float64)


def generate_gaussian_mixture(
    n_points: int,
    n_features: int,
    n_clusters: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    centers = make_centers(n_clusters, n_features)
    labels = np.arange(n_points, dtype=np.int32) % n_clusters
    rng.shuffle(labels)

    points = np.empty((n_points, n_features), dtype=np.float64)
    for cluster in range(n_clusters):
        mask = labels == cluster
        count = int(mask.sum())
        scale = 1.0 + 0.15 * (cluster % 5)
        points[mask] = rng.normal(loc=centers[cluster], scale=scale, size=(count, n_features))

    return points, labels


def write_dataset(output_dir: Path, stem: str, points: np.ndarray, labels: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    header = ",".join(f"x{i + 1}" for i in range(points.shape[1]))
    np.savetxt(output_dir / f"{stem}.csv", points, delimiter=",", header=header, comments="", fmt="%.8f")
    np.savetxt(output_dir / f"{stem}_cluster_labels.csv", labels, delimiter=",", fmt="%d")


def parse_sizes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-style Gaussian K-Means datasets.")
    parser.add_argument("--output-dir", default="datasets/paper_gaussian")
    parser.add_argument("--sizes-2d", default=",".join(str(size) for size in PAPER_2D_SIZES))
    parser.add_argument("--sizes-3d", default=",".join(str(size) for size in PAPER_3D_SIZES))
    parser.add_argument("--clusters-2d", type=int, default=8)
    parser.add_argument("--clusters-3d", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20240503)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.seed)

    for n_points in parse_sizes(args.sizes_2d):
        stem = f"paper_2d_{n_points}"
        if not args.overwrite and (output_dir / f"{stem}.csv").exists():
            continue
        points, labels = generate_gaussian_mixture(n_points, 2, args.clusters_2d, rng)
        write_dataset(output_dir, stem, points, labels)
        print(f"Wrote {stem}: {n_points} rows, 2 features, {args.clusters_2d} clusters")

    for n_points in parse_sizes(args.sizes_3d):
        stem = f"paper_3d_{n_points}"
        if not args.overwrite and (output_dir / f"{stem}.csv").exists():
            continue
        points, labels = generate_gaussian_mixture(n_points, 3, args.clusters_3d, rng)
        write_dataset(output_dir, stem, points, labels)
        print(f"Wrote {stem}: {n_points} rows, 3 features, {args.clusters_3d} clusters")


if __name__ == "__main__":
    main()
