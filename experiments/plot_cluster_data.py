from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def reduce_to_2d(values: np.ndarray) -> tuple[np.ndarray, str]:
    if values.shape[1] == 2:
        return values, "feature space"

    centered = values - values.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T, "PCA projection"


def load_labels(labels_path: str | None, frame: pd.DataFrame, label_column: str | None) -> np.ndarray | None:
    if labels_path:
        labels = pd.read_csv(labels_path, header=None).iloc[:, 0]
        return labels.to_numpy()
    if label_column:
        return frame[label_column].to_numpy()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a colored sample of a clustering dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--features", type=int, required=True)
    parser.add_argument("--label-column", help="Column name to use as labels, for example Cover_Type.")
    parser.add_argument("--sample", type=int, default=50_000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Cluster data")
    parser.add_argument("--seed", type=int, default=20240503)
    args = parser.parse_args()

    frame = pd.read_csv(args.dataset)
    values = frame.iloc[:, : args.features].to_numpy(dtype=np.float64)
    labels = load_labels(args.labels, frame, args.label_column)

    if labels is not None and len(labels) != len(values):
        raise ValueError("Labels length does not match dataset rows")

    rng = np.random.default_rng(args.seed)
    if len(values) > args.sample:
        indices = rng.choice(len(values), size=args.sample, replace=False)
        values = values[indices]
        if labels is not None:
            labels = labels[indices]

    projected, space_name = reduce_to_2d(values)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    if labels is None:
        ax.scatter(projected[:, 0], projected[:, 1], s=4, alpha=0.55)
    else:
        scatter = ax.scatter(projected[:, 0], projected[:, 1], c=labels, s=4, alpha=0.6, cmap="tab10")
        fig.colorbar(scatter, ax=ax, label="Label")

    ax.set_title(args.title)
    ax.set_xlabel(f"{space_name} axis 1")
    ax.set_ylabel(f"{space_name} axis 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
