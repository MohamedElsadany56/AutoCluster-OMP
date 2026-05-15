from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_efficiency(csv_path: str | Path, output_dir: str | Path) -> Path:
    data = pd.read_csv(csv_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig_path = output / "efficiency.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    for (algorithm, version), group in data.groupby(["algorithm", "version"]):
        group = group.sort_values("threads")
        ax.plot(group["threads"], group["efficiency"], marker="o", label=f"{algorithm} {version}")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Efficiency")
    ax.set_ylim(bottom=0)
    ax.set_title("Parallel Efficiency by Thread Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    return fig_path
