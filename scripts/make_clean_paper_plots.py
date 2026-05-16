from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_csv(filename: str) -> Path:
    """
    Search for a CSV file in common result locations.
    """
    candidates = [
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "results" / "raw" / filename,
        PROJECT_ROOT / "results" / "processed" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = list(PROJECT_ROOT.rglob(filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find CSV file: {filename}")


def normalize_versions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize version names if different scripts used different labels.
    """
    df = df.copy()

    if "version" not in df.columns:
        raise ValueError("CSV must contain a 'version' column.")

    df["version"] = df["version"].astype(str).str.lower()
    df["version"] = df["version"].replace(
        {
            "auto_openmp": "auto",
            "manual_openmp": "manual",
            "seq": "sequential",
            "serial": "sequential",
        }
    )

    return df


def plot_speedup_manual_vs_auto(
    csv_file: str,
    output_name: str,
    title: str,
) -> None:
    """
    Plot speedup vs threads for manual OpenMP and auto-generated OpenMP.
    This is useful for showing how close auto OpenMP is to manual OpenMP.
    """
    csv_path = find_csv(csv_file)
    df = normalize_versions(pd.read_csv(csv_path))

    required = {"version", "threads", "speedup"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_file} is missing columns: {missing}")

    df = df[df["version"].isin(["manual", "auto"])].copy()
    df["threads"] = df["threads"].astype(int)
    df = df.sort_values(["version", "threads"])

    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    label_map = {
        "manual": "Manual OpenMP",
        "auto": "Auto-generated OpenMP",
    }

    for version, group in df.groupby("version"):
        group = group.sort_values("threads")
        ax.plot(
            group["threads"],
            group["speedup"],
            marker="o",
            linewidth=2,
            label=label_map.get(version, version),
        )

    ax.set_title(title)
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Speedup")
    ax.set_xticks(sorted(df["threads"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_runtime_manual_vs_auto(
    csv_file: str,
    output_name: str,
    title: str,
) -> None:
    """
    Optional readable runtime plot for one selected workload.
    """
    csv_path = find_csv(csv_file)
    df = normalize_versions(pd.read_csv(csv_path))

    required = {"version", "threads", "runtime_seconds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_file} is missing columns: {missing}")

    df = df[df["version"].isin(["manual", "auto"])].copy()
    df["threads"] = df["threads"].astype(int)
    df = df.sort_values(["version", "threads"])

    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    label_map = {
        "manual": "Manual OpenMP",
        "auto": "Auto-generated OpenMP",
    }

    for version, group in df.groupby("version"):
        group = group.sort_values("threads")
        ax.plot(
            group["threads"],
            group["runtime_seconds"],
            marker="o",
            linewidth=2,
            label=label_map.get(version, version),
        )

    ax.set_title(title)
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_xticks(sorted(df["threads"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_fcm_schedule_summary(
    csv_file: str = "fcm_schedule_hyperparam_results.csv",
) -> None:
    """
    Summarize the whole FCM schedule sweep into readable bar charts.
    Instead of plotting hundreds of lines, this groups results by schedule
    and plots median speedup/runtime at 8 threads.
    """
    csv_path = find_csv(csv_file)
    df = normalize_versions(pd.read_csv(csv_path))

    required = {"version", "threads", "schedule", "runtime_seconds", "speedup"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_file} is missing columns: {missing}")

    df["threads"] = df["threads"].astype(int)

    target_threads = 8 if 8 in set(df["threads"]) else int(df["threads"].max())

    auto = df[
        (df["version"] == "auto")
        & (df["threads"] == target_threads)
    ].copy()

    if auto.empty:
        raise ValueError("No auto OpenMP rows found for schedule summary.")

    summary = (
        auto.groupby("schedule", as_index=False)
        .agg(
            median_runtime=("runtime_seconds", "median"),
            mean_runtime=("runtime_seconds", "mean"),
            median_speedup=("speedup", "median"),
            max_speedup=("speedup", "max"),
        )
        .sort_values("median_speedup", ascending=False)
    )

    summary_csv = OUTPUT_DIR / "table_fcm_schedule_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(summary["schedule"], summary["median_speedup"])
    ax.set_title(f"FCM Auto OpenMP Median Speedup by Schedule ({target_threads} Threads)")
    ax.set_xlabel("OpenMP Schedule")
    ax.set_ylabel("Median Speedup")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "fig_fcm_schedule_median_speedup.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")

    summary_runtime = summary.sort_values("median_runtime", ascending=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(summary_runtime["schedule"], summary_runtime["median_runtime"])
    ax.set_title(f"FCM Auto OpenMP Median Runtime by Schedule ({target_threads} Threads)")
    ax.set_xlabel("OpenMP Schedule")
    ax.set_ylabel("Median Runtime (seconds)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "fig_fcm_schedule_median_runtime.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_kmeans_gaussian_summary(
    csv_file: str = "kmeans_paper_gaussian_results.csv",
) -> None:
    """
    Optional compact K-Means Gaussian plot.
    Shows auto OpenMP speedup at 8 threads across Gaussian workloads.
    """
    csv_path = find_csv(csv_file)
    df = normalize_versions(pd.read_csv(csv_path))

    required = {"version", "threads", "workload_name", "speedup"}
    missing = required - set(df.columns)
    if missing:
        print(f"Skipping Gaussian summary. Missing columns in {csv_file}: {missing}")
        return

    df["threads"] = df["threads"].astype(int)
    target_threads = 8 if 8 in set(df["threads"]) else int(df["threads"].max())

    auto = df[
        (df["version"] == "auto")
        & (df["threads"] == target_threads)
    ].copy()

    if auto.empty:
        print("Skipping Gaussian summary. No auto rows found.")
        return

    name_map = {
        "paper_2d_100000": "G2D-100K",
        "paper_2d_200000": "G2D-200K",
        "paper_2d_500000": "G2D-500K",
        "paper_3d_100000": "G3D-100K",
        "paper_3d_200000": "G3D-200K",
        "paper_3d_400000": "G3D-400K",
        "paper_3d_800000": "G3D-800K",
        "paper_3d_1000000": "G3D-1M",
    }
    order = [
    "G2D-100K",
    "G2D-200K",
    "G2D-500K",
    "G3D-100K",
    "G3D-200K",
    "G3D-400K",
    "G3D-800K",
    "G3D-1M",]

    auto["paper_workload"] = auto["workload_name"].replace(name_map)
    auto["paper_workload"] = pd.Categorical(
        auto["paper_workload"],
        categories=order,
        ordered=True,
    )
    auto = auto.sort_values("paper_workload")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(auto["paper_workload"], auto["speedup"])
    ax.set_title(f"K-Means Auto OpenMP Speedup on Gaussian Workloads ({target_threads} Threads)")
    ax.set_xlabel("Workload")
    ax.set_ylabel("Speedup")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "fig_kmeans_gaussian_auto_speedup.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    """
    Main plots recommended for the IEEE paper.
    """

    # Main K-Means plot: real-world Covertype dataset.
    plot_speedup_manual_vs_auto(
        csv_file="kmeans_covtype_results.csv",
        output_name="fig_kmeans_covtype_speedup.png",
        title="K-Means Speedup on Covertype Dataset",
    )

    # Optional K-Means runtime plot.
    plot_runtime_manual_vs_auto(
        csv_file="kmeans_covtype_results.csv",
        output_name="fig_kmeans_covtype_runtime.png",
        title="K-Means Runtime on Covertype Dataset",
    )

    # Main FCM plot: selected workload, readable manual vs auto comparison.
    plot_speedup_manual_vs_auto(
        csv_file="fcm_10k_15c_20f_1_results.csv",
        output_name="fig_fcm_10k_speedup.png",
        title="Fuzzy C-Means Speedup on 10K-15C-20F Workload",
    )

    # Optional FCM runtime plot.
    plot_runtime_manual_vs_auto(
        csv_file="fcm_10k_15c_20f_1_results.csv",
        output_name="fig_fcm_10k_runtime.png",
        title="Fuzzy C-Means Runtime on 10K-15C-20F Workload",
    )

    # Compact schedule comparison, replacing the crowded sweep plots.
    plot_fcm_schedule_summary(
        csv_file="fcm_schedule_hyperparam_results.csv",
    )

    # Optional compact K-Means Gaussian summary.
    plot_kmeans_gaussian_summary(
        csv_file="kmeans_paper_gaussian_results.csv",
    )

    print("\nDone. Upload the generated PNG files from:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()