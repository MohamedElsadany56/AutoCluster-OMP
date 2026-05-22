from pathlib import Path
import csv
import json


FIELDS = [
    "algorithm",
    "version",
    "threads",
    "runtime_seconds",
    "speedup",
    "efficiency",
    "checksum",
    "correctness",
    "schedule",
    "workload_name",
    "dataset",
    "labels",
    "clusters",
    "features",
    "iterations",
    "repeat",
    "fuzziness",
    "objective",
]


def export_csv(results: list, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for result in results:
            row = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            writer.writerow({field: row.get(field) for field in FIELDS})
    return path


def generate_diagnostics(results: list[dict]) -> list[str]:
    notes: list[str] = []
    for row in results:
        if row.get("threads") == 1 and row.get("version") in {"manual", "auto"} and row.get("speedup", 1.0) < 1.0:
            notes.append("One-thread OpenMP is slower due to parallel region overhead.")
        if row.get("threads", 1) >= 4 and row.get("efficiency", 1.0) < 0.6:
            notes.append("Efficiency drops at higher thread counts due to synchronization, memory bandwidth, or sequential sections.")
        if row.get("workload_name") == "small" and row.get("speedup", 0.0) <= 1.1:
            notes.append("Small workloads may not benefit from OpenMP because overhead dominates useful work.")
    return sorted(set(notes))


def export_analysis_json(
    analyses: list,
    output_path: str | Path,
    workload_info: dict | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "detected_loops": len(analyses),
        "parallelized_loops": sum(1 for item in analyses if item.decision == "parallelize"),
        "warning_loops": sum(1 for item in analyses if item.decision == "warn"),
        "workload_info": workload_info or {},
        "loop_decisions": [item.to_dict() if hasattr(item, "to_dict") else item for item in analyses],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def export_markdown_report(results: list[dict], output_path: str | Path, title: str = "AutoCluster-OMP Report") -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "## Results", ""]
    lines.append("| Algorithm | Version | Threads | Runtime (s) | Speedup | Efficiency | Correctness |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for row in results:
        lines.append(
            f"| {row.get('algorithm')} | {row.get('version')} | {row.get('threads')} | "
            f"{float(row.get('runtime_seconds', 0)):.6f} | {float(row.get('speedup', 0)):.3f} | "
            f"{float(row.get('efficiency', 0)):.3f} | {row.get('correctness')} |"
        )
    notes = generate_diagnostics(results)
    if notes:
        lines.extend(["", "## Diagnostics", ""])
        lines.extend(f"- {note}" for note in notes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
