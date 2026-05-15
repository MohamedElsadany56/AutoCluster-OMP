from pathlib import Path

from autocluster_omp.reports.diagnostics import generate_diagnostics


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
