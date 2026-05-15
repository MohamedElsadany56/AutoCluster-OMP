from pathlib import Path
import csv


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
