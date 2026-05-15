from pathlib import Path
import json


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
