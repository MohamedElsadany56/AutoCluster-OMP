from pathlib import Path

from autocluster_omp.analyzer import (
    check_loop_safety,
    detect_for_loops,
    classify_loop,
    explain_classification,
)
from autocluster_omp.models import LoopAnalysis


def write_code(source: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def build_parallel_for_pragma(schedule: str, chunk_size: int | None = None) -> str:
    allowed = {"static", "dynamic", "guided", "auto"}

    if schedule not in allowed:
        raise ValueError(f"Unsupported schedule: {schedule}")

    if schedule in {"dynamic", "guided"} and chunk_size is not None:
        return f"#pragma omp parallel for schedule({schedule}, {chunk_size})"

    return f"#pragma omp parallel for schedule({schedule})"


PARALLELIZABLE = {"kmeans_assignment_loop", "fuzzy_membership_update_loop"}


def analyze_source(source: str, algorithm: str, schedule: str = "static") -> list[LoopAnalysis]:
    pragma = build_parallel_for_pragma(schedule)
    analyses: list[LoopAnalysis] = []
    for loop in detect_for_loops(source):
        classification = classify_loop(loop)
        safety, safety_reason = check_loop_safety(loop, classification)
        _, default_reason = explain_classification(classification)
        should_parallelize = classification in PARALLELIZABLE and safety == "safe"
        reason = safety_reason if safety != "unknown" else default_reason
        analyses.append(
            LoopAnalysis(
                loop=loop,
                classification=classification,
                decision="parallelize" if should_parallelize else "warn" if safety == "unsafe" else "skip",
                reason=reason,
                suggested_pragma=pragma if should_parallelize else None,
                safety=safety,
            )
        )
    return analyses


def generate_openmp_source(
    source: str,
    algorithm: str,
    schedule: str = "static",
    chunk_size: int | None = None,
) -> tuple[str, list[LoopAnalysis]]:
    pragma = build_parallel_for_pragma(schedule, chunk_size)
    analyses = analyze_source(source, algorithm, schedule)
    line_to_pragma: dict[int, str] = {}
    for analysis in analyses:
        if analysis.decision == "parallelize":
            analysis.suggested_pragma = pragma
            line_to_pragma[analysis.loop.start_line] = pragma

    generated_lines: list[str] = []
    lines = source.splitlines(keepends=True)
    for line_number, line in enumerate(lines, start=1):
        if line_number in line_to_pragma and "#pragma omp parallel for" not in line:
            indent = line[: len(line) - len(line.lstrip())]
            generated_lines.append(f"{indent}{line_to_pragma[line_number]}\n")
        generated_lines.append(line)

    return "".join(generated_lines), analyses


def generate_from_file(
    input_path: str,
    algorithm: str,
    schedule: str = "static",
    chunk_size: int | None = None,
) -> tuple[str, list[LoopAnalysis]]:
    with open(input_path, "r", encoding="utf-8") as handle:
        source = handle.read()
    return generate_openmp_source(source, algorithm, schedule, chunk_size)


def transform_source(source: str, schedule: str = "static") -> tuple[str, list]:
    return generate_openmp_source(source, "kmeans", schedule)


def transform_file(input_path: str, schedule: str = "static") -> tuple[str, list]:
    return generate_from_file(input_path, "kmeans", schedule)


def transform_source_fcm(source: str, schedule: str = "static") -> tuple[str, list]:
    return generate_openmp_source(source, "fuzzy-cmeans", schedule)


def transform_file_fcm(input_path: str, schedule: str = "static") -> tuple[str, list]:
    return generate_from_file(input_path, "fuzzy-cmeans", schedule)
