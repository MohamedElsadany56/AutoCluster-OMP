from autocluster_omp.analyzer.dependency_checker import check_loop_safety
from autocluster_omp.analyzer.loop_detector import detect_for_loops
from autocluster_omp.analyzer.pattern_classifier import classify_loop
from autocluster_omp.analyzer.safety_rules import explain_classification
from autocluster_omp.generator.pragma_builder import build_parallel_for_pragma
from autocluster_omp.models import LoopAnalysis


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
