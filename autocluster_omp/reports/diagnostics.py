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
