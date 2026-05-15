import re
from autocluster_omp.models import LoopInfo


def brace_balance(text: str) -> int:
    return text.count("{") - text.count("}")


def detect_for_loops(source: str) -> list[LoopInfo]:
    lines = source.splitlines(keepends=True)
    loops: list[LoopInfo] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if re.search(r"\bfor\s*\(", line):
            start = i
            header = line.strip()
            balance = brace_balance(line)
            j = i

            while balance <= 0 and j + 1 < len(lines):
                j += 1
                balance += brace_balance(lines[j])
                if "{" in lines[j]:
                    break

            while balance > 0 and j + 1 < len(lines):
                j += 1
                balance += brace_balance(lines[j])

            loops.append(
                LoopInfo(
                    start_line=start + 1,
                    end_line=j + 1,
                    header=header,
                    body="".join(lines[start:j + 1]),
                )
            )

            i = j + 1
        else:
            i += 1

    return loops