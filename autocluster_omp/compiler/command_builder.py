from pathlib import Path


def build_gcc_command(
    source: str | Path,
    output: str | Path,
    openmp: bool = False,
    defines: dict[str, int | float | str] | None = None,
) -> list[str]:
    command = ["gcc", str(source), "-O2"]
    if openmp:
        command.append("-fopenmp")
    if defines:
        for key, value in defines.items():
            command.append(f"-D{key}={value}")
    command.extend(["-lm", "-o", str(output)])
    return command
