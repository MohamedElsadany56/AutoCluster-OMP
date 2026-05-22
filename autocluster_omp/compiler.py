from pathlib import Path
import shutil
import subprocess


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


class GCCNotFoundError(RuntimeError):
    pass


def compile_c_source(
    source: str | Path,
    output: str | Path,
    openmp: bool = False,
    defines: dict[str, int | float | str] | None = None,
) -> Path:
    if shutil.which("gcc") is None:
        raise GCCNotFoundError("GCC was not found in PATH. Install GCC or run inside WSL/Linux.")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_gcc_command(source, output_path, openmp=openmp, defines=defines)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "GCC compilation failed:\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return output_path
