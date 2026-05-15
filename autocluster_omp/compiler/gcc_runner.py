from pathlib import Path
import shutil
import subprocess

from autocluster_omp.compiler.command_builder import build_gcc_command


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
