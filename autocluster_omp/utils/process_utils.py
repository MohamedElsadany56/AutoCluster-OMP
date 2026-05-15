import subprocess


def run_command(command: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, env=env, check=False)
