from pathlib import Path
import subprocess


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> Path:
    resolved = ensure_parent(path)
    resolved.write_text(content, encoding="utf-8")
    return resolved


def run_command(command: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, env=env, check=False)


def leading_whitespace(text: str) -> str:
    return text[: len(text) - len(text.lstrip())]
