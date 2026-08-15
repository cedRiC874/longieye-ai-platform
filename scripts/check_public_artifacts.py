"""Reject commonly sensitive research files if they become Git-tracked."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DENIED_DIRECTORIES = {
    "artifacts",
    "checkpoints",
    "data",
    "exports",
    "models",
    "private",
    "research_packages",
}
DENIED_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".csv",
    ".db",
    ".feather",
    ".joblib",
    ".key",
    ".npy",
    ".npz",
    ".onnx",
    ".p12",
    ".parquet",
    ".pem",
    ".pfx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pt2",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".torchscript",
    ".tsv",
    ".xls",
    ".xlsx",
}
MAX_TRACKED_FILE_BYTES = 5 * 1024 * 1024
ALLOWED_RESEARCH_TEMPLATES = {"configs/research_manifest.template.json"}
SENSITIVE_DATA_TOKENS = {
    "checkpoint",
    "oof",
    "participant",
    "patient",
    "prediction",
    "preprocessing",
    "split",
}


def policy_violations(paths: Iterable[str]) -> list[str]:
    violations: list[str] = []
    for raw_path in paths:
        normalized = raw_path.replace("\\", "/")
        lowered_path = normalized.lower()
        path = PurePosixPath(normalized)
        lowered_parts = tuple(part.lower() for part in path.parts)
        if any(part.startswith(".env") for part in lowered_parts):
            violations.append(normalized)
            continue
        if any(part in DENIED_DIRECTORIES for part in lowered_parts):
            violations.append(normalized)
            continue
        if (
            lowered_path.startswith("configs/")
            and "research" in lowered_path
            and lowered_path not in ALLOWED_RESEARCH_TEMPLATES
        ):
            violations.append(normalized)
            continue
        if path.suffix.lower() in DENIED_SUFFIXES:
            violations.append(normalized)
            continue
        if (
            path.suffix.lower() in {".json", ".jsonl", ".txt"}
            and any(token in path.stem.lower() for token in SENSITIVE_DATA_TOKENS)
            and path.suffix.lower() != ".md"
        ):
            violations.append(normalized)
    return sorted(set(violations))


def tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    ]


def main() -> None:
    paths = tracked_paths()
    violations = policy_violations(paths)
    for relative_path in paths:
        path = PROJECT_ROOT / relative_path
        try:
            if path.stat().st_size > MAX_TRACKED_FILE_BYTES:
                violations.append(relative_path.replace("\\", "/"))
        except OSError:
            violations.append(relative_path.replace("\\", "/"))
    violations = sorted(set(violations))
    if violations:
        rendered = "\n".join(f"- {path}" for path in violations)
        raise SystemExit(
            "Public-artifact policy rejected tracked files:\n" + rendered
        )
    print(
        f"Public-artifact policy passed for {len(paths)} tracked or candidate files."
    )


if __name__ == "__main__":
    main()
