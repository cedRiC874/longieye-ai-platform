"""Reject commonly sensitive research files if they become Git-tracked."""

from __future__ import annotations

import hashlib
import os
import stat
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
    ".doc",
    ".docx",
    ".7z",
    ".feather",
    ".joblib",
    ".gz",
    ".key",
    ".npy",
    ".npz",
    ".onnx",
    ".p12",
    ".parquet",
    ".pem",
    ".pfx",
    ".pdf",
    ".ppt",
    ".pptx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pt2",
    ".rar",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tar.gz",
    ".torchscript",
    ".tsv",
    ".whl",
    ".xls",
    ".xlsx",
    ".zip",
}
MAX_TRACKED_FILE_BYTES = 5 * 1024 * 1024
ALLOWED_RESEARCH_TEMPLATES = {"configs/research_manifest.template.json"}
ALLOWED_SYNTHETIC_IMAGES = {
    "examples/synthetic_fundus/od.png": "f232b6fc7a44b1c96d259cfced275c8bbfe84d3914234bae50515e7e1cd3e2dc",
    "examples/synthetic_fundus/os.png": "8b07a05783bd18a35f0581b325628268930b093267aba3b955e9abf64b64a16e",
}
ALLOWED_PUBLIC_VECTOR_IMAGES = {
    "docs/assets/architecture.svg": "80577690c4896f6ecf7dcbb36478bd004a1707f1554a7c67b744f7188adec59c"
}
DENIED_IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".cr2",
    ".dat",
    ".dcm",
    ".dicom",
    ".dng",
    ".gif",
    ".heic",
    ".heif",
    ".j2k",
    ".jp2",
    ".jpx",
    ".jpeg",
    ".jpg",
    ".nef",
    ".nii",
    ".pgm",
    ".png",
    ".ppm",
    ".raw",
    ".svg",
    ".tif",
    ".tiff",
    ".webp",
}
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
            path.suffix.lower() in DENIED_IMAGE_SUFFIXES
            or lowered_path.endswith(".nii.gz")
        ):
            if (
                normalized not in ALLOWED_SYNTHETIC_IMAGES
                and normalized not in ALLOWED_PUBLIC_VECTOR_IMAGES
            ):
                violations.append(normalized)
            continue
        if (
            path.suffix.lower() in {".json", ".jsonl", ".txt"}
            and any(token in path.stem.lower() for token in SENSITIVE_DATA_TOKENS)
            and path.suffix.lower() != ".md"
        ):
            violations.append(normalized)
    return sorted(set(violations))


def content_looks_like_denied_image(prefix: bytes) -> bool:
    """Best-effort magic-byte guard for images/containers with another suffix."""

    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if prefix.startswith(b"\xff\xd8\xff"):
        return True
    if prefix.startswith((b"GIF87a", b"GIF89a", b"BM")):
        return True
    if prefix.startswith(
        (
            b"%PDF-",
            b"PK\x03\x04",
            b"PK\x05\x06",
            b"7z\xbc\xaf'\x1c",
            b"Rar!",
            b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
        )
    ):
        return True
    if prefix.startswith(b"\x1f\x8b"):
        return True
    if prefix.startswith((b"II*\x00", b"MM\x00*")):
        return True
    if prefix.startswith((b"P5\n", b"P5\r", b"P5 ", b"P6\n", b"P6\r", b"P6 ")):
        return True
    if prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP":
        return True
    if len(prefix) >= 132 and prefix[128:132] == b"DICM":
        return True
    if len(prefix) >= 263 and prefix[257:263] in {b"ustar\x00", b"ustar "}:
        return True
    if prefix.startswith(b"\x00\x00\x00\x0cjP  \r\n\x87\n") or prefix.startswith(
        b"\xffO\xffQ"
    ):
        return True
    if len(prefix) >= 12 and prefix[4:8] == b"ftyp" and prefix[8:12] in {
        b"avif",
        b"avis",
        b"heic",
        b"heif",
        b"heix",
        b"hevc",
        b"hevx",
        b"mif1",
        b"msf1",
    }:
        return True
    if len(prefix) >= 348:
        nifti_header = int.from_bytes(prefix[:4], "little") == 348 or int.from_bytes(
            prefix[:4], "big"
        ) == 348
        if nifti_header and prefix[344:348] in {b"n+1\x00", b"ni1\x00"}:
            return True
    lowered_text = prefix[:4096].lstrip(b"\xef\xbb\xbf\x00\t\r\n ").lower()
    if lowered_text.startswith(b"<svg") or (
        lowered_text.startswith(b"<?xml") and b"<svg" in lowered_text
    ):
        return True
    return False


class ArtifactReadError(ValueError):
    """Raised when a candidate cannot be inspected as a bounded regular file."""


def _read_bounded_regular_file(
    path: Path, *, max_total_bytes: int, prefix_bytes: int | None = None
) -> bytes:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ArtifactReadError
        if before.st_size > max_total_bytes:
            raise ArtifactReadError
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (
                opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino
                or opened.st_size != before.st_size
                or not stat.S_ISREG(opened.st_mode)
            ):
                raise ArtifactReadError
            read_limit = (
                max_total_bytes + 1 if prefix_bytes is None else prefix_bytes
            )
            contents = stream.read(read_limit)
    except ArtifactReadError:
        raise
    except OSError:
        raise ArtifactReadError from None
    if prefix_bytes is None and len(contents) != before.st_size:
        raise ArtifactReadError
    if (
        prefix_bytes is not None
        and before.st_size <= prefix_bytes
        and len(contents) != before.st_size
    ):
        raise ArtifactReadError
    return contents


def content_violations(paths: Iterable[str]) -> list[str]:
    allowed = set(ALLOWED_SYNTHETIC_IMAGES) | set(ALLOWED_PUBLIC_VECTOR_IMAGES)
    violations: list[str] = []
    for raw_path in paths:
        normalized = raw_path.replace("\\", "/")
        if normalized in allowed:
            continue
        path = PROJECT_ROOT / normalized
        try:
            prefix = _read_bounded_regular_file(
                path,
                max_total_bytes=MAX_TRACKED_FILE_BYTES,
                prefix_bytes=4096,
            )
        except ArtifactReadError:
            violations.append(normalized)
            continue
        if content_looks_like_denied_image(prefix):
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
    violations.extend(content_violations(paths))
    normalized_paths = {path.replace("\\", "/") for path in paths}
    allowed_hashed_assets = ALLOWED_SYNTHETIC_IMAGES | ALLOWED_PUBLIC_VECTOR_IMAGES
    for relative_path, expected_sha256 in allowed_hashed_assets.items():
        path = PROJECT_ROOT / relative_path
        if relative_path not in normalized_paths:
            violations.append(relative_path)
            continue
        try:
            contents = _read_bounded_regular_file(
                path, max_total_bytes=64 * 1024
            )
            actual_sha256 = hashlib.sha256(contents).hexdigest()
        except ArtifactReadError:
            violations.append(relative_path)
            continue
        if actual_sha256 != expected_sha256:
            violations.append(relative_path)
    for relative_path in paths:
        path = PROJECT_ROOT / relative_path
        try:
            file_stat = path.lstat()
            if (
                stat.S_ISLNK(file_stat.st_mode)
                or not stat.S_ISREG(file_stat.st_mode)
                or file_stat.st_size > MAX_TRACKED_FILE_BYTES
            ):
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
