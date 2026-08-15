"""Verify that a wheel contains the current source, not stale build output."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from email.parser import BytesParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "longieye"
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".csv",
    ".db",
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
    ".torchscript",
    ".tsv",
    ".xls",
    ".xlsx",
}


class BuiltPackageError(RuntimeError):
    """Raised when a built wheel differs from the reviewed source tree."""


def verify_wheel(wheel_path: Path) -> dict[str, object]:
    if wheel_path.suffix != ".whl" or not wheel_path.is_file():
        raise BuiltPackageError("expected exactly one wheel file")
    source_files = sorted(SOURCE_ROOT.glob("*.py"))
    expected_members = {f"longieye/{path.name}": path for path in source_files}

    try:
        with zipfile.ZipFile(wheel_path) as archive:
            names = set(archive.namelist())
            missing = sorted(set(expected_members) - names)
            if missing:
                raise BuiltPackageError("wheel is missing reviewed source modules")
            package_members = {
                name
                for name in names
                if name.startswith("longieye/") and not name.endswith("/")
            }
            if package_members != set(expected_members):
                raise BuiltPackageError("wheel package inventory differs from source")
            for member, source_path in expected_members.items():
                if archive.read(member) != source_path.read_bytes():
                    raise BuiltPackageError(
                        f"wheel contains stale source for {source_path.name}"
                    )
            if any(
                Path(name).suffix.lower() in FORBIDDEN_SUFFIXES
                or Path(name).name.lower().startswith(".env")
                for name in names
            ):
                raise BuiltPackageError("wheel contains a forbidden data or model file")

            metadata_names = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise BuiltPackageError("wheel metadata inventory is invalid")
            metadata_bytes = archive.read(metadata_names[0])
            metadata = BytesParser().parsebytes(metadata_bytes)
            if metadata.get("Version") != "0.3.0":
                raise BuiltPackageError("wheel version does not match Sprint 2")
            normalized_metadata = metadata_bytes.replace(b"\r\n", b"\n")
            _, separator, description = normalized_metadata.partition(b"\n\n")
            expected_description = (PROJECT_ROOT / "README.md").read_bytes().replace(
                b"\r\n", b"\n"
            )
            if (
                not separator
                or description.rstrip(b"\n")
                != expected_description.rstrip(b"\n")
            ):
                raise BuiltPackageError("wheel contains stale README metadata")
    except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
        raise BuiltPackageError("unable to inspect wheel") from exc

    return {
        "valid": True,
        "wheel": wheel_path.name,
        "sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
        "source_modules_verified": len(expected_members),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="确认 wheel 与当前已审阅源码逐字节一致。"
    )
    parser.add_argument("wheel", nargs="+", type=Path)
    args = parser.parse_args()
    if len(args.wheel) != 1:
        raise SystemExit("Built-package policy requires exactly one wheel.")
    try:
        result = verify_wheel(args.wheel[0])
    except BuiltPackageError as exc:
        raise SystemExit(f"Built-package policy failed: {exc}") from None
    print(
        "Built-package policy passed: "
        f"{result['wheel']} ({result['source_modules_verified']} source modules)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
