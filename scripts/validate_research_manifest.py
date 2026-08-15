"""Validate the public research manifest template without loading an artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from longieye.research import (  # noqa: E402
    ResearchArtifactError,
    ResearchManifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="校验研究模型 manifest；此命令不会读取或加载 checkpoint。"
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "configs" / "research_manifest.template.json",
    )
    args = parser.parse_args()
    try:
        manifest = ResearchManifest.from_path(args.manifest)
    except ResearchArtifactError as exc:
        print(
            json.dumps(
                {"valid": False, "error": {"code": exc.code, "message": str(exc)}},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        manifest.approval_request()
        approval_request_ready = True
    except ResearchArtifactError:
        approval_request_ready = False
    print(
        json.dumps(
            {
                "valid": True,
                "schema_version": 1,
                "approval_request_ready": approval_request_ready,
                "external_approval_required": True,
                "checkpoint_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
