"""Build a private engineering-only report for an authorized research package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from longieye.comparison import (  # noqa: E402
    AdapterComparisonError,
    build_engineering_comparison,
    comparison_markdown,
)
from longieye.domain import LongitudinalCase, VisitMeasurements  # noqa: E402
from longieye.features import extract_features  # noqa: E402
from longieye.model import DemoRiskModel  # noqa: E402
from longieye.research import (  # noqa: E402
    ExternalJsonApprovalPolicy,
    ResearchArtifactError,
    ResearchModelAdapter,
)


def load_synthetic_features() -> dict[str, float]:
    payload = json.loads(
        (PROJECT_ROOT / "examples" / "request.json").read_text(encoding="utf-8")
    )
    case = LongitudinalCase(
        y1=VisitMeasurements(**payload["y1"]),
        y2=VisitMeasurements(**payload["y2"]),
        followup_months=payload["followup_months"],
    )
    return extract_features(case)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "在隔离目录比较公开合成 adapter 与已授权研究 adapter 的工程开销；"
            "不比较模型效果。"
        )
    )
    parser.add_argument("research_package", type=Path)
    parser.add_argument(
        "--approval-receipt",
        required=True,
        type=Path,
        help="由包外受控位置提供、绑定全部工件哈希的审批回执。",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "private" / "comparison",
        help="默认位于 Git 忽略的私有目录。",
    )
    args = parser.parse_args()

    demo = DemoRiskModel.from_path(PROJECT_ROOT / "configs" / "demo_model.json")
    try:
        research = ResearchModelAdapter.from_package(
            args.research_package,
            approval_policy=ExternalJsonApprovalPolicy(args.approval_receipt),
        )
    except ResearchArtifactError as exc:
        print(
            json.dumps(
                {"error": {"code": exc.code, "message": str(exc)}},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        report = build_engineering_comparison(
            demo,
            research,
            load_synthetic_features(),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "adapter_comparison.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (args.output_dir / "adapter_comparison.md").write_text(
            comparison_markdown(report), encoding="utf-8", newline="\n"
        )
    except (AdapterComparisonError, ResearchArtifactError, OSError):
        print(
            json.dumps(
                {
                    "error": {
                        "code": "comparison_failed",
                        "message": "工程比较失败；未生成报告。",
                    }
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    print(comparison_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
