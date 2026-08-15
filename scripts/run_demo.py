"""Run one local prediction without starting the web server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from longieye.domain import LongitudinalCase, VisitMeasurements  # noqa: E402
from longieye.model import DemoRiskModel  # noqa: E402
from longieye.service import RiskPredictionService  # noqa: E402


FEATURE_LABELS = {
    "sex_y1": "Y1 性别编码",
    "height_delta_cm": "身高变化（cm）",
    "weight_delta_kg": "体重变化（kg）",
    "sbp_delta_mmhg": "收缩压变化（mmHg）",
    "dbp_delta_mmhg": "舒张压变化（mmHg）",
    "waist_delta_cm": "腰围变化（cm）",
    "wears_glasses_delta": "佩戴眼镜状态变化",
    "axial_length_od_delta_mm": "右眼眼轴变化（mm）",
    "axial_length_os_delta_mm": "左眼眼轴变化（mm）",
}


def load_case(path: Path) -> tuple[LongitudinalCase, str | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("演示请求必须是 JSON 对象。")
    y1 = cast(dict[str, Any], payload.get("y1"))
    y2 = cast(dict[str, Any], payload.get("y2"))
    followup_months = payload.get("followup_months", 12)
    case_id = payload.get("case_id")
    if not isinstance(y1, dict) or not isinstance(y2, dict):
        raise ValueError("演示请求必须包含 y1 和 y2 两次随访。")
    if not isinstance(followup_months, int) or isinstance(followup_months, bool):
        raise ValueError("followup_months 必须是整数。")
    if case_id is not None and not isinstance(case_id, str):
        raise ValueError("case_id 必须是文本或 null。")
    return (
        LongitudinalCase(
            y1=VisitMeasurements(**y1),
            y2=VisitMeasurements(**y2),
            followup_months=followup_months,
        ),
        case_id,
    )


def chinese_demo_output(result: dict[str, object]) -> str:
    model = cast(dict[str, object], result["model"])
    predictions = cast(dict[str, dict[str, float]], result["predictions"])
    features = cast(dict[str, float], result["derived_features"])
    lines = [
        "LongiEye 中文命令行演示",
        "=" * 30,
        f"案例代号：{result['case_id'] or '未提供'}",
        f"请求追踪 ID：{result['request_id']}",
        f"模型：{model['model_id']}",
        f"模型阶段：{model['model_stage']}（合成演示）",
        "训练数据：固定随机种子生成的确定性合成数据",
        "",
        "双眼合成演示分数（不是临床风险概率）：",
        f"  右眼（OD）：{predictions['od']['demo_probability']:.6f}",
        f"  左眼（OS）：{predictions['os']['demo_probability']:.6f}",
        "",
        "纵向派生特征（Y2 - Y1）：",
    ]
    for name, label in FEATURE_LABELS.items():
        value = features[name]
        display = f"{value:.0f}" if name == "sex_y1" else f"{value:+.6g}"
        lines.append(f"  {label}：{display}")
    lines.extend(["", f"安全提示：{result['disclaimer']}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行一次 LongiEye 本地合成推理演示。")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "examples" / "request.json",
        help="脱敏 JSON 请求文件。",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="输出适合录屏讲解的中文结果；默认保留机器可读 JSON。",
    )
    args = parser.parse_args()
    if args.human and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    case, case_id = load_case(args.input)
    model = DemoRiskModel.from_path(PROJECT_ROOT / "configs" / "demo_model.json")
    result = RiskPredictionService(model).predict(case, case_id=case_id)
    if args.human:
        print(chinese_demo_output(result))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
