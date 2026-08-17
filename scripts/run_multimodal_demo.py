"""Run the offline, fully synthetic multimodal demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from longieye.features import extract_features  # noqa: E402
from longieye.fusion import StructuredAnchoredFusionAdapter  # noqa: E402
from longieye.imaging import ImageArtifactError, RasterImage, decode_synthetic_png  # noqa: E402
from longieye.model import DemoRiskModel  # noqa: E402
from scripts.generate_synthetic_fundus import (  # noqa: E402
    MAX_PNG_BYTES,
    OUTPUT_PATHS,
    generate_eye_rgb,
)
from scripts.run_demo import load_case  # noqa: E402


SCENARIO_IMAGES = {
    "both": ("od", "os"),
    "missing-os": ("od",),
    "missing-both": (),
}


class MultimodalDemoError(ValueError):
    """Stable CLI boundary error that never includes a filesystem path."""


def _read_fixed_fixture(eye: str) -> bytes:
    path = OUTPUT_PATHS[eye]
    try:
        if path.is_symlink():
            raise MultimodalDemoError("合成图像不能是符号链接。")
        file_stat = path.stat()
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > MAX_PNG_BYTES:
            raise MultimodalDemoError("合成图像文件不符合大小或类型限制。")
        with path.open("rb") as handle:
            raw_bytes = handle.read(MAX_PNG_BYTES + 1)
    except FileNotFoundError:
        raise MultimodalDemoError("缺少合成图像；请先运行生成脚本。") from None
    except OSError:
        raise MultimodalDemoError("无法读取固定的合成图像。") from None
    if len(raw_bytes) > MAX_PNG_BYTES:
        raise MultimodalDemoError("合成图像文件超过大小限制。")
    return raw_bytes


def load_fixture_images(scenario: str) -> dict[str, RasterImage]:
    if scenario not in SCENARIO_IMAGES:
        raise MultimodalDemoError("不支持的演示场景。")
    images: dict[str, RasterImage] = {}
    for eye in SCENARIO_IMAGES[scenario]:
        expected_sha256 = hashlib.sha256(generate_eye_rgb(eye)).hexdigest()
        try:
            images[eye] = decode_synthetic_png(
                _read_fixed_fixture(eye),
                expected_eye=eye,
                expected_pixel_sha256=expected_sha256,
            )
        except ImageArtifactError:
            raise MultimodalDemoError("固定合成图像未通过完整性校验。") from None
    return images


def run_scenario(scenario: str) -> dict[str, object]:
    case, _ = load_case(PROJECT_ROOT / "examples" / "request.json")
    model = DemoRiskModel.from_path(PROJECT_ROOT / "configs" / "demo_model.json")
    adapter = StructuredAnchoredFusionAdapter(model)
    result = adapter.predict_with_images(
        extract_features(case),
        load_fixture_images(scenario),
    )
    payload = result.as_dict()
    payload["scenario"] = scenario
    payload["image_source"] = "procedurally_generated_synthetic_only"
    return payload


def chinese_output(payload: dict[str, object]) -> str:
    predictions = payload["predictions"]
    if not isinstance(predictions, dict):
        raise MultimodalDemoError("多模态演示输出合同无效。")
    mode_labels = {
        "multimodal": "双眼合成图像分支均启用",
        "partial_fallback": "单眼图像分支回退",
        "structured_fallback": "双眼均回退到结构化分支",
    }
    eye_labels = {"od": "右眼（OD）", "os": "左眼（OS）"}
    lines = [
        "LongiEye 全合成多模态离线演示",
        "=" * 36,
        f"场景：{payload['scenario']}",
        f"整体模式：{mode_labels[str(payload['mode'])]}",
        "图像来源：代码确定性生成，不含真实眼底图像",
        "",
    ]
    for eye in ("od", "os"):
        result = predictions[eye]
        lines.extend(
            [
                f"{eye_labels[eye]}：",
                f"  合成演示分数：{float(result['demo_score']):.6f}",
                f"  结构化锚点：{float(result['structured_anchor_score']):.6f}",
                f"  分支模式：{result['mode']}",
                f"  回退原因：{result['reason_code'] or '无'}",
            ]
        )
    lines.extend(["", f"安全提示：{payload['disclaimer']}"])
    return "\n".join(lines)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="运行不接入公开 API 的全合成多模态演示。"
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIO_IMAGES),
        default="both",
        help="选择双眼融合、单眼缺失或双眼缺失回退。",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="输出适合录屏讲解的中文结果。",
    )
    args = parser.parse_args()
    try:
        payload = run_scenario(args.scenario)
    except (MultimodalDemoError, ValueError):
        print("多模态演示失败；请检查固定合成工件与本地环境。", file=sys.stderr)
        return 1
    if args.human:
        print(chinese_output(payload))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
