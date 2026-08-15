from pathlib import Path

from longieye.model import DemoRiskModel
from longieye.service import RiskPredictionService
from scripts.run_demo import chinese_demo_output, load_case


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_human_demo_output_is_chinese_and_preserves_safety_boundary():
    case, case_id = load_case(PROJECT_ROOT / "examples" / "request.json")
    model = DemoRiskModel.from_path(PROJECT_ROOT / "configs" / "demo_model.json")
    result = RiskPredictionService(model).predict(case, case_id=case_id)

    output = chinese_demo_output(result)

    assert output.startswith("LongiEye 中文命令行演示")
    assert "右眼（OD）：0.165514" in output
    assert "左眼（OS）：0.120860" in output
    assert "纵向派生特征（Y2 - Y1）" in output
    assert "不可用于诊断、筛查或治疗决策" in output
