from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.generate_synthetic_fundus as fixture_generator
from scripts.benchmark_multimodal import build_report, markdown_report
from scripts.generate_synthetic_fundus import (
    FixtureError,
    OUTPUT_PATHS,
    check_fixtures,
    generate_eye_rgb,
)
from scripts.run_multimodal_demo import chinese_output, run_scenario


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKED_REPORT_JSON = PROJECT_ROOT / "benchmarks" / "multimodal_latest.json"
TRACKED_REPORT_MARKDOWN = PROJECT_ROOT / "benchmarks" / "multimodal_latest.md"


def nested_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from nested_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from nested_keys(nested)


def test_tracked_synthetic_fundus_fixtures_match_integer_generator():
    check_fixtures()

    assert len(generate_eye_rgb("od")) == 128 * 128 * 3
    assert len(generate_eye_rgb("os")) == 128 * 128 * 3
    assert generate_eye_rgb("od") != generate_eye_rgb("os")
    assert all(path.is_file() for path in OUTPUT_PATHS.values())


def configure_temporary_fixture_output(monkeypatch, tmp_path):
    output_directory = tmp_path / "examples" / "synthetic_fundus"
    output_paths = {
        "od": output_directory / "od.png",
        "os": output_directory / "os.png",
    }
    monkeypatch.setattr(fixture_generator, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(fixture_generator, "OUTPUT_DIRECTORY", output_directory)
    monkeypatch.setattr(fixture_generator, "OUTPUT_PATHS", output_paths)
    return output_directory, output_paths


def test_fixture_writer_refuses_a_symlinked_output_directory(monkeypatch, tmp_path):
    output_directory, output_paths = configure_temporary_fixture_output(
        monkeypatch, tmp_path
    )
    output_directory.mkdir(parents=True)
    original_is_symlink = Path.is_symlink

    def fake_is_symlink(path):
        return path == output_directory or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    with pytest.raises(FixtureError, match="输出目录不能包含符号链接"):
        fixture_generator.write_fixtures()
    assert not any(path.exists() for path in output_paths.values())


def test_fixture_writer_refuses_a_symlinked_target_without_touching_it(
    monkeypatch, tmp_path
):
    output_directory, output_paths = configure_temporary_fixture_output(
        monkeypatch, tmp_path
    )
    output_directory.mkdir(parents=True)
    output_paths["od"].write_bytes(b"private sentinel")
    original_is_symlink = Path.is_symlink

    def fake_is_symlink(path):
        return path == output_paths["od"] or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    with pytest.raises(FixtureError, match="输出不能是符号链接"):
        fixture_generator.write_fixtures()
    assert output_paths["od"].read_bytes() == b"private sentinel"
    assert not output_paths["os"].exists()


def test_fixture_writer_uses_atomic_replace_and_removes_temporary_files(
    monkeypatch, tmp_path
):
    output_directory, output_paths = configure_temporary_fixture_output(
        monkeypatch, tmp_path
    )
    real_replace = fixture_generator.os.replace
    replacements = []

    def recording_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(fixture_generator.os, "replace", recording_replace)
    fixture_generator.write_fixtures()

    assert [destination for _, destination in replacements] == list(
        output_paths.values()
    )
    assert all(source.parent == output_directory for source, _ in replacements)
    assert not list(output_directory.glob("*.tmp"))
    assert all(path.is_file() for path in output_paths.values())


def test_multimodal_chinese_demo_covers_fusion_and_exact_fallback():
    fused = run_scenario("both")
    partial = run_scenario("missing-os")
    fallback = run_scenario("missing-both")

    assert fused["mode"] == "multimodal"
    assert fused["predictions"]["od"]["demo_score"] == 0.17631
    assert fused["predictions"]["os"]["demo_score"] == 0.128808
    assert partial["mode"] == "partial_fallback"
    assert partial["predictions"]["os"] == {
        "demo_score": 0.12086,
        "structured_anchor_score": 0.12086,
        "mode": "tabular_only",
        "reason_code": "image_missing",
        "logit_adjustment": 0.0,
    }
    assert fallback["mode"] == "structured_fallback"
    output = chinese_output(partial)
    assert output.startswith("LongiEye 全合成多模态离线演示")
    assert "代码确定性生成，不含真实眼底图像" in output
    assert "不可用于诊断、筛查或治疗决策" in output


def test_multimodal_benchmark_records_only_aggregate_engineering_evidence():
    report = build_report(warmup=1, iterations=2)
    markdown = markdown_report(report)
    report_keys = set(nested_keys(report))

    assert [result["scenario"] for result in report["results"]] == [
        "both_images",
        "missing_os",
        "missing_both",
    ]
    assert [result["synthetic_image_branches"] for result in report["results"]] == [
        2,
        1,
        0,
    ]
    assert report["clinical_use"] is False
    for forbidden in (
        "demo_score",
        "structured_anchor_score",
        "pixel_sha256",
        "embedding",
        "case_id",
        "image_path",
        "auc",
    ):
        assert forbidden not in report_keys
    assert "不保存图像、路径、像素" in markdown
    assert "不是临床性能指标" in markdown


def test_tracked_multimodal_benchmark_is_schema_bound_and_safely_rendered():
    report = json.loads(TRACKED_REPORT_JSON.read_text(encoding="utf-8"))
    report_keys = set(nested_keys(report))

    assert set(report) == {
        "schema_version",
        "package_version",
        "fusion_contract_version",
        "image_embedding_contract_version",
        "generated_at_utc",
        "model_stage",
        "image_source",
        "clinical_use",
        "environment",
        "measurement",
        "results",
    }
    assert report["schema_version"] == 1
    assert report["package_version"] == "0.4.0"
    assert report["fusion_contract_version"] == "structured-anchor-logit-residual-v1"
    assert report["image_embedding_contract_version"] == "synthetic-fundus-statistics-v1"
    assert report["clinical_use"] is False
    assert set(report["environment"]) == {"python", "implementation", "platform"}
    assert set(report["measurement"]) == {
        "sequential",
        "includes_file_io",
        "warmup_iterations",
        "iterations_per_scenario",
        "persists_case_outputs",
    }
    expected_result_keys = {
        "iterations",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "mean_ms",
        "throughput_requests_per_second",
        "scenario",
        "expected_result_mode",
        "synthetic_image_branches",
        "structured_fallback_branches",
        "warmup_iterations",
        "errors",
    }
    assert all(set(result) == expected_result_keys for result in report["results"])
    for forbidden in (
        "demo_score",
        "structured_anchor_score",
        "pixel_sha256",
        "embedding",
        "case_id",
        "image_path",
        "auc",
    ):
        assert forbidden not in report_keys
    assert TRACKED_REPORT_MARKDOWN.read_text(encoding="utf-8") == markdown_report(
        report
    )
