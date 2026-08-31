"""Run the real OLSM public-cohort internal validation and write aggregates."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import stat
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from longieye.public_validation import (
    PRIMARY_MODEL,
    load_olsm_tsv_bytes,
    run_internal_validation,
    serialize_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "f842768f2f1333d2d9cb468b0263655cea018bc5"
SOURCE_URL = (
    "https://raw.githubusercontent.com/lbraglia/aplore3/"
    f"{SOURCE_COMMIT}/rawdata/MYOPIA/MYOPIA.txt"
)
SOURCE_SHA256 = "4147ac651d8f1a975be7eefb39a49044e9a0e5a8f97ddfb40bc764f70680784d"
SOURCE_MAX_BYTES = 100_000
SOURCE_CACHE = PROJECT_ROOT / "build" / "public_cohort" / "MYOPIA.txt"


def _read_verified_cache(path: Path) -> bytes | None:
    try:
        before = path.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_size > SOURCE_MAX_BYTES
        ):
            return None
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (
                opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino
                or opened.st_size != before.st_size
                or not stat.S_ISREG(opened.st_mode)
            ):
                return None
            contents = stream.read(SOURCE_MAX_BYTES + 1)
    except OSError:
        return None
    if len(contents) != before.st_size:
        return None
    if hashlib.sha256(contents).hexdigest() != SOURCE_SHA256:
        return None
    return contents


def ensure_source(path: Path) -> bytes:
    if path.resolve(strict=False) != SOURCE_CACHE.resolve(strict=False):
        raise RuntimeError("public cohort cache path is fixed under build/public_cohort")
    cached = _read_verified_cache(path)
    if cached is not None:
        return cached
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "LongiEye-Public-Cohort-Evaluation/0.5"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            contents = response.read(SOURCE_MAX_BYTES + 1)
    except OSError:
        raise RuntimeError("public cohort download failed") from None
    if len(contents) > SOURCE_MAX_BYTES:
        raise RuntimeError("public cohort download exceeded size contract")
    digest = hashlib.sha256(contents).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError("public cohort digest mismatch")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".MYOPIA.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return contents


def canonicalize(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite report value")
        return float(format(value, ".10g"))
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    if isinstance(value, dict):
        return {key: canonicalize(item) for key, item in value.items()}
    raise TypeError(f"unsupported report value: {type(value).__name__}")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        canonicalize(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def _svg_line_chart(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[float], list[float], str, str]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    path: Path,
    bands: list[tuple[list[float], list[float], list[float], str]] | None = None,
    error_bars: list[tuple[list[float], list[float], list[float], str]]
    | None = None,
) -> None:
    width, height = 860, 600
    left, right, top, bottom = 92, 28, 74, 78
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_pos(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="36" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#12314a">{html.escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#f8fbfc" stroke="#cbd8df"/>',
    ]
    tick_count = 5
    for index in range(tick_count + 1):
        x_value = x_min + (x_max - x_min) * index / tick_count
        y_value = y_min + (y_max - y_min) * index / tick_count
        x = x_pos(x_value)
        y = y_pos(y_value)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#e4ecef"/>',
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e4ecef"/>',
                f'<text x="{x:.2f}" y="{top + plot_height + 25}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#526675">{x_value:.2f}</text>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#526675">{y_value:.2f}</text>',
            ]
        )
    for x_values, lower_values, upper_values, color in bands or []:
        upper_points = [
            f"{x_pos(x):.2f},{y_pos(y):.2f}"
            for x, y in zip(x_values, upper_values, strict=True)
        ]
        lower_points = [
            f"{x_pos(x):.2f},{y_pos(y):.2f}"
            for x, y in reversed(
                list(zip(x_values, lower_values, strict=True))
            )
        ]
        polygon_points = " ".join(upper_points + lower_points)
        parts.append(
            f'<polygon points="{polygon_points}" fill="{color}" fill-opacity="0.14" stroke="none"/>'
        )
    for x_values, lower_values, upper_values, color in error_bars or []:
        for x_value, lower_value, upper_value in zip(
            x_values, lower_values, upper_values, strict=True
        ):
            x = x_pos(x_value)
            lower = y_pos(lower_value)
            upper = y_pos(upper_value)
            parts.extend(
                [
                    f'<line x1="{x:.2f}" y1="{lower:.2f}" x2="{x:.2f}" y2="{upper:.2f}" stroke="{color}" stroke-width="1.5"/>',
                    f'<line x1="{x - 4:.2f}" y1="{lower:.2f}" x2="{x + 4:.2f}" y2="{lower:.2f}" stroke="{color}" stroke-width="1.5"/>',
                    f'<line x1="{x - 4:.2f}" y1="{upper:.2f}" x2="{x + 4:.2f}" y2="{upper:.2f}" stroke="{color}" stroke-width="1.5"/>',
                ]
            )
    for name, x_values, y_values, color, dash in series:
        points = " ".join(
            f"{x_pos(x):.2f},{y_pos(y):.2f}" for x, y in zip(x_values, y_values, strict=True)
        )
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"{dash_attr}/>'
        )
        if not dash:
            for x_value, y_value in zip(x_values, y_values, strict=True):
                parts.append(
                    f'<circle cx="{x_pos(x_value):.2f}" cy="{y_pos(y_value):.2f}" r="4" fill="{color}"/>'
                )
    parts.extend(
        [
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#273743">{html.escape(x_label)}</text>',
            f'<text x="24" y="{top + plot_height / 2:.2f}" text-anchor="middle" transform="rotate(-90 24 {top + plot_height / 2:.2f})" font-family="Arial, sans-serif" font-size="15" fill="#273743">{html.escape(y_label)}</text>',
        ]
    )
    legend_x = left + 14
    legend_y = top + 22
    parts.append(
        f'<rect x="{legend_x - 8}" y="{legend_y - 16}" width="190" height="{len(series) * 23 + 10}" rx="4" fill="#ffffff" fill-opacity="0.88" stroke="#d6e1e6"/>'
    )
    for index, (name, _, _, color, dash) in enumerate(series):
        y = legend_y + index * 23
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 30}" y2="{y}" stroke="{color}" stroke-width="3"{dash_attr}/>',
                f'<text x="{legend_x + 38}" y="{y + 4}" font-family="Arial, sans-serif" font-size="13" fill="#273743">{html.escape(name)}</text>',
            ]
        )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")


def write_calibration_svg(report: dict[str, object], path: Path) -> None:
    bins = report["calibration_curve"]
    assert isinstance(bins, list)
    predicted = [float(item["predicted_mean"]) for item in bins]
    observed = [float(item["observed_rate"]) for item in bins]
    observed_lower = [float(item["observed_rate_ci_95"][0]) for item in bins]
    observed_upper = [float(item["observed_rate_ci_95"][1]) for item in bins]
    upper = max(0.4, max(predicted + observed) * 1.08)
    upper = min(1.0, math.ceil(upper * 10) / 10)
    _svg_line_chart(
        title="OLSM five-year incident myopia: calibration",
        x_label="OOF predicted probability",
        y_label="Observed event rate",
        series=[
            ("Perfect calibration", [0.0, upper], [0.0, upper], "#8a99a5", "7 6"),
            ("Full ridge model", predicted, observed, "#0b8588", ""),
        ],
        x_min=0.0,
        x_max=upper,
        y_min=0.0,
        y_max=upper,
        path=path,
        error_bars=[(predicted, observed_lower, observed_upper, "#0b8588")],
    )


def write_dca_svg(report: dict[str, object], path: Path) -> None:
    dca = report["dca"]
    assert isinstance(dca, dict)
    thresholds = [float(value) for value in dca["thresholds"]]
    model_values = dca["model_net_benefit"]
    assert isinstance(model_values, dict)
    treat_all = [float(value) for value in dca["treat_all"]]
    primary_ci = dca["primary_ci_95"]
    assert isinstance(primary_ci, dict)
    all_values = [float(value) for values in model_values.values() for value in values]
    all_values.extend(treat_all)
    y_min = min(-0.02, math.floor(min(all_values) * 100) / 100)
    y_max = max(0.12, math.ceil(max(all_values) * 100) / 100)
    _svg_line_chart(
        title="Exploratory decision curve analysis",
        x_label="Risk threshold for hypothetical enhanced monitoring",
        y_label="Net benefit",
        series=[
            ("Full", thresholds, list(model_values["full"]), "#0b8588", ""),
            ("Ocular", thresholds, list(model_values["ocular"]), "#315f8c", ""),
            (
                "Refraction only",
                thresholds,
                list(model_values["refraction_only"]),
                "#d18b22",
                "",
            ),
            ("Treat all", thresholds, treat_all, "#9a5661", "6 5"),
            ("Treat none", thresholds, [0.0] * len(thresholds), "#657783", "3 5"),
        ],
        x_min=thresholds[0],
        x_max=thresholds[-1],
        y_min=y_min,
        y_max=y_max,
        path=path,
        bands=[
            (
                thresholds,
                [float(value) for value in primary_ci["lower"]],
                [float(value) for value in primary_ci["upper"]],
                "#0b8588",
            )
        ],
    )


def format_ci(value: float, interval: list[float]) -> str:
    return f"{value:.3f} ({interval[0]:.3f}-{interval[1]:.3f})"


def write_markdown(report: dict[str, object], path: Path) -> None:
    dataset = report["dataset"]
    validation = report["validation"]
    models = report["models"]
    comparisons = report["model_comparisons"]
    temporal = report["entry_cohort_chronological_split_sensitivity"]
    sensitivity = report["split_sensitivity"]
    primary_ci = report["primary_calibration_ci_95"]
    assert isinstance(dataset, dict)
    assert isinstance(validation, dict)
    assert isinstance(models, dict)
    assert isinstance(comparisons, list)
    assert isinstance(temporal, dict)
    assert isinstance(sensitivity, dict)
    assert isinstance(primary_ci, dict)

    lines = [
        "# Public longitudinal cohort validation",
        "",
        "> Status: real public measurements and a five-year longitudinal outcome; pilot internal validation only.",
        "",
        "## Cohort and provenance",
        "",
        f"- Dataset: {dataset['name']}.",
        f"- Source: `aplore3 0.9`, pinned commit `{dataset['source_commit']}`.",
        f"- Raw source SHA-256: `{dataset['source_sha256']}`.",
        "- Rights status: the `aplore3` package is GPL-3; underlying cohort-data reuse rights were not independently re-licensed by this project, so only aggregate results are prepared for review.",
        f"- Participants: {dataset['participants']}; incident-myopia events: {dataset['events']}; non-events: {dataset['non_events']}.",
        f"- Missing cells in the distributed subset: {dataset['missing_cells']}.",
        "- Public package documentation: https://search.r-project.org/CRAN/refmans/aplore3/html/myopia.html",
        "- Primary study context: https://pmc.ncbi.nlm.nih.gov/articles/PMC2871403/",
        "- Raw participant rows and out-of-fold predictions are not committed.",
        "",
        "The OLSM subset contains baseline right-eye measurements and whether myopia developed during the first five years. It does not contain repeated Y1/Y2 predictors, so this analysis does not reproduce the original nine-feature LongiEye contract.",
        "",
        "## Versioned validation design",
        "",
        f"- Repeated nested stratified cross-fitting: {validation['crossfit_repetitions']} repeats, each with {validation['outer_folds']} outer folds and {validation['inner_folds']} inner folds.",
        f"- Primary estimand: {validation['primary_estimand']}.",
        f"- Hyperparameter target: {validation['tuning_metric']}; ridge grid: `{validation['ridge_lambda_grid']}`.",
        f"- Uncertainty: {validation['bootstrap_repetitions']} participant bootstrap repetitions conditional on the repeated OOF predictions; models are not refit inside each bootstrap sample.",
        "- Primary model: full ridge logistic regression.",
        f"- Comparisons: three contrasts defined in this analysis version, {validation['permutation_repetitions']} participant-level randomization permutations, then Holm family-wise error control at 0.05.",
        "- DCA action: hypothetical enhanced monitoring only; thresholds 0.03-0.30 are fixed in this analysis version, not independently preregistered.",
        "",
        "## Internal-validation results",
        "",
        "| Model | Features | AUC (95% CI) | Brier (95% CI) | Calibration intercept | Calibration slope |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("refraction_only", "ocular", "full"):
        row = models[name]
        assert isinstance(row, dict)
        lines.append(
            f"| `{name}` | {len(row['features'])} | "
            f"{format_ci(float(row['auc']), list(row['auc_ci_95']))} | "
            f"{format_ci(float(row['brier']), list(row['brier_ci_95']))} | "
            f"{float(row['calibration_intercept']):.3f} | {float(row['calibration_slope']):.3f} |"
        )
    full = models[PRIMARY_MODEL]
    assert isinstance(full, dict)
    lines.extend(
        [
            "",
            f"Primary calibration-intercept bootstrap CI: {primary_ci['intercept'][0]:.3f} to {primary_ci['intercept'][1]:.3f}. Primary calibration-slope bootstrap CI: {primary_ci['slope'][0]:.3f} to {primary_ci['slope'][1]:.3f}.",
            "",
            "![Calibration curve](assets/public_cohort_calibration.svg)",
            "",
            "## Decision curve analysis",
            "",
            "The curve is exploratory because no clinician-derived utility study defines the relative cost of extra monitoring versus a missed future myopia case.",
            "",
            "![Exploratory DCA](assets/public_cohort_dca.svg)",
            "",
            "## Multiple-comparison results",
            "",
            "| Comparison | Delta AUC (95% CI) | Raw p | Holm-adjusted p | Reject at FWER 0.05 |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| `{row['comparison']}` | {float(row['delta_auc']):.3f} "
            f"({row['delta_auc_ci_95'][0]:.3f}-{row['delta_auc_ci_95'][1]:.3f}) | "
            f"{float(row['p_value_raw']):.4f} | {float(row['p_value_holm']):.4f} | "
            f"{'Yes' if row['reject_fwer_0_05'] else 'No'} |"
        )
    lines.extend(
        [
            "",
            "## Sensitivity checks",
            "",
            f"- Entry-cohort chronological split sensitivity: train entries from 1990-1992 ({temporal['train_participants']} participants, {temporal['train_events']} events); test entries from 1993-1995 ({temporal['test_participants']} participants, {temporal['test_events']} events). Follow-up windows overlap in calendar time, so this is not prospective temporal validation.",
            f"- Full-model temporal AUC: {temporal['models']['full']['auc']:.3f}; Brier: {temporal['models']['full']['brier']:.3f}.",
            f"- The {validation['crossfit_repetitions']} nested-CV repeats gave full-model AUC range {sensitivity['auc_min']:.3f}-{sensitivity['auc_max']:.3f} (median {sensitivity['auc_median']:.3f}); their participant-level OOF probabilities were averaged before primary evaluation.",
            "",
            "## Interpretation boundary",
            "",
            "This is a real public longitudinal-outcome experiment, but it remains pilot internal validation. There are only 81 events, no external cohort, and no repeated Y1/Y2 predictor measurements in the public subset. Bootstrap intervals are conditional on the repeated cross-fitted predictions and do not include full pipeline-refit uncertainty. The results do not validate the synthetic API, do not reproduce the private thesis model, and do not establish clinical utility or deployment readiness.",
            "",
            "The safe portfolio claim is:",
            "",
            "> Built a reproducible public-cohort validation pipeline with repeated nested participant-level cross-fitting, conditional bootstrap uncertainty, calibration, exploratory DCA, and Holm-adjusted paired model comparisons for five-year incident myopia.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    source_bytes = ensure_source(SOURCE_CACHE)
    data = load_olsm_tsv_bytes(source_bytes)
    report, final_models = run_internal_validation(
        data, bootstrap_repetitions=args.bootstrap
    )
    report["dataset"].update(
        {
            "name": "Orinda Longitudinal Study of Myopia public subset",
            "source_package": "aplore3 0.9",
            "source_commit": SOURCE_COMMIT,
            "source_file": "rawdata/MYOPIA/MYOPIA.txt",
            "source_url": SOURCE_URL,
            "source_sha256": SOURCE_SHA256,
            "package_license": "GPL-3",
            "package_documentation": "https://search.r-project.org/CRAN/refmans/aplore3/html/myopia.html",
            "rights_review": "Package-level GPL-3 confirmed; underlying cohort-data reuse rights not independently verified.",
        }
    )
    report = canonicalize(report)

    report_json = PROJECT_ROOT / "benchmarks" / "public_cohort_validation.json"
    model_json = PROJECT_ROOT / "build" / "public_cohort" / "public_cohort_model.json"
    report_md = PROJECT_ROOT / "docs" / "PUBLIC_COHORT_VALIDATION.md"
    calibration_svg = PROJECT_ROOT / "docs" / "assets" / "public_cohort_calibration.svg"
    dca_svg = PROJECT_ROOT / "docs" / "assets" / "public_cohort_dca.svg"

    write_json(report_json, report)
    write_json(
        model_json,
        {
            "schema_version": 1,
            "model_id": "olsm-five-year-myopia-full-ridge-v1",
            "model_stage": "public_cohort_internal_validation",
            "clinical_use": False,
            "training_data": "OLSM public subset via aplore3 0.9",
            "source_commit": SOURCE_COMMIT,
            "source_sha256": SOURCE_SHA256,
            "outcome": "incident myopia during first five years",
            "validation_reference": "benchmarks/public_cohort_validation.json",
            "model": serialize_pipeline(final_models[PRIMARY_MODEL]),
        },
    )
    write_calibration_svg(report, calibration_svg)
    write_dca_svg(report, dca_svg)
    write_markdown(report, report_md)

    print(
        json.dumps(
            {
                "status": "completed",
                "participants": report["dataset"]["participants"],
                "events": report["dataset"]["events"],
                "primary_auc": report["models"][PRIMARY_MODEL]["auc"],
                "primary_brier": report["models"][PRIMARY_MODEL]["brier"],
                "report": str(report_md),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
