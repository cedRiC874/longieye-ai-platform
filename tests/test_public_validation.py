from __future__ import annotations

import csv
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")

from longieye.public_validation import (  # noqa: E402
    EXPECTED_HEADERS,
    PublicValidationError,
    auc_score,
    calibration_statistics,
    decision_curve_net_benefit,
    holm_adjust,
    load_olsm_tsv,
    nested_oof_predictions,
    paired_permutation_auc_p_value,
    stratified_folds,
)


def _write_valid_fixture(path: Path, *, duplicate_last_id: bool = False) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXPECTED_HEADERS, delimiter="\t")
        writer.writeheader()
        for index in range(618):
            event = 1 if index < 81 else 0
            subject_id = 617 if duplicate_last_id and index == 617 else index + 1
            writer.writerow(
                {
                    "ID": subject_id,
                    "STUDYYEAR": 1990 + index % 6,
                    "MYOPIC": event,
                    "AGE": 5 + index % 5,
                    "GENDER": index % 2,
                    "SPHEQ": 0.1 + index / 1000,
                    "AL": 21.0 + index / 1000,
                    "ACD": 3.2,
                    "LT": 3.5,
                    "VCD": 14.5,
                    "SPORTHR": 5 + index % 10,
                    "READHR": index % 8,
                    "COMPHR": index % 4,
                    "STUDYHR": index % 6,
                    "TVHR": index % 12,
                    "DIOPTERHR": 10 + index % 20,
                    "MOMMY": index % 2,
                    "DADMY": (index // 2) % 2,
                }
            )


def test_public_cohort_loader_accepts_pinned_contract(tmp_path: Path) -> None:
    source = tmp_path / "myopia.tsv"
    _write_valid_fixture(source)

    data = load_olsm_tsv(source)

    assert data.size == 618
    assert data.events == 81
    assert np.unique(data.ids).size == 618


def test_public_cohort_loader_rejects_duplicate_participant_id(
    tmp_path: Path,
) -> None:
    source = tmp_path / "myopia.tsv"
    _write_valid_fixture(source, duplicate_last_id=True)

    with pytest.raises(PublicValidationError) as exc_info:
        load_olsm_tsv(source)

    assert exc_info.value.code == "cohort_id_invalid"


def test_public_cohort_loader_rejects_short_and_extra_rows(tmp_path: Path) -> None:
    valid = tmp_path / "valid.tsv"
    _write_valid_fixture(valid)
    lines = valid.read_text(encoding="utf-8").splitlines()

    for name, malformed_row in (
        ("short", "\t".join(lines[1].split("\t")[:-1])),
        ("extra", lines[1] + "\textra"),
    ):
        candidate = tmp_path / f"{name}.tsv"
        candidate.write_text(
            "\n".join([lines[0], malformed_row, *lines[2:]]) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(PublicValidationError) as exc_info:
            load_olsm_tsv(candidate)
        assert exc_info.value.code == "cohort_schema_invalid"


def test_auc_handles_ties() -> None:
    outcomes = np.asarray([0, 0, 1, 1], dtype=float)
    predictions = np.asarray([0.1, 0.4, 0.4, 0.8], dtype=float)

    assert auc_score(outcomes, predictions) == pytest.approx(0.875)

    strong = np.asarray([0.05, 0.1, 0.8, 0.9], dtype=float)
    weak = np.asarray([0.4, 0.6, 0.4, 0.6], dtype=float)
    p_value = paired_permutation_auc_p_value(
        outcomes, strong, weak, repetitions=1000, seed=9
    )
    assert 0.0 < p_value <= 1.0
    with pytest.raises(PublicValidationError, match="1000 permutations"):
        paired_permutation_auc_p_value(
            outcomes, strong, weak, repetitions=999, seed=9
        )


def test_holm_adjustment_is_monotone_in_rank_order() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_decision_curve_uses_standard_net_benefit_formula() -> None:
    outcomes = np.asarray([1, 1, 0, 0], dtype=float)
    probabilities = np.asarray([0.9, 0.6, 0.7, 0.1], dtype=float)

    result = decision_curve_net_benefit(
        outcomes, probabilities, np.asarray([0.5], dtype=float)
    )

    assert result[0] == pytest.approx(0.25)


def test_stratified_folds_are_deterministic_and_non_overlapping() -> None:
    outcomes = np.asarray([0] * 50 + [1] * 20, dtype=float)

    first = stratified_folds(outcomes, 5, 42)
    second = stratified_folds(outcomes, 5, 42)

    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert np.unique(np.concatenate(first)).size == outcomes.size
    assert all(outcomes[fold].min() == 0 and outcomes[fold].max() == 1 for fold in first)


def test_nested_cross_fitting_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(120, 3))
    outcome_probability = 1.0 / (1.0 + np.exp(-(values[:, 0] - values[:, 1])))
    outcomes = (rng.random(120) < outcome_probability).astype(float)

    first, lambdas_first = nested_oof_predictions(
        values, outcomes, ("a", "b", "c"), seed=123
    )
    second, lambdas_second = nested_oof_predictions(
        values, outcomes, ("a", "b", "c"), seed=123
    )

    assert np.array_equal(first, second)
    assert lambdas_first == lambdas_second
    assert np.all((first > 0.0) & (first < 1.0))


def test_calibration_statistics_are_finite() -> None:
    outcomes = np.asarray([0, 0, 0, 1, 1, 1], dtype=float)
    probabilities = np.asarray([0.05, 0.2, 0.3, 0.6, 0.8, 0.95], dtype=float)

    intercept, slope, brier = calibration_statistics(outcomes, probabilities)

    assert np.isfinite([intercept, slope, brier]).all()
    assert 0.0 <= brier <= 1.0
