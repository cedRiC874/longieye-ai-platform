"""Reproducible internal validation for the public OLSM myopia cohort.

This module is intentionally separate from the public FastAPI service.  The
cohort has baseline predictors and a five-year incident-myopia outcome, not the
Y1/Y2 nine-feature contract used by the synthetic service.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np


EXPECTED_HEADERS: Final[tuple[str, ...]] = (
    "ID",
    "STUDYYEAR",
    "MYOPIC",
    "AGE",
    "GENDER",
    "SPHEQ",
    "AL",
    "ACD",
    "LT",
    "VCD",
    "SPORTHR",
    "READHR",
    "COMPHR",
    "STUDYHR",
    "TVHR",
    "DIOPTERHR",
    "MOMMY",
    "DADMY",
)

FEATURE_SETS: Final[dict[str, tuple[str, ...]]] = {
    "refraction_only": ("spheq",),
    "ocular": ("age", "gender", "spheq", "al", "acd", "lt", "vcd"),
    "full": (
        "age",
        "gender",
        "spheq",
        "al",
        "acd",
        "lt",
        "vcd",
        "sporthr",
        "readhr",
        "comphr",
        "studyhr",
        "tvhr",
        "mommy",
        "dadmy",
    ),
}

RIDGE_LAMBDAS: Final[tuple[float, ...]] = (0.01, 0.1, 1.0, 10.0, 100.0)
PRIMARY_MODEL: Final[str] = "full"


class PublicValidationError(ValueError):
    """Raised when the public cohort or validation contract is invalid."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CohortData:
    ids: np.ndarray
    study_year: np.ndarray
    outcome: np.ndarray
    columns: dict[str, np.ndarray]

    @property
    def size(self) -> int:
        return int(self.outcome.size)

    @property
    def events(self) -> int:
        return int(self.outcome.sum())


@dataclass(frozen=True)
class FittedPipeline:
    feature_names: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    intercept: float
    ridge_lambda: float

    def predict(self, raw_values: np.ndarray) -> np.ndarray:
        transformed = _transform(raw_values, self.medians, self.means, self.scales)
        return sigmoid(self.intercept + transformed @ self.coefficients)


def _parse_number(raw: str, *, field: str, row_number: int) -> float:
    if raw.strip() == "":
        return math.nan
    try:
        value = float(raw)
    except ValueError:
        raise PublicValidationError(
            "cohort_value_invalid", f"Invalid {field} at row {row_number}."
        ) from None
    if not math.isfinite(value):
        raise PublicValidationError(
            "cohort_value_invalid", f"Non-finite {field} at row {row_number}."
        )
    return value


def load_olsm_tsv(path: Path) -> CohortData:
    """Read a local OLSM TSV and apply the strict cohort contract."""
    try:
        contents = path.read_bytes()
    except OSError:
        raise PublicValidationError(
            "cohort_read_failed", "The public cohort file could not be read."
        ) from None
    return load_olsm_tsv_bytes(contents)


def load_olsm_tsv_bytes(contents: bytes) -> CohortData:
    """Parse already-verified OLSM bytes without reopening a mutable path."""

    try:
        text = contents.decode("utf-8")
    except UnicodeError:
        raise PublicValidationError(
            "cohort_read_failed", "The public cohort file is not UTF-8."
        ) from None
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t")
    if tuple(reader.fieldnames or ()) != EXPECTED_HEADERS:
        raise PublicValidationError(
            "cohort_schema_invalid", "The public cohort header contract changed."
        )
    records = list(reader)

    if len(records) != 618:
        raise PublicValidationError(
            "cohort_size_invalid", "The public cohort row count changed."
        )

    ids: list[int] = []
    years: list[int] = []
    outcomes: list[int] = []
    numeric_names = (
        "age",
        "gender",
        "spheq",
        "al",
        "acd",
        "lt",
        "vcd",
        "sporthr",
        "readhr",
        "comphr",
        "studyhr",
        "tvhr",
        "diopterhr",
        "mommy",
        "dadmy",
    )
    values: dict[str, list[float]] = {name: [] for name in numeric_names}

    source_map = {name.lower(): name for name in EXPECTED_HEADERS}
    for row_number, record in enumerate(records, start=2):
        if None in record or any(record.get(header) is None for header in EXPECTED_HEADERS):
            raise PublicValidationError(
                "cohort_schema_invalid", f"Malformed column count at row {row_number}."
            )
        try:
            subject_id = int(record["ID"])
            study_year = int(record["STUDYYEAR"])
            outcome = int(record["MYOPIC"])
        except (TypeError, ValueError):
            raise PublicValidationError(
                "cohort_value_invalid", f"Invalid identifier/outcome at row {row_number}."
            ) from None
        if subject_id <= 0 or not 1990 <= study_year <= 1995 or outcome not in {0, 1}:
            raise PublicValidationError(
                "cohort_value_invalid", f"Out-of-contract value at row {row_number}."
            )
        ids.append(subject_id)
        years.append(study_year)
        outcomes.append(outcome)
        for name in numeric_names:
            values[name].append(
                _parse_number(record[source_map[name]], field=name, row_number=row_number)
            )

    if len(set(ids)) != len(ids):
        raise PublicValidationError(
            "cohort_id_invalid", "Participant identifiers are not unique."
        )
    if sum(outcomes) != 81:
        raise PublicValidationError(
            "cohort_event_count_invalid", "The public cohort event count changed."
        )

    columns = {name: np.asarray(column, dtype=float) for name, column in values.items()}
    for binary_name in ("gender", "mommy", "dadmy"):
        observed = columns[binary_name][np.isfinite(columns[binary_name])]
        if not np.isin(observed, (0.0, 1.0)).all():
            raise PublicValidationError(
                "cohort_value_invalid", f"{binary_name} is not binary."
            )

    return CohortData(
        ids=np.asarray(ids, dtype=int),
        study_year=np.asarray(years, dtype=int),
        outcome=np.asarray(outcomes, dtype=float),
        columns=columns,
    )


def feature_matrix(data: CohortData, feature_names: tuple[str, ...]) -> np.ndarray:
    try:
        return np.column_stack([data.columns[name] for name in feature_names])
    except KeyError:
        raise PublicValidationError(
            "feature_contract_invalid", "Unknown public-cohort feature."
        ) from None


def sigmoid(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.empty_like(array)
    nonnegative = array >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-array[nonnegative]))
    exponent = np.exp(array[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0 - 1e-8)
    return np.log(clipped / (1.0 - clipped))


def auc_score(outcomes: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(outcomes, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        raise PublicValidationError(
            "metric_undefined", "AUC requires both outcome classes."
        )
    order = np.argsort(p, kind="mergesort")
    sorted_values = p[order]
    ranks = np.empty(y.size, dtype=float)
    start = 0
    while start < y.size:
        stop = start + 1
        while stop < y.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    positive_rank_sum = float(ranks[y == 1].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def brier_score(outcomes: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities) - np.asarray(outcomes)) ** 2))


def log_loss(outcomes: np.ndarray, probabilities: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
    y = np.asarray(outcomes, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _prepare(
    train_values: np.ndarray, test_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(train_values, axis=0)
    if not np.isfinite(medians).all():
        raise PublicValidationError(
            "feature_missing_all", "A training fold contains an all-missing feature."
        )
    train_imputed = np.where(np.isnan(train_values), medians, train_values)
    test_imputed = np.where(np.isnan(test_values), medians, test_values)
    means = train_imputed.mean(axis=0)
    scales = train_imputed.std(axis=0, ddof=0)
    scales = np.where(scales < 1e-12, 1.0, scales)
    return (
        (train_imputed - means) / scales,
        (test_imputed - means) / scales,
        medians,
        means,
        scales,
    )


def _transform(
    raw_values: np.ndarray,
    medians: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    imputed = np.where(np.isnan(raw_values), medians, raw_values)
    return (imputed - means) / scales


def _fit_design(
    design: np.ndarray,
    outcomes: np.ndarray,
    *,
    ridge_lambda: float,
    penalize_intercept: bool = False,
    max_iterations: int = 100,
) -> np.ndarray:
    x = np.asarray(design, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    coefficients = np.zeros(x.shape[1], dtype=float)
    penalty = np.ones(x.shape[1], dtype=float)
    if not penalize_intercept:
        penalty[0] = 0.0

    def objective(beta: np.ndarray) -> float:
        eta = x @ beta
        likelihood = np.logaddexp(0.0, eta).sum() - float(y @ eta)
        return float(likelihood + 0.5 * ridge_lambda * np.sum(penalty * beta**2))

    current_objective = objective(coefficients)
    for _ in range(max_iterations):
        probabilities = sigmoid(x @ coefficients)
        weights = np.clip(probabilities * (1.0 - probabilities), 1e-8, None)
        gradient = x.T @ (probabilities - y) + ridge_lambda * penalty * coefficients
        hessian = (x.T * weights) @ x + ridge_lambda * np.diag(penalty)
        hessian += np.eye(hessian.shape[0]) * 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if float(np.max(np.abs(step))) < 1e-9:
            break
        multiplier = 1.0
        accepted = False
        while multiplier >= 1e-6:
            candidate = coefficients - multiplier * step
            candidate_objective = objective(candidate)
            if candidate_objective <= current_objective + 1e-12:
                coefficients = candidate
                current_objective = candidate_objective
                accepted = True
                break
            multiplier *= 0.5
        if not accepted:
            break
    if not np.isfinite(coefficients).all():
        raise PublicValidationError("model_fit_failed", "Model fitting was non-finite.")
    return coefficients


def fit_pipeline(
    raw_values: np.ndarray,
    outcomes: np.ndarray,
    feature_names: tuple[str, ...],
    ridge_lambda: float,
) -> FittedPipeline:
    prepared, _, medians, means, scales = _prepare(raw_values, raw_values)
    design = np.column_stack([np.ones(prepared.shape[0]), prepared])
    fitted = _fit_design(design, outcomes, ridge_lambda=ridge_lambda)
    return FittedPipeline(
        feature_names=feature_names,
        medians=medians,
        means=means,
        scales=scales,
        intercept=float(fitted[0]),
        coefficients=fitted[1:],
        ridge_lambda=float(ridge_lambda),
    )


def stratified_folds(
    outcomes: np.ndarray, folds: int, seed: int
) -> tuple[np.ndarray, ...]:
    y = np.asarray(outcomes, dtype=int)
    if folds < 2 or min(int(y.sum()), int(y.size - y.sum())) < folds:
        raise PublicValidationError(
            "fold_contract_invalid", "Each fold requires both outcome classes."
        )
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(folds)]
    for value in (0, 1):
        indices = np.flatnonzero(y == value)
        rng.shuffle(indices)
        for position, index in enumerate(indices.tolist()):
            buckets[position % folds].append(index)
    result: list[np.ndarray] = []
    for bucket in buckets:
        array = np.asarray(bucket, dtype=int)
        rng.shuffle(array)
        result.append(array)
    combined = np.concatenate(result)
    if np.unique(combined).size != y.size:
        raise PublicValidationError("fold_contract_invalid", "Fold overlap detected.")
    return tuple(result)


def _select_lambda(
    raw_values: np.ndarray,
    outcomes: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    seed: int,
    folds: int,
) -> float:
    split = stratified_folds(outcomes, folds, seed)
    all_indices = np.arange(outcomes.size)
    losses: list[tuple[float, float]] = []
    for ridge_lambda in RIDGE_LAMBDAS:
        fold_losses: list[float] = []
        for validation_indices in split:
            train_indices = np.setdiff1d(all_indices, validation_indices, assume_unique=True)
            pipeline = fit_pipeline(
                raw_values[train_indices],
                outcomes[train_indices],
                feature_names,
                ridge_lambda,
            )
            fold_losses.append(
                log_loss(
                    outcomes[validation_indices],
                    pipeline.predict(raw_values[validation_indices]),
                )
            )
        losses.append((float(np.mean(fold_losses)), ridge_lambda))
    losses.sort(key=lambda item: (round(item[0], 12), item[1]))
    return float(losses[0][1])


def nested_oof_predictions(
    raw_values: np.ndarray,
    outcomes: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    seed: int,
    outer_folds: int = 5,
    inner_folds: int = 5,
) -> tuple[np.ndarray, list[float]]:
    outer = stratified_folds(outcomes, outer_folds, seed)
    all_indices = np.arange(outcomes.size)
    predictions = np.full(outcomes.size, np.nan, dtype=float)
    selected: list[float] = []
    for fold_index, validation_indices in enumerate(outer):
        train_indices = np.setdiff1d(all_indices, validation_indices, assume_unique=True)
        ridge_lambda = _select_lambda(
            raw_values[train_indices],
            outcomes[train_indices],
            feature_names,
            seed=seed + 1000 + fold_index,
            folds=inner_folds,
        )
        selected.append(ridge_lambda)
        pipeline = fit_pipeline(
            raw_values[train_indices],
            outcomes[train_indices],
            feature_names,
            ridge_lambda,
        )
        predictions[validation_indices] = pipeline.predict(raw_values[validation_indices])
    if not np.isfinite(predictions).all():
        raise PublicValidationError(
            "oof_prediction_invalid", "OOF predictions are incomplete."
        )
    return predictions, selected


def calibration_statistics(
    outcomes: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float, float]:
    y = np.asarray(outcomes, dtype=float)
    linear_predictor = logit(probabilities)
    offset_intercept = 0.0
    for _ in range(100):
        p = sigmoid(offset_intercept + linear_predictor)
        gradient = float(np.sum(p - y))
        hessian = float(np.sum(np.clip(p * (1.0 - p), 1e-8, None)))
        step = gradient / hessian
        offset_intercept -= step
        if abs(step) < 1e-10:
            break
    design = np.column_stack([np.ones(y.size), linear_predictor])
    fitted = _fit_design(design, y, ridge_lambda=1e-8)
    return float(offset_intercept), float(fitted[1]), brier_score(y, probabilities)


def decision_curve_net_benefit(
    outcomes: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    y = np.asarray(outcomes, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    result: list[float] = []
    for threshold in thresholds:
        predicted_positive = p >= threshold
        true_positive = int(np.sum(predicted_positive & (y == 1)))
        false_positive = int(np.sum(predicted_positive & (y == 0)))
        result.append(
            true_positive / y.size
            - false_positive / y.size * threshold / (1.0 - threshold)
        )
    return np.asarray(result, dtype=float)


def treat_all_net_benefit(outcomes: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    prevalence = float(np.mean(outcomes))
    return prevalence - (1.0 - prevalence) * thresholds / (1.0 - thresholds)


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    array = np.asarray(p_values, dtype=float)
    if np.any((array < 0.0) | (array > 1.0) | ~np.isfinite(array)):
        raise PublicValidationError("p_value_invalid", "P-values must be in [0, 1].")
    order = np.argsort(array)
    adjusted_sorted = np.empty(array.size, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (array.size - rank) * array[index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty(array.size, dtype=float)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted.tolist()


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise PublicValidationError("bootstrap_failed", "No finite bootstrap values.")
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return float(lower), float(upper)


def paired_permutation_auc_p_value(
    outcomes: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> float:
    """Two-sided paired randomization test for an AUC difference."""

    if repetitions < 1000:
        raise PublicValidationError(
            "permutation_contract_invalid", "At least 1000 permutations are required."
        )
    observed = auc_score(outcomes, candidate) - auc_score(outcomes, reference)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(repetitions):
        swap = rng.random(outcomes.size) < 0.5
        permuted_candidate = np.where(swap, reference, candidate)
        permuted_reference = np.where(swap, candidate, reference)
        difference = auc_score(outcomes, permuted_candidate) - auc_score(
            outcomes, permuted_reference
        )
        if abs(difference) >= abs(observed) - 1e-15:
            extreme += 1
    return float((extreme + 1) / (repetitions + 1))


def calibration_bins(
    outcomes: np.ndarray, probabilities: np.ndarray, bins: int = 8
) -> list[dict[str, float | int | list[float]]]:
    order = np.argsort(probabilities, kind="mergesort")
    groups = np.array_split(order, bins)
    result: list[dict[str, float | int | list[float]]] = []
    for index, group in enumerate(groups, start=1):
        if group.size == 0:
            continue
        result.append(
            {
                "bin": index,
                "n": int(group.size),
                "predicted_mean": float(np.mean(probabilities[group])),
                "observed_rate": float(np.mean(outcomes[group])),
            }
        )
    return result


def run_internal_validation(
    data: CohortData,
    *,
    seed: int = 20260831,
    bootstrap_repetitions: int = 2000,
    crossfit_repetitions: int = 5,
    permutation_repetitions: int = 10_000,
) -> tuple[dict[str, object], dict[str, FittedPipeline]]:
    if bootstrap_repetitions < 200:
        raise PublicValidationError(
            "bootstrap_contract_invalid", "At least 200 bootstrap repetitions are required."
        )
    if crossfit_repetitions < 2:
        raise PublicValidationError(
            "crossfit_contract_invalid", "Repeated cross-fitting requires two runs."
        )
    if permutation_repetitions < 1000:
        raise PublicValidationError(
            "permutation_contract_invalid", "At least 1000 permutations are required."
        )
    outcomes = data.outcome
    predictions: dict[str, np.ndarray] = {}
    selected_lambdas: dict[str, list[list[float]]] = {}
    repeat_aucs: dict[str, list[float]] = {}
    raw_matrices: dict[str, np.ndarray] = {}

    for model_name, features in FEATURE_SETS.items():
        raw = feature_matrix(data, features)
        raw_matrices[model_name] = raw
        repeated_predictions: list[np.ndarray] = []
        repeated_lambdas: list[list[float]] = []
        for repetition in range(crossfit_repetitions):
            model_predictions, selected = nested_oof_predictions(
                raw, outcomes, features, seed=seed + repetition
            )
            repeated_predictions.append(model_predictions)
            repeated_lambdas.append(selected)
        predictions[model_name] = np.mean(repeated_predictions, axis=0)
        selected_lambdas[model_name] = repeated_lambdas
        repeat_aucs[model_name] = [
            auc_score(outcomes, values) for values in repeated_predictions
        ]

    rng = np.random.default_rng(seed + 1)
    bootstrap_indices = rng.integers(
        0, data.size, size=(bootstrap_repetitions, data.size), dtype=np.int32
    )
    bootstrap_auc = {
        name: np.empty(bootstrap_repetitions, dtype=float) for name in FEATURE_SETS
    }
    bootstrap_brier = {
        name: np.empty(bootstrap_repetitions, dtype=float) for name in FEATURE_SETS
    }
    primary_calibration_intercept = np.empty(bootstrap_repetitions, dtype=float)
    primary_calibration_slope = np.empty(bootstrap_repetitions, dtype=float)
    calibration_bin_count = 8
    calibration_bin_ids = np.empty(data.size, dtype=int)
    for bin_index, group in enumerate(
        np.array_split(
            np.argsort(predictions[PRIMARY_MODEL], kind="mergesort"),
            calibration_bin_count,
        )
    ):
        calibration_bin_ids[group] = bin_index
    bootstrap_calibration_rate = np.empty(
        (bootstrap_repetitions, calibration_bin_count), dtype=float
    )

    thresholds = np.round(np.arange(0.03, 0.301, 0.01), 2)
    primary_dca = np.empty((bootstrap_repetitions, thresholds.size), dtype=float)

    for repetition, indices in enumerate(bootstrap_indices):
        sampled_outcome = outcomes[indices]
        if sampled_outcome.min() == sampled_outcome.max():
            for values in bootstrap_auc.values():
                values[repetition] = math.nan
            for values in bootstrap_brier.values():
                values[repetition] = math.nan
            primary_calibration_intercept[repetition] = math.nan
            primary_calibration_slope[repetition] = math.nan
            bootstrap_calibration_rate[repetition, :] = math.nan
            primary_dca[repetition, :] = math.nan
            continue
        for model_name in FEATURE_SETS:
            sampled_prediction = predictions[model_name][indices]
            bootstrap_auc[model_name][repetition] = auc_score(
                sampled_outcome, sampled_prediction
            )
            bootstrap_brier[model_name][repetition] = brier_score(
                sampled_outcome, sampled_prediction
            )
        calibration_intercept, calibration_slope, _ = calibration_statistics(
            sampled_outcome, predictions[PRIMARY_MODEL][indices]
        )
        primary_calibration_intercept[repetition] = calibration_intercept
        primary_calibration_slope[repetition] = calibration_slope
        sampled_bin_ids = calibration_bin_ids[indices]
        for bin_index in range(calibration_bin_count):
            bin_outcomes = sampled_outcome[sampled_bin_ids == bin_index]
            bootstrap_calibration_rate[repetition, bin_index] = (
                float(np.mean(bin_outcomes)) if bin_outcomes.size else math.nan
            )
        primary_dca[repetition, :] = decision_curve_net_benefit(
            sampled_outcome, predictions[PRIMARY_MODEL][indices], thresholds
        )

    model_results: dict[str, object] = {}
    for model_name, features in FEATURE_SETS.items():
        calibration_intercept, calibration_slope, brier = calibration_statistics(
            outcomes, predictions[model_name]
        )
        auc = auc_score(outcomes, predictions[model_name])
        model_results[model_name] = {
            "features": list(features),
            "selected_ridge_lambdas_by_crossfit_repeat": selected_lambdas[
                model_name
            ],
            "auc": auc,
            "auc_ci_95": list(percentile_interval(bootstrap_auc[model_name])),
            "brier": brier,
            "brier_ci_95": list(percentile_interval(bootstrap_brier[model_name])),
            "calibration_intercept": calibration_intercept,
            "calibration_slope": calibration_slope,
        }

    comparisons_spec = (
        ("ocular_vs_refraction_only", "ocular", "refraction_only"),
        ("full_vs_refraction_only", "full", "refraction_only"),
        ("full_vs_ocular", "full", "ocular"),
    )
    raw_p_values: list[float] = []
    comparison_rows: list[dict[str, object]] = []
    for comparison_index, (name, candidate, reference) in enumerate(comparisons_spec):
        differences = bootstrap_auc[candidate] - bootstrap_auc[reference]
        differences = differences[np.isfinite(differences)]
        raw_p = paired_permutation_auc_p_value(
            outcomes,
            predictions[candidate],
            predictions[reference],
            repetitions=permutation_repetitions,
            seed=seed + 5000 + comparison_index,
        )
        raw_p_values.append(raw_p)
        comparison_rows.append(
            {
                "comparison": name,
                "candidate": candidate,
                "reference": reference,
                "delta_auc": auc_score(outcomes, predictions[candidate])
                - auc_score(outcomes, predictions[reference]),
                "delta_auc_ci_95": list(percentile_interval(differences)),
                "p_value_raw": raw_p,
            }
        )
    adjusted = holm_adjust(raw_p_values)
    for row, adjusted_p in zip(comparison_rows, adjusted, strict=True):
        row["p_value_holm"] = adjusted_p
        row["reject_fwer_0_05"] = bool(adjusted_p < 0.05)

    primary_probabilities = predictions[PRIMARY_MODEL]
    dca_models = {
        name: decision_curve_net_benefit(outcomes, probabilities, thresholds)
        for name, probabilities in predictions.items()
    }
    dca_lower = np.nanquantile(primary_dca, 0.025, axis=0)
    dca_upper = np.nanquantile(primary_dca, 0.975, axis=0)
    calibration_curve = calibration_bins(
        outcomes, primary_probabilities, bins=calibration_bin_count
    )
    for bin_index, row in enumerate(calibration_curve):
        row["observed_rate_ci_95"] = list(
            percentile_interval(bootstrap_calibration_rate[:, bin_index])
        )

    temporal_train = np.flatnonzero(data.study_year <= 1992)
    temporal_test = np.flatnonzero(data.study_year >= 1993)
    temporal_results: dict[str, object] = {}
    for model_name, features in FEATURE_SETS.items():
        raw = raw_matrices[model_name]
        ridge_lambda = _select_lambda(
            raw[temporal_train],
            outcomes[temporal_train],
            features,
            seed=seed + 2000,
            folds=5,
        )
        pipeline = fit_pipeline(
            raw[temporal_train], outcomes[temporal_train], features, ridge_lambda
        )
        temporal_prediction = pipeline.predict(raw[temporal_test])
        temporal_results[model_name] = {
            "ridge_lambda": ridge_lambda,
            "auc": auc_score(outcomes[temporal_test], temporal_prediction),
            "brier": brier_score(outcomes[temporal_test], temporal_prediction),
        }

    sensitivity_aucs = repeat_aucs[PRIMARY_MODEL]

    final_models: dict[str, FittedPipeline] = {}
    for model_name, features in FEATURE_SETS.items():
        ridge_lambda = _select_lambda(
            raw_matrices[model_name],
            outcomes,
            features,
            seed=seed + 3000,
            folds=5,
        )
        final_models[model_name] = fit_pipeline(
            raw_matrices[model_name], outcomes, features, ridge_lambda
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "analysis_id": "olsm-five-year-myopia-internal-validation-v1",
        "analysis_date": "2026-08-31",
        "dataset": {
            "participants": data.size,
            "events": data.events,
            "non_events": data.size - data.events,
            "missing_cells": int(
                sum(np.isnan(column).sum() for column in data.columns.values())
            ),
        },
        "task": {
            "prediction_time": "initial examination",
            "outcome": "incident myopia at any time during the first five years",
            "unit": "participant; right-eye baseline ocular measurements",
            "analysis_type": "pilot internal validation",
        },
        "validation": {
            "outer_folds": 5,
            "inner_folds": 5,
            "seed": seed,
            "bootstrap_repetitions": bootstrap_repetitions,
            "bootstrap_unit": "participant",
            "bootstrap_scope": "conditional on repeated OOF predictions; models are not refit inside bootstrap samples",
            "crossfit_repetitions": crossfit_repetitions,
            "crossfit_seeds": list(range(seed, seed + crossfit_repetitions)),
            "permutation_repetitions": permutation_repetitions,
            "tuning_metric": "log_loss",
            "ridge_lambda_grid": list(RIDGE_LAMBDAS),
            "primary_estimand": "pooled participant-level AUC after averaging five repeated OOF probabilities; each fold uses its own fitted pipeline",
            "oof_predictions_persisted": False,
        },
        "models": model_results,
        "primary_model": PRIMARY_MODEL,
        "primary_calibration_ci_95": {
            "intercept": list(percentile_interval(primary_calibration_intercept)),
            "slope": list(percentile_interval(primary_calibration_slope)),
        },
        "calibration_curve": calibration_curve,
        "dca": {
            "status": "exploratory_hypothetical_monitoring_action",
            "thresholds": thresholds.tolist(),
            "model_net_benefit": {
                name: values.tolist() for name, values in dca_models.items()
            },
            "primary_ci_95": {
                "lower": dca_lower.tolist(),
                "upper": dca_upper.tolist(),
            },
            "treat_all": treat_all_net_benefit(outcomes, thresholds).tolist(),
            "treat_none": np.zeros(thresholds.size).tolist(),
        },
        "model_comparisons": comparison_rows,
        "multiplicity": {
            "family": "three analysis-version-defined paired AUC comparisons",
            "raw_p_value_method": "paired participant-level randomization test",
            "method": "Holm family-wise error control",
            "alpha": 0.05,
        },
        "entry_cohort_chronological_split_sensitivity": {
            "train_years": "1990-1992",
            "test_years": "1993-1995",
            "train_participants": int(temporal_train.size),
            "train_events": int(outcomes[temporal_train].sum()),
            "test_participants": int(temporal_test.size),
            "test_events": int(outcomes[temporal_test].sum()),
            "models": temporal_results,
        },
        "split_sensitivity": {
            "model": PRIMARY_MODEL,
            "seeds": list(range(seed, seed + crossfit_repetitions)),
            "auc_values": sensitivity_aucs,
            "auc_min": float(min(sensitivity_aucs)),
            "auc_median": float(np.median(sensitivity_aucs)),
            "auc_max": float(max(sensitivity_aucs)),
        },
        "limitations": [
            "Only 81 events; results are pilot internal validation, not confirmation.",
            "The public subset contains baseline predictors and a longitudinal outcome, not repeated Y1/Y2 predictors.",
            "No external cohort was available; cross-validation is not external validation.",
            "DCA uses a hypothetical enhanced-monitoring action and does not establish clinical utility.",
            "The dataset is an OLSM subset distributed through an unofficial companion R package.",
            "The pooled cross-fitted AUC compares probabilities produced by fold-specific fitted pipelines.",
        ],
    }
    return report, final_models


def serialize_pipeline(model: FittedPipeline) -> dict[str, object]:
    return {
        "feature_order": list(model.feature_names),
        "ridge_lambda": model.ridge_lambda,
        "preprocessing": {
            "median": model.medians.tolist(),
            "mean": model.means.tolist(),
            "scale": model.scales.tolist(),
        },
        "head": {
            "intercept": model.intercept,
            "coefficients": model.coefficients.tolist(),
        },
    }
