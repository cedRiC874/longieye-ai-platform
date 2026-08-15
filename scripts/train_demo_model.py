"""Train a reproducible two-head logistic model on synthetic data only."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from longieye.features import FEATURE_ORDER  # noqa: E402


ARTIFACT_SIGNIFICANT_DIGITS = 12


def canonicalize_artifact_numbers(value: Any) -> Any:
    """Remove insignificant cross-platform libm noise before JSON serialization."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Model artifacts cannot contain non-finite numbers")
        return float(format(value, f".{ARTIFACT_SIGNIFICANT_DIGITS}g"))
    if isinstance(value, list):
        return [canonicalize_artifact_numbers(item) for item in value]
    if isinstance(value, dict):
        return {
            key: canonicalize_artifact_numbers(item) for key, item in value.items()
        }
    return value


def sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def sample_feature_vector(rng: random.Random) -> list[float]:
    return [
        float(rng.randint(0, 1)),
        rng.gauss(5.0, 2.5),
        rng.gauss(4.0, 2.8),
        rng.gauss(2.0, 7.0),
        rng.gauss(1.0, 5.0),
        rng.gauss(3.0, 3.5),
        float(rng.choices([-1, 0, 1], weights=[1, 14, 5], k=1)[0]),
        max(-0.05, rng.gauss(0.23, 0.12)),
        max(-0.05, rng.gauss(0.22, 0.12)),
    ]


def synthetic_logits(values: list[float]) -> tuple[float, float]:
    sex, _, weight, sbp, _, _, glasses, al_od, al_os = values
    common = 0.18 * (sex - 0.5) + 0.025 * (weight - 4.0) + 0.01 * (sbp - 2.0)
    od = -2.35 + common + 0.28 * glasses + 5.2 * (al_od - 0.23) + 1.5 * (
        al_os - 0.22
    )
    os = -2.38 + common + 0.28 * glasses + 1.5 * (al_od - 0.23) + 5.2 * (
        al_os - 0.22
    )
    return od, os


def generate_dataset(
    n_samples: int, seed: int
) -> tuple[list[list[float]], list[list[int]]]:
    rng = random.Random(seed)
    features: list[list[float]] = []
    targets: list[list[int]] = []
    for _ in range(n_samples):
        values = sample_feature_vector(rng)
        logits = synthetic_logits(values)
        labels = [int(rng.random() < sigmoid(logit)) for logit in logits]
        features.append(values)
        targets.append(labels)
    return features, targets


def fit_normalization(values: list[list[float]]) -> tuple[list[float], list[float]]:
    means = [sum(row[index] for row in values) / len(values) for index in range(9)]
    variances = [
        sum((row[index] - means[index]) ** 2 for row in values) / len(values)
        for index in range(9)
    ]
    stds = [max(math.sqrt(value), 1e-6) for value in variances]
    return means, stds


def standardize(
    values: list[list[float]], means: list[float], stds: list[float]
) -> list[list[float]]:
    return [
        [
            (value - means[index]) / stds[index]
            for index, value in enumerate(row)
        ]
        for row in values
    ]


def fit_logistic(
    values: list[list[float]],
    targets: list[int],
    epochs: int = 900,
    learning_rate: float = 0.08,
    l2: float = 0.002,
) -> tuple[float, list[float]]:
    weights = [0.0] * len(values[0])
    prevalence = min(0.999, max(0.001, sum(targets) / len(targets)))
    intercept = math.log(prevalence / (1.0 - prevalence))
    for _ in range(epochs):
        gradient = [0.0] * len(weights)
        intercept_gradient = 0.0
        for row, target in zip(values, targets, strict=True):
            probability = sigmoid(
                intercept + sum(w * x for w, x in zip(weights, row, strict=True))
            )
            error = probability - target
            intercept_gradient += error
            for index, value in enumerate(row):
                gradient[index] += error * value
        scale = 1.0 / len(values)
        intercept -= learning_rate * intercept_gradient * scale
        for index in range(len(weights)):
            weights[index] -= learning_rate * (
                gradient[index] * scale + l2 * weights[index]
            )
    return intercept, weights


def auc_score(targets: list[int], probabilities: list[float]) -> float:
    positives = [(p, y) for p, y in zip(probabilities, targets, strict=True) if y == 1]
    negatives = [(p, y) for p, y in zip(probabilities, targets, strict=True) if y == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for positive, _ in positives:
        for negative, _ in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def evaluate(
    values: list[list[float]], targets: list[int], intercept: float, weights: list[float]
) -> dict[str, float]:
    probabilities = [
        sigmoid(intercept + sum(w * x for w, x in zip(weights, row, strict=True)))
        for row in values
    ]
    brier = sum(
        (probability - target) ** 2
        for probability, target in zip(probabilities, targets, strict=True)
    ) / len(targets)
    return {"auc": auc_score(targets, probabilities), "brier": brier}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "configs" / "demo_model.json"
    )
    args = parser.parse_args()
    if args.samples < 500:
        raise ValueError("Use at least 500 synthetic samples for a stable demo")

    values, targets = generate_dataset(args.samples, args.seed)
    split = int(args.samples * 0.8)
    train_values, validation_values = values[:split], values[split:]
    train_targets, validation_targets = targets[:split], targets[split:]
    means, stds = fit_normalization(train_values)
    train_standardized = standardize(train_values, means, stds)
    validation_standardized = standardize(validation_values, means, stds)

    heads: dict[str, dict[str, object]] = {}
    metrics: dict[str, dict[str, float]] = {}
    validation_reference: dict[str, dict[str, float]] = {}
    for index, eye in enumerate(("od", "os")):
        eye_targets = [row[index] for row in train_targets]
        intercept, coefficients = fit_logistic(train_standardized, eye_targets)
        heads[eye] = {"intercept": intercept, "coefficients": coefficients}
        metrics[eye] = evaluate(
            validation_standardized,
            [row[index] for row in validation_targets],
            intercept,
            coefficients,
        )
        validation_eye_targets = [row[index] for row in validation_targets]
        prevalence = sum(validation_eye_targets) / len(validation_eye_targets)
        validation_reference[eye] = {
            "positive_rate": prevalence,
            "constant_probability_brier": prevalence * (1.0 - prevalence),
        }

    payload = {
        "feature_order": list(FEATURE_ORDER),
        "normalization": {"mean": means, "std": stds},
        "heads": heads,
        "metadata": {
            "model_id": "longieye-synthetic-static-sex-delta8-v0",
            "model_stage": "demo_synthetic",
            "training_data": "deterministic synthetic data only",
            "synthetic_seed": args.seed,
            "training_samples": len(train_values),
            "validation_samples": len(validation_values),
            "validation_metrics": metrics,
            "validation_reference": validation_reference,
            "clinical_use": False,
        },
    }
    payload = canonicalize_artifact_numbers(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    args.output.write_text(
        serialized + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
