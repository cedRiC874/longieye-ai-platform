# Model card: LongiEye public synthetic static-sex + delta8 demo

## Summary

| Field | Value |
| --- | --- |
| Model ID | `longieye-synthetic-static-sex-delta8-v0` |
| Stage | `demo_synthetic` |
| Task | Two independent OD/OS binary demonstration scores |
| Algorithm | Standardized logistic regression with two heads |
| Training source | 1,600 deterministic synthetic records |
| Validation source | 400 deterministic synthetic records |
| Clinical use | **No** |

This model exists to demonstrate a reproducible service boundary. It is not the thesis model and does not reproduce thesis performance.

## Inputs

The service derives one static value and eight Y2-minus-Y1 changes:

1. Y1 sex encoding;
2. height change;
3. weight change;
4. systolic blood-pressure change;
5. diastolic blood-pressure change;
6. waist change;
7. glasses-wearing change;
8. OD axial-length change;
9. OS axial-length change.

The API intentionally excludes spherical-equivalent predictors, participant identifiers, image paths and clinical free text.

## Outputs

The two values named `demo_probability` are synthetic-model outputs between 0 and 1. They are not clinically calibrated probabilities and have no diagnostic threshold.

## Synthetic validation results

| Head | Positive rate | Synthetic AUC | Model Brier | Constant-probability Brier |
| --- | ---: | ---: | ---: | ---: |
| OD | 0.1150 | 0.7020 | 0.0961 | 0.1018 |
| OS | 0.1125 | 0.6167 | 0.0992 | 0.0998 |

The exact values are stored in `configs/demo_model.json` and regenerated with the model. These numbers test whether the software artifact can learn its synthetic data-generating pattern. They must not be quoted as medical performance or compared directly with a clinical cohort.

## Relationship to the source research

The feature contract was informed by a private thesis repository, but this model card intentionally contains no thesis performance number. A research metric may be published only after it is mapped to one named experiment, a fixed source commit, an aggregate result file, its calculation script and the corresponding thesis table.

Different cohorts, modalities, ablations and validation designs must remain separate. Research results belong in an authorization-reviewed card created from the [research model-card template](RESEARCH_MODEL_CARD_TEMPLATE.md), not in this synthetic model card. The default research-artifact status is `NOT_AUTHORIZED`; no research checkpoint, preprocessing statistics, participant-level record or OOF prediction is packaged here.

## Intended uses

- Demonstrating API, validation, observability and reproducibility practices.
- Exercising a stable model-adapter contract before an authorized research export.
- Teaching how to separate research claims from engineering-demo claims.

## Out-of-scope uses

- Diagnosis, screening, treatment or patient communication.
- Ranking real people by risk.
- Comparing demographic groups or drawing fairness conclusions.
- Replacing professional ophthalmic assessment.

## Limitations and risks

- Synthetic rows cannot represent clinical prevalence, missingness, acquisition shift or subgroup behavior.
- The binary `sex_code` mirrors a historical research encoding and does not represent gender identity.
- The simple logistic form cannot represent the multimodal image branch.
- The local benchmark measures process RSS and Python allocation peaks, but it does not characterize container limits, accelerator memory or production resource behavior.
- The local benchmark excludes network, reverse-proxy and concurrent-load overhead.

## Reproducibility

```powershell
python scripts/train_demo_model.py
python -m pytest -q
python scripts/benchmark_service.py
```

The default synthetic seed is `20260815`. Artifact floats are serialized to 12 significant digits to remove insignificant platform math-library noise, so supported CI environments regenerate the same versioned JSON artifact.
