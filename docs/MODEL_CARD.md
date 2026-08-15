# Model card: LongiEye synthetic static-sex + delta8 demo

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

The private thesis repository reported a locked longitudinal clinical experiment on 1,547 participants with 140/130 OD/OS events. Its delta-only clinical macro AUC was `0.8824 +/- 0.0105`. That result belongs to the research pipeline and is not produced by this service. No research checkpoint, patient-level record or OOF prediction is packaged here.

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
- Python allocation memory is measured, but operating-system RSS and container behavior are not yet measured.
- The local benchmark excludes network, reverse-proxy and concurrent-load overhead.

## Reproducibility

```powershell
python scripts/train_demo_model.py
python -m pytest -q
python scripts/benchmark_service.py
```

The default synthetic seed is `20260815`; regenerated coefficients and metrics should be deterministic on the same Python implementation.
