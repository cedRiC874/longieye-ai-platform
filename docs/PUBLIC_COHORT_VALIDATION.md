# Public longitudinal cohort validation

> Status: real public measurements and a five-year longitudinal outcome; pilot internal validation only.
> Scope first: this experiment validates the evaluation pipeline, not LongiEye's Y1-to-Y2 delta-feature model. OLSM contains baseline right-eye predictors and a future outcome, so the task is baseline prognosis rather than longitudinal trajectory modeling.

## Cohort and provenance

- Dataset: Orinda Longitudinal Study of Myopia public subset.
- Source: `aplore3 0.9`, pinned commit `f842768f2f1333d2d9cb468b0263655cea018bc5`.
- Raw source SHA-256: `4147ac651d8f1a975be7eefb39a49044e9a0e5a8f97ddfb40bc764f70680784d`.
- Rights status: the `aplore3` package is GPL-3; underlying cohort-data reuse rights were not independently re-licensed by this project, so only aggregate results are prepared for review.
- Participants: 618; incident-myopia events: 81; non-events: 537.
- Missing cells in the distributed subset: 0.
- Public package documentation: https://search.r-project.org/CRAN/refmans/aplore3/html/myopia.html
- Primary study context: https://pmc.ncbi.nlm.nih.gov/articles/PMC2871403/
- Raw participant rows and out-of-fold predictions are not committed.

The OLSM subset contains baseline right-eye measurements and whether myopia developed during the first five years. It does not contain repeated Y1/Y2 predictors, so this analysis does not reproduce the original nine-feature LongiEye contract.

## Versioned validation design

- Repeated nested stratified cross-fitting: 5 repeats, each with 5 outer folds and 5 inner folds.
- Primary estimand: pooled participant-level AUC after averaging five repeated OOF probabilities; each fold uses its own fitted pipeline.
- Hyperparameter target: log_loss; ridge grid: `[0.01, 0.1, 1.0, 10.0, 100.0]`.
- Uncertainty: 2000 participant bootstrap repetitions conditional on the repeated OOF predictions; models are not refit inside each bootstrap sample.
- Primary model: full ridge logistic regression.
- Comparisons: three contrasts defined in this analysis version, 10000 participant-level randomization permutations, then Holm family-wise error control at 0.05.
- DCA action: hypothetical enhanced monitoring only; thresholds 0.03-0.30 are fixed in this analysis version, not independently preregistered.

## Internal-validation results

The event rate is 0.131; a constant prevalence prediction has Brier 0.114 and defines BSS = 0.

| Model | Features | AUC (95% CI) | Brier (95% CI) | BSS (95% CI) | Calibration intercept | Calibration slope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `refraction_only` | 1 | 0.859 (0.820-0.900) | 0.083 (0.069-0.099) | 0.271 (0.177-0.359) | -0.004 | 1.002 |
| `ocular` | 7 | 0.864 (0.825-0.902) | 0.083 (0.068-0.099) | 0.272 (0.176-0.360) | -0.009 | 0.992 |
| `full` | 14 | 0.872 (0.830-0.911) | 0.080 (0.066-0.096) | 0.295 (0.194-0.388) | -0.009 | 0.934 |

Primary calibration-intercept bootstrap CI: -0.306 to 0.273. Primary calibration-slope bootstrap CI: 0.764 to 1.170.

## Main substantive finding

The honest result is that baseline spherical-equivalent refraction alone carries almost all detectable predictive signal. The 14-feature full model improves pooled AUC by only 0.013 over the one-feature model, its confidence interval crosses zero, and the Holm-adjusted comparison is not significant. In a screening context this suggests that a cheap single measurement may allocate follow-up resources nearly as well as a more complex examination bundle; it is a resource-allocation hypothesis, not a clinical recommendation.

![Calibration curve](assets/public_cohort_calibration.svg)

## Decision curve analysis

The curve is exploratory because no clinician-derived utility study defines the relative cost of extra monitoring versus a missed future myopia case.

![Exploratory DCA](assets/public_cohort_dca.svg)

## Multiple-comparison results

| Comparison | Delta AUC (95% CI) | Raw p | Holm-adjusted p | Reject at FWER 0.05 |
| --- | ---: | ---: | ---: | --- |
| `ocular_vs_refraction_only` | 0.005 (-0.010-0.020) | 0.5007 | 1.0000 | No |
| `full_vs_refraction_only` | 0.013 (-0.014-0.038) | 0.3479 | 1.0000 | No |
| `full_vs_ocular` | 0.008 (-0.017-0.030) | 0.5338 | 1.0000 | No |

## Sensitivity checks

- Entry-cohort chronological split sensitivity: train entries from 1990-1992 (316 participants, 36 events); test entries from 1993-1995 (302 participants, 45 events). Follow-up windows overlap in calendar time, so this is not prospective temporal validation.
- Full-model temporal AUC: 0.858; Brier: 0.094.
- The 5 nested-CV repeats gave full-model AUC range 0.862-0.884 (median 0.867); their participant-level OOF probabilities were averaged before primary evaluation.

## Interpretation boundary

This is a real public longitudinal-outcome experiment, but it remains pilot internal validation. There are only 81 events for 14 full-model features, a nominal EPV of about 5.8 versus the conventional 10-EPV heuristic. Ridge regularization and nested validation mitigate, but do not eliminate, overfitting uncertainty. There is no external cohort and no repeated Y1/Y2 predictor measurement in the public subset. Bootstrap intervals are conditional on the repeated cross-fitted predictions and do not include full pipeline-refit uncertainty. The results validate the evaluation pipeline, not the LongiEye delta-feature model, synthetic API, private thesis model, clinical utility, or deployment readiness.

The safe portfolio claim is:

> Built a reproducible public-cohort validation pipeline with repeated nested participant-level cross-fitting, conditional bootstrap uncertainty, calibration, exploratory DCA, and Holm-adjusted paired model comparisons for five-year incident myopia.
