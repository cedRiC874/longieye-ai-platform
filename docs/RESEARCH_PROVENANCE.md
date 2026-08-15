# Research provenance and privacy boundary

## Source

The feature contract was derived from the private repository
`cedRiC874/myopia-thesis-code`, especially:

- `clinical_representations_10x5/run_config.json`;
- `al_delta_ablation_10x5/run_config.json`;
- `longitudinal_incident_common.py`;
- the supplementary static-sex robustness design.

The source research used aligned Y1/Y2 clinical variables, excluded SE1/SE2 from the predictor set, and evaluated OD/OS targets independently. LongiEye keeps those conceptual boundaries but does not copy participant-level inputs, outputs or checkpoints.

## Auditable source snapshot

| Source file in private research repository | Git blob SHA |
| --- | --- |
| `README.md` | `57769ad88b03ad90f615db76e0a5d0620f399d06` |
| `clinical_representations_10x5/run_config.json` | `740de855ed32ea3a32d1bcadbe2ff0a5c2c8c131` |
| `al_delta_ablation_10x5/run_config.json` | `d27d936b522c18bdbdef4591442fd17a87c39a66` |
| `longitudinal_incident_common.py` | `4dba96f7136e8ee6e5715dfba71319f142d9a848` |
| `longitudinal_multimodal_models.py` | `58cb4a7e9406a13242bb141b470917e074f440da` |

These content-addressed blob IDs record exactly which files informed the public contract. Before publishing research metrics in a resume, also pin the private repository commit and map each number to its aggregate result file and thesis table.

## Explicit exclusions

- Raw spreadsheets and fundus images.
- Participant identifiers and split assignments.
- Image paths, hashes and OOF predictions.
- Research checkpoints, caches and credentials.
- Claims that the synthetic model reproduces thesis performance.

## Publication gate

Before publishing this repository, confirm institutional ownership and choose an appropriate software license. The absence of a license means others do not automatically receive reuse rights.
