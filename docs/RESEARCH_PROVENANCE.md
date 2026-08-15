# Research provenance and privacy boundary

## Source

The feature contract was derived from a private thesis workspace. Its remote
name, internal paths, commit and blob identifiers are intentionally not listed
in the public repository while research-artifact authorization remains pending.

The source research used aligned Y1/Y2 clinical variables, excluded SE1/SE2 from the predictor set, and evaluated OD/OS targets independently. LongiEye keeps those conceptual boundaries but does not copy participant-level inputs, outputs or checkpoints.

## Auditable source snapshot

The exact source snapshot belongs in an access-controlled evidence registry. A
Phase B review must bind the private source commit, approved allowlist, aggregate
result file, calculation script and thesis table to one authorization record.
Only a non-sensitive reference ID may be copied into a public model card after
release approval.

## Current authorization state

The current source metadata, checkpoint and preprocessing-statistics status is `NOT_AUTHORIZED`, and public research mode is disabled.

Any Phase B evaluation must first complete the [research-artifact authorization checklist](RESEARCH_ARTIFACT_AUTHORIZATION.md) and a card based on the [research model-card template](RESEARCH_MODEL_CARD_TEMPLATE.md). Approval for local evaluation does not permit a Git commit, CI upload, container layer, release, screenshot or public demonstration.

Artifact and preprocessing SHA-256 values are integrity identifiers only. They are not digital signatures, proof of ownership, privacy approval or permission to distribute.

## Explicit exclusions

- Raw spreadsheets and fundus images.
- Participant identifiers and split assignments.
- Image paths, hashes and OOF predictions.
- Research checkpoints, caches and credentials.
- Claims that the synthetic model reproduces thesis performance.

## Publication gate

Before adding any research-derived artifact, source identifier or metric, confirm institutional ownership and record the allowed release scope. This repository currently has no software license, so public visibility does not automatically grant reuse rights.
