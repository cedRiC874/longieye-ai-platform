# Portfolio roadmap

## Sprint 0 — completed scaffold

- Privacy-safe domain contract.
- Synthetic dual-eye model training.
- CLI and FastAPI inference paths.
- Unit tests, Docker and CI configuration.

## Sprint 1 — reproducible service (completed)

- API integration tests and stable, non-echoing error responses.
- End-to-end request IDs and privacy-safe JSON logs.
- Pinned runtime/test lockfile and local virtual environment.
- P50/P95/P99, sequential throughput and Python peak-memory report.
- Model card, operations guide, architecture SVG and one-minute recording script.

The actual screen recording remains a manual portfolio step so the owner can narrate the work and verify that no private desktop content is visible.

## Sprint 2 — research model adapter

- Define an authorization checklist for using research artifacts.
- Export preprocessing statistics and a model card.
- Implement a PyTorch adapter without changing the public API.
- Compare synthetic and research modes without mixing their metrics.

## Sprint 3 — multimodal extension

- Add a public or fully synthetic fundus image example.
- Implement image preprocessing and quality checks.
- Add an image encoder interface and clinical-anchored fusion adapter.
- Benchmark fallback behavior when the image branch is unavailable.
