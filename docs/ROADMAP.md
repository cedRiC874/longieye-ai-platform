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
- P50/P95/P99, sequential throughput, process RSS and Python peak-memory report.
- Model card, operations guide, architecture SVG and one-minute recording script.

The actual screen recording remains a manual portfolio step so the owner can narrate the work and verify that no private desktop content is visible.

## Sprint 2 — authorization-gated research adapter

Sprint 2 keeps the public `/predict` path on the deterministic synthetic JSON model. It builds an auditable adapter boundary first; it does not publish or activate a private research checkpoint by default.

### Phase A — public adapter foundation (completed)

Phase A must remain reproducible without any private research artifact:

- `RiskModelBackend` removes the service's concrete-model type dependency, while the current demo service separately rejects every non-synthetic stage to preserve response semantics.
- `ResearchModelAdapter` validates manifest metadata, feature/output contracts, preprocessing hash, exact package inventory, bounded regular files, model-card completion and golden vectors.
- A package-external `ApprovalPolicy` receipt binds the requested scope, source commit and all artifact/card/preprocessing/self-test hashes; manifest self-approval is not accepted.
- Internal `_TorchStateDictRuntime` is reachable only after package verification. It lazily imports locked CPU-only PyTorch, uses `weights_only=True`, checks allowlisted keys/shape/dtype/finite values, disables gradients, verifies parameter integrity before readiness/inference and runs under `inference_mode()`.
- Public tests generate a deterministic synthetic state dict only inside a temporary directory; no `.pt` is tracked or copied into Docker.
- Tests cover multi-vector synthetic parity, determinism and fail-closed checksum, receipt, inventory, schema/type, feature order, shape, dtype and NaN/Inf paths. Batch support is outside the single-case API contract.
- The generic comparison builder reports only already-loaded contract/runtime evidence, explicitly says that it does not verify authorization itself, and leaves both synthetic sanity and authorized research metric namespaces unavailable. The standard CLI calls it only after the package gate.
- The public API remains pinned to the synthetic JSON backend; no environment switch can silently enable research mode.

Local validation reports `79 passed` when the optional PyTorch runtime is installed. GitHub Actions also verifies the default Python 3.10/3.11/3.12 matrix, deterministic Windows artifact generation, wheel isolation and the mandatory PyTorch adapter suite. Phase A is complete.

Phase A proves adapter engineering. It does not prove that a real checkpoint was exported correctly and does not establish clinical performance.

### Phase B — authorized real-artifact evaluation

Phase B starts only after [the research-artifact authorization checklist](RESEARCH_ARTIFACT_AUTHORIZATION.md) produces a package-external approval receipt for the exact request hash:

- Pin the private source commit and map every exported file to an approved allowlist.
- Export only the approved model and preprocessing statistics into an isolated local/private location.
- Complete the [research model card](RESEARCH_MODEL_CARD_TEMPLATE.md) with cohort, target, validation design, evidence locations, limitations and distribution status.
- Run parity against the approved source runtime and bind the report to exact artifact and preprocessing SHA-256 values.
- Measure artifact size, cold load, first inference, warm P50/P95/P99 and RSS in a declared environment.
- Map every proposed research metric to its aggregate result file, calculation script and thesis table before using it in a resume.
- Review output semantics and API versioning separately before any future research endpoint or response contract is considered.

Local-use approval does not authorize a GitHub commit, container image, CI artifact, release upload, screenshot or public demonstration. SHA-256 is an integrity check, not a digital signature, source attestation or release authorization.

## Sprint 3 — multimodal extension

- Add a public or fully synthetic fundus image example.
- Implement image preprocessing and quality checks.
- Add an image encoder interface and clinical-anchored fusion adapter.
- Benchmark fallback behavior when the image branch is unavailable.
