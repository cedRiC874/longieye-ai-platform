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

The Sprint 2 baseline reported `79 passed` with the optional PyTorch runtime. The current v0.5 suite reports `155 passed` with both optional PyTorch and public-cohort dependencies, including the unchanged research-adapter contract. GitHub Actions verifies the default Python 3.10/3.11/3.12 matrix, deterministic Windows artifacts, wheel isolation, the mandatory PyTorch adapter suite and the public-cohort validation suite. Phase A remains complete.

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

## Sprint 3 — synthetic multimodal extension (completed)

Sprint 3A adds a fully synthetic, offline-only image branch without changing the public `/predict` request or response contract:

- Two 128×128 OD/OS fundus-like fixtures are generated entirely by fixed integer drawing code and carry visible `SYNTHETIC` watermarks. No patient image or public medical dataset is used.
- A standard-library canonical PNG codec accepts only RGB8, exact `IHDR → IDAT → IEND` inventory, fixed dimensions, bounded decompression, filter 0, valid CRCs and no metadata.
- File and pixel SHA-256 registries bind each tracked fixture to its eye. Untrusted eye labels, duplicate images, swapped preprocessing and arbitrary raster artifacts fail closed.
- The engineering quality gate checks brightness, contrast, clipping, field coverage and sharpness before deterministic 32×32 area pooling.
- `ImageEncoder` and `DeterministicFundusEncoder` demonstrate the adapter boundary with five inspectable statistics; there is no learned visual representation and no training claim.
- `StructuredAnchoredFusionAdapter` adds a maximum `0.35` logit residual per eye. Missing or rejected images fall back bit-for-bit to the existing structured score; provenance conflict rejects the whole request.
- A current-request failure is isolated per eye, then locks the image component not-ready for later requests. Outputs expose branch mode and safe reason codes, never paths, hashes, pixels or embeddings.
- The generic public HTTP path still rejects image, path, URL and Base64 fields. The multimodal stage is not accepted by `RiskPredictionService`.
- Local aggregate benchmarks cover both images, one missing image and both missing images without persisting per-case outputs or model-quality metrics.

Sprint 3A proves image-contract engineering, graceful degradation and synthetic provenance controls. It does not prove that a CNN was trained, that multimodal prediction is better, or that any image path is clinically valid. Real-image work requires a new authorization and model contract rather than reinterpreting this synthetic fixture boundary.

## Sprint 4 — public longitudinal-outcome validation (completed)

Sprint 4 adds a real public-cohort evaluation track without silently changing the synthetic HTTP contract:

- The OLSM public subset is downloaded from a commit-pinned `aplore3 0.9` source and verified against a fixed SHA-256. Raw rows remain under ignored `build/` storage.
- The cohort contains 618 baseline non-myopic children and a five-year incident-myopia outcome (81 events). The distributed subset has stable participant IDs and no missing cells.
- Three ridge-logistic models defined in this analysis version use repeated nested 5×5 participant-level cross-fitting. Hyperparameters are selected only inside outer training folds.
- AUC, Brier, calibration intercept/slope and calibration bins are calculated after participant-level averaging of five repeated OOF predictions. The 2,000 participant bootstrap intervals are conditional on those predictions and do not refit the full pipeline.
- Exploratory DCA uses a code-fixed 0.03–0.30 threshold range and a hypothetical enhanced-monitoring action. It was not independently preregistered and is not evidence of clinical utility.
- Three paired AUC comparisons form one family and use Holm FWER control. None is significant after correction.
- A temporal entry-year split and five additional nested-CV seeds are reported as sensitivity checks.
- Git contains only aggregate JSON, deterministic SVG curves and the report. Participant rows, folds, OOF predictions and frozen parameters stay in ignored local storage.

The primary full-model AUC is 0.872 (95% conditional bootstrap CI 0.830–0.911), Brier score 0.080 (0.066–0.096), calibration intercept -0.009 and slope 0.934. With only 81 events and no external cohort, this is explicitly pilot internal validation.

This track does not reproduce the private thesis model: the public subset has baseline right-eye predictors and a longitudinal outcome, not repeated Y1/Y2 measurements or the nine-feature delta contract. A future service endpoint would require a separately versioned API and intended-use review.
