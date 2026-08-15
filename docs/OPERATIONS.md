# Operations guide

## Environment setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

Regenerate the synthetic artifact and run the service:

```powershell
.\.venv\Scripts\python.exe scripts\train_demo_model.py
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LONGIEYE_MODEL_PATH` | `configs/demo_model.json` | Versioned JSON model artifact |
| `LONGIEYE_LOG_LEVEL` | `INFO` | Application log level |

Only a `demo_synthetic` artifact is accepted by the public scaffold. A research adapter must be introduced as a separate, reviewed implementation rather than silently replacing this file.

## Sprint 2 research-artifact policy

The public service remains pinned to the synthetic JSON backend. `ResearchModelAdapter` and its lazy optional internal `_TorchStateDictRuntime` are offline adapter foundations; the runtime has no public raw-checkpoint loader. Installing PyTorch, setting an artifact path or possessing a private checkpoint must not silently switch `/predict` into research mode.

Public CI may load only a deterministic synthetic state dict generated inside pytest's temporary directory. A real research export stays outside the repository, build context, CI cache, container image and release artifacts unless [the authorization checklist](RESEARCH_ARTIFACT_AUTHORIZATION.md) records a separate public-release decision for its exact contents.

SHA-256 detects byte changes. It is not a digital signature, source attestation, malware review, privacy approval or clinical validation. The package cannot self-approve: a package-external `ApprovalPolicy` receipt must bind the requested scope, source commit, checkpoint, preprocessing, model card and golden cases. Code verifies that the v1 JSON receipt is outside the package and that its contents match; it does not inspect ACLs or authenticate an institutional signer. The deployment owner must provide those external controls.

Inference rechecks the loaded receipt's expiry. `readiness()` additionally rereads the external policy receipt to detect revocation or replacement; a failed instance remains not-ready and must be reconstructed after a newly approved receipt is issued.

Validate the public, non-loadable manifest template and tracked-file policy:

```powershell
.\.venv\Scripts\python.exe scripts\validate_research_manifest.py
.\.venv\Scripts\python.exe scripts\check_public_artifacts.py
```

The tracked-file policy is a path/extension/size guardrail, not content-aware DLP or proof that all Git history is safe. Release review must still inspect history, secrets and the final build context separately.

The separate CI adapter job installs the pinned CPU-only PyTorch 2.13.0 test runtime from `requirements.research.lock`. It generates its synthetic state dict under pytest's temporary directory; no `.pt` file is committed. The default local environment reports `74 passed, 5 skipped`; the CI job sets `LONGIEYE_REQUIRE_TORCH=1`, so a missing import fails instead of skipping.

After a real package receives explicit local-use authorization, an engineering-only report can be created in the Git-ignored `artifacts/private/comparison` directory. The receipt must be stored in a separately controlled location, not inside the three-file research package:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.research.lock
.\.venv\Scripts\python.exe scripts\compare_adapters.py `
  <authorized-package-directory> `
  --approval-receipt <external-controlled-receipt.json>
```

The package must contain exactly `manifest.json`, its named model card and one checkpoint; symlinks, extra files and oversized files are rejected. The standard comparison CLI invokes the generic builder only after this package gate succeeds. The builder itself accepts protocol backends and therefore records, but does not independently authenticate, authorization metadata; the generated report states that limitation. The command measures only already-loaded sequential P50/P95/P99 and excludes both synthetic sanity metrics and research quality metrics. It does not yet measure cold load, RSS or artifact size. A local report must not be committed or shown publicly unless its exact contents receive separate release approval.

`torch.load(..., weights_only=True)` narrows the accepted object surface but is not a sandbox and does not eliminate denial-of-service or memory-exhaustion risk. Load only a trusted, explicitly approved checkpoint in an isolated environment; the one-megabyte package limit is an additional bound, not a trust substitute.

## Request tracing

Every HTTP response includes `X-Request-ID`. A caller may provide a value containing at most 64 letters, digits, dots, underscores or hyphens. Unsafe values are replaced with a UUID.

The JSON log contains method, path, status, duration, model ID and request ID. Request bodies, clinical values and `case_id` are intentionally not logged.

```json
{"level":"INFO","message":"HTTP 请求处理完成","request_id":"demo-001","event":"http_request_completed","http_method":"POST","http_path":"/predict","status_code":200,"duration_ms":1.234}
```

## Error contract

```json
{
  "request_id": "trace-id",
  "error": {
    "code": "request_validation_error",
    "message": "请求参数校验失败。",
    "details": [
      {
        "location": ["body", "y1", "height_cm"],
        "message": "该字段必须是数字。",
        "type": "float_parsing"
      }
    ]
  }
}
```

Validation details never include the submitted value. Unexpected failures return a generic message while the structured log records only the exception type.

## Performance benchmark

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_service.py
```

Outputs are written to `benchmarks/latest.json` and `benchmarks/latest.md`. The timing pass records warmup counts, P50/P95/P99 latency and sequential throughput with memory tracing disabled. A separate pass records process RSS and Python `tracemalloc` peaks. It is an in-process baseline, not a production load test.

## Health check

`GET /health` exposes only service version, model ID, model stage and `clinical_use=false`. It does not check an external database because Sprint 1 has no external dependencies.
