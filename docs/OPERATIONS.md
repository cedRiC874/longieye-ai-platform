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
