# LongiEye local benchmark

Generated: `2026-08-15T11:03:51.863524+00:00`
Model: `longieye-synthetic-static-sex-delta8-v0`
Artifact SHA-256: `a00a54be7d973b9834b5405e6bdaed512e9ae48186296f2af07793cab6ab3e32`
Python: `3.12.13`
Platform: `Windows-11-10.0.26200-SP0`

| Mode | Iterations | P50 ms | P95 ms | P99 ms | Requests/s | RSS delta MB | Python peak MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| core_service | 5000 | 0.008 | 0.009 | 0.012 | 118028.4 | 0.008 | 0.001 |
| in_process_asgi | 500 | 1.921 | 2.573 | 3.107 | 502.2 | 1.020 | 0.393 |

`core_service` measures feature extraction, inference and response assembly.
`in_process_asgi` additionally measures validation, middleware and JSON handling.
Both modes run in one process and exclude network, proxy and container overhead.
Timing runs with memory tracing disabled; memory is measured in a separate loop.
RSS is sampled from the process and Python peak uses `tracemalloc`.
These are local engineering measurements for a synthetic model, not clinical metrics.
