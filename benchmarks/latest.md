# LongiEye local benchmark

Generated: `2026-08-15T10:36:43.135565+00:00`
Model: `longieye-synthetic-static-sex-delta8-v0`
Artifact SHA-256: `2b6c7fac04adf08281ef07fa5ddfeef83722c768622e6cf953d0bd916b55a97e`
Python: `3.12.13`
Platform: `Windows-11-10.0.26200-SP0`

| Mode | Iterations | P50 ms | P95 ms | P99 ms | Requests/s | RSS delta MB | Python peak MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| core_service | 5000 | 0.008 | 0.009 | 0.013 | 114052.4 | 0.008 | 0.001 |
| in_process_asgi | 500 | 1.863 | 2.631 | 3.078 | 513.8 | 0.805 | 0.394 |

`core_service` measures feature extraction, inference and response assembly.
`in_process_asgi` additionally measures validation, middleware and JSON handling.
Both modes run in one process and exclude network, proxy and container overhead.
Timing runs with memory tracing disabled; memory is measured in a separate loop.
RSS is sampled from the process and Python peak uses `tracemalloc`.
These are local engineering measurements for a synthetic model, not clinical metrics.
