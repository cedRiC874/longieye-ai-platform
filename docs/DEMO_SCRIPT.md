# One-minute portfolio demo script

## Before recording

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` and keep `benchmarks/latest.md` ready in another tab.

## 60-second narration

**0–10 seconds:** “LongiEye turns my longitudinal myopia research into a privacy-safe engineering demo. The public service contains no participant data or research checkpoint.”

**10–25 seconds:** Show `/predict`. “The request contains two visits. The domain layer validates physiological ranges and derives static sex plus eight changes; spherical equivalent and identifiers are excluded.”

**25–38 seconds:** Execute the example. “The response carries one trace ID across the header, JSON body and structured logs. Every result is labelled synthetic and not for clinical use.”

**38–50 seconds:** Show the error example. “Invalid input returns a stable error code without echoing the submitted value, which reduces accidental sensitive-data exposure.”

**50–60 seconds:** Show the benchmark. “The repository includes repeatable P50/P95/P99 and memory measurements, tests, CI and a model card. The next adapter can replace the synthetic backend without changing the API.”

Do not present the synthetic AUC or returned scores as medical evidence. Record the demo only after confirming that the screen contains no private repository paths or personal notifications.
