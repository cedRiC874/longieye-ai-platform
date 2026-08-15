FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LONGIEYE_MODEL_PATH=/app/configs/demo_model.json

WORKDIR /app

COPY pyproject.toml README.md requirements.runtime.lock ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.runtime.lock \
    && pip install --no-cache-dir --no-deps . \
    && groupadd --system longieye \
    && useradd --system --gid longieye --home-dir /app longieye

COPY app ./app
COPY configs ./configs

RUN chown -R longieye:longieye /app
USER longieye

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
