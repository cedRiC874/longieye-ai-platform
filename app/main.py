"""FastAPI entrypoint for the LongiEye engineering demo."""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from longieye import __version__
from longieye.domain import LongitudinalCase, VisitMeasurements
from longieye.model import DemoRiskModel
from longieye.service import RiskPredictionService
from longieye.telemetry import (
    configure_logging,
    normalize_request_id,
    reset_request_id,
    set_request_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv("LONGIEYE_MODEL_PATH", PROJECT_ROOT / "configs" / "demo_model.json")
)
SERVICE = RiskPredictionService(DemoRiskModel.from_path(MODEL_PATH))
LOGGER = configure_logging()

app = FastAPI(
    title="LongiEye AI Platform",
    version=__version__,
    description=(
        "Privacy-first longitudinal modeling demo. Synthetic engineering "
        "output only; not for clinical use."
    ),
)


class StrictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, str_strip_whitespace=True
    )


def error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: list[dict[str, object]] | None = None,
) -> JSONResponse:
    error_payload: dict[str, object] = {"code": code, "message": message}
    payload: dict[str, object] = {
        "request_id": request.state.request_id,
        "error": error_payload,
    }
    if details:
        error_payload["details"] = details
    return JSONResponse(status_code=status_code, content=payload)


@app.middleware("http")
async def request_telemetry(request: Request, call_next):
    request_id = normalize_request_id(request.headers.get("X-Request-ID"))
    request.state.request_id = request_id
    token = set_request_id(request_id)
    started = time.perf_counter()
    try:
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception(
                "Unhandled request failure",
                extra={
                    "event": "http_request_failed",
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "error_code": "internal_server_error",
                },
            )
            response = error_response(
                request,
                status_code=500,
                code="internal_server_error",
                message="An unexpected error occurred.",
            )
        duration_ms = (time.perf_counter() - started) * 1000.0
        response.headers["X-Request-ID"] = request_id
        LOGGER.info(
            "HTTP request completed",
            extra={
                "event": "http_request_completed",
                "http_method": request.method,
                "http_path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 3),
                "model_id": SERVICE.model.metadata.get("model_id"),
            },
        )
        return response
    finally:
        reset_request_id(token)


@app.exception_handler(RequestValidationError)
async def request_validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    details = [
        {
            "location": [str(item) for item in error.get("loc", ())],
            "message": error.get("msg", "Invalid value"),
            "type": error.get("type", "validation_error"),
        }
        for error in exc.errors()
    ]
    return error_response(
        request,
        status_code=422,
        code="request_validation_error",
        message="Request validation failed.",
        details=details,
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    code = "domain_validation_error" if exc.status_code == 422 else "http_error"
    message = str(exc.detail) if isinstance(exc.detail, str) else "Request failed."
    return error_response(
        request,
        status_code=exc.status_code,
        code=code,
        message=message,
    )


class VisitRequest(StrictRequest):
    sex_code: int = Field(ge=0, le=1)
    height_cm: float = Field(ge=80, le=220)
    weight_kg: float = Field(ge=15, le=250)
    sbp_mmhg: float = Field(ge=60, le=240)
    dbp_mmhg: float = Field(ge=30, le=160)
    waist_cm: float = Field(ge=30, le=200)
    wears_glasses: int = Field(ge=0, le=1)
    axial_length_od_mm: float = Field(ge=15, le=35)
    axial_length_os_mm: float = Field(ge=15, le=35)

    def to_domain(self) -> VisitMeasurements:
        return VisitMeasurements(**self.model_dump())


class PredictionRequest(StrictRequest):
    case_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9._-]+$",
        description="Non-identifying caller alias; never place personal data here.",
    )
    followup_months: int = Field(default=12, ge=12, le=12)
    y1: VisitRequest
    y2: VisitRequest


class HealthResponse(BaseModel):
    status: str
    model_stage: str
    model_id: str
    api_version: str
    clinical_use: bool


class ModelResponse(BaseModel):
    model_id: str
    model_stage: str
    training_data: str


class EyePredictionResponse(BaseModel):
    demo_probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    request_id: str
    case_id: str | None
    model: ModelResponse
    predictions: dict[str, EyePredictionResponse]
    derived_features: dict[str, float]
    clinical_use: bool
    disclaimer: str


class ErrorDetailResponse(BaseModel):
    location: list[str]
    message: str
    type: str


class ErrorBodyResponse(BaseModel):
    code: str
    message: str
    details: list[ErrorDetailResponse] | None = None


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorBodyResponse


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_stage": SERVICE.model.metadata.get("model_stage"),
        "model_id": SERVICE.model.metadata.get("model_id"),
        "api_version": app.version,
        "clinical_use": False,
    }


@app.get("/ready", response_model=HealthResponse)
def ready() -> dict[str, object]:
    return health()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def predict(payload: PredictionRequest, request: Request) -> dict[str, object]:
    try:
        case = LongitudinalCase(
            y1=payload.y1.to_domain(),
            y2=payload.y2.to_domain(),
            followup_months=payload.followup_months,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SERVICE.predict(
        case,
        case_id=payload.case_id,
        request_id=request.state.request_id,
    )
