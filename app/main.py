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
    title="LongiEye AI Platform｜纵向近视建模工程演示",
    version=__version__,
    description=(
        "隐私优先的纵向近视建模工程演示。所有结果仅来自合成数据，"
        "不可用于诊断、筛查或治疗决策。"
    ),
    openapi_tags=[
        {"name": "服务状态", "description": "检查服务进程与演示模型是否可用。"},
        {
            "name": "演示推理",
            "description": "提交两次随访数据，生成不可用于临床的双眼合成演示结果。",
        },
    ],
)

VALIDATION_MESSAGES = {
    "missing": "缺少必填字段。",
    "extra_forbidden": "不允许提交未定义字段。",
    "int_type": "该字段必须是整数。",
    "float_type": "该字段必须是数字。",
    "string_type": "该字段必须是文本。",
    "greater_than_equal": "数值低于允许范围。",
    "less_than_equal": "数值高于允许范围。",
    "string_too_short": "文本长度不足。",
    "string_too_long": "文本长度超过限制。",
    "string_pattern_mismatch": "文本格式不符合要求。",
}


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
                "未处理的请求异常",
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
                message="服务发生未预期错误。",
            )
        duration_ms = (time.perf_counter() - started) * 1000.0
        response.headers["X-Request-ID"] = request_id
        LOGGER.info(
            "HTTP 请求处理完成",
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
    details = []
    for error in exc.errors():
        error_type = str(error.get("type", "validation_error"))
        details.append(
            {
                "location": [str(item) for item in error.get("loc", ())],
                "message": VALIDATION_MESSAGES.get(error_type, "字段值无效。"),
                "type": error_type,
            }
        )
    return error_response(
        request,
        status_code=422,
        code="request_validation_error",
        message="请求参数校验失败。",
        details=details,
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    code = "domain_validation_error" if exc.status_code == 422 else "http_error"
    message = (
        str(exc.detail)
        if exc.status_code == 422 and isinstance(exc.detail, str)
        else "请求处理失败。"
    )
    return error_response(
        request,
        status_code=exc.status_code,
        code=code,
        message=message,
    )


class VisitRequest(StrictRequest):
    sex_code: int = Field(
        ge=0,
        le=1,
        title="性别编码",
        description="静态二元编码；Y1 与 Y2 必须保持一致。",
    )
    height_cm: float = Field(
        ge=80, le=220, title="身高", description="身高，单位为厘米（cm）。"
    )
    weight_kg: float = Field(
        ge=15, le=250, title="体重", description="体重，单位为千克（kg）。"
    )
    sbp_mmhg: float = Field(
        ge=60,
        le=240,
        title="收缩压",
        description="收缩压，单位为毫米汞柱（mmHg）。",
    )
    dbp_mmhg: float = Field(
        ge=30,
        le=160,
        title="舒张压",
        description="舒张压，单位为毫米汞柱（mmHg）。",
    )
    waist_cm: float = Field(
        ge=30, le=200, title="腰围", description="腰围，单位为厘米（cm）。"
    )
    wears_glasses: int = Field(
        ge=0,
        le=1,
        title="是否佩戴眼镜",
        description="二元编码：0 表示否，1 表示是。",
    )
    axial_length_od_mm: float = Field(
        ge=15,
        le=35,
        title="右眼眼轴长度",
        description="右眼（OD）眼轴长度，单位为毫米（mm）。",
    )
    axial_length_os_mm: float = Field(
        ge=15,
        le=35,
        title="左眼眼轴长度",
        description="左眼（OS）眼轴长度，单位为毫米（mm）。",
    )

    def to_domain(self) -> VisitMeasurements:
        return VisitMeasurements(**self.model_dump())


class PredictionRequest(StrictRequest):
    case_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9._-]+$",
        title="案例代号",
        description="调用方提供的非识别性别名；不得填写姓名、证件号或其他个人信息。",
    )
    followup_months: int = Field(
        default=12,
        ge=12,
        le=12,
        title="随访间隔（月）",
        description="当前演示特征契约固定为12个月。",
    )
    y1: VisitRequest = Field(
        title="第一次随访（Y1）", description="基线时间点的脱敏测量值。"
    )
    y2: VisitRequest = Field(
        title="第二次随访（Y2）", description="12个月后时间点的脱敏测量值。"
    )


class HealthResponse(BaseModel):
    status: str = Field(description="服务状态。")
    model_stage: str = Field(description="模型所处阶段；演示环境固定为 demo_synthetic。")
    model_id: str = Field(description="当前加载的模型标识。")
    api_version: str = Field(description="API 版本。")
    clinical_use: bool = Field(description="是否允许临床使用；演示环境始终为 false。")


class ModelResponse(BaseModel):
    model_id: str = Field(description="模型标识。")
    model_stage: str = Field(description="模型阶段。")
    training_data: str = Field(description="训练数据来源说明。")


class EyePredictionResponse(BaseModel):
    demo_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="合成模型演示分数，不是经过临床校准的风险概率。",
    )


class PredictionResponse(BaseModel):
    request_id: str = Field(description="贯穿响应头、响应体和日志的请求追踪 ID。")
    case_id: str | None = Field(description="调用方提供的非识别性案例代号。")
    model: ModelResponse = Field(description="本次推理使用的演示模型信息。")
    predictions: dict[str, EyePredictionResponse] = Field(
        description="右眼（od）和左眼（os）的合成演示分数。"
    )
    derived_features: dict[str, float] = Field(
        description="由两次随访计算得到的静态值与纵向变化量。"
    )
    clinical_use: bool = Field(description="是否允许临床使用；始终为 false。")
    disclaimer: str = Field(description="非临床用途安全提示。")


class ErrorDetailResponse(BaseModel):
    location: list[str] = Field(description="错误字段在请求中的位置。")
    message: str = Field(description="不回显输入值的中文错误说明。")
    type: str = Field(description="稳定的框架校验错误类型。")


class ErrorBodyResponse(BaseModel):
    code: str = Field(description="供客户端判断的稳定错误码。")
    message: str = Field(description="面向用户的中文错误说明。")
    details: list[ErrorDetailResponse] | None = Field(
        default=None, description="可选的字段级错误列表。"
    )


class ErrorResponse(BaseModel):
    request_id: str = Field(description="请求追踪 ID。")
    error: ErrorBodyResponse = Field(description="结构化错误内容。")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["服务状态"],
    summary="查看服务存活状态",
    description="返回服务版本和已加载演示模型的安全状态。",
)
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_stage": SERVICE.model.metadata.get("model_stage"),
        "model_id": SERVICE.model.metadata.get("model_id"),
        "api_version": app.version,
        "clinical_use": False,
    }


@app.get(
    "/ready",
    response_model=HealthResponse,
    tags=["服务状态"],
    summary="查看模型就绪状态",
    description="确认经过校验的合成模型制品已成功加载。",
)
def ready() -> dict[str, object]:
    return health()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["演示推理"],
    summary="生成双眼合成演示结果",
    description=(
        "提交两个固定相隔12个月的脱敏随访时间点，返回 OD/OS 双眼合成演示分数。"
        "输出仅用于展示数据校验、特征契约和推理服务，不可用于临床。"
    ),
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
