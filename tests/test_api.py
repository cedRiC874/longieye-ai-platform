import asyncio
import json
from pathlib import Path

import httpx

from app.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def async_request(method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        return await client.request(method, path, **kwargs)


def api_request(method: str, path: str, **kwargs):
    return asyncio.run(async_request(method, path, **kwargs))


def request_payload():
    return json.loads((PROJECT_ROOT / "examples" / "request.json").read_text())


def test_health_exposes_demo_safety_state():
    response = api_request("GET", "/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_stage": "demo_synthetic",
        "model_id": "longieye-synthetic-static-sex-delta8-v0",
        "api_version": "0.2.0",
        "clinical_use": False,
    }
    assert response.headers["X-Request-ID"]

    ready_response = api_request("GET", "/ready")
    assert ready_response.status_code == 200
    assert ready_response.json() == response.json()


def test_predict_returns_two_eye_result_and_no_clinical_claim():
    response = api_request("POST", "/predict", json=request_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["case_id"] == "demo-001"
    assert body["clinical_use"] is False
    assert body["predictions"] == {
        "od": {"demo_probability": 0.165514},
        "os": {"demo_probability": 0.12086},
    }
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_predict_rejects_inconsistent_static_sex():
    payload = request_payload()
    payload["y2"]["sex_code"] = 1
    response = api_request("POST", "/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "domain_validation_error"
    assert "sex_code must remain static" in body["error"]["message"]
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_request_id_is_preserved_only_when_safe():
    response = api_request(
        "POST",
        "/predict",
        json=request_payload(),
        headers={"X-Request-ID": "portfolio-demo-001"},
    )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "portfolio-demo-001"
    assert response.json()["request_id"] == "portfolio-demo-001"


def test_validation_error_is_structured_and_does_not_echo_input():
    payload = request_payload()
    payload["y1"]["height_cm"] = "do-not-echo-this-value"
    response = api_request("POST", "/predict", json=payload)
    body = response.json()
    assert response.status_code == 422
    assert body["error"]["code"] == "request_validation_error"
    assert body["error"]["details"][0]["location"] == ["body", "y1", "height_cm"]
    assert "do-not-echo-this-value" not in response.text


def test_extra_identity_fields_are_rejected_instead_of_ignored():
    payload = request_payload()
    payload["patient_id"] = "private-identifier"
    response = api_request("POST", "/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "request_validation_error"
    assert body["error"]["details"][0]["type"] == "extra_forbidden"
    assert "private-identifier" not in response.text


def test_case_alias_and_followup_contract_are_strict():
    payload = request_payload()
    payload["case_id"] = "a name with spaces"
    response = api_request("POST", "/predict", json=payload)
    assert response.status_code == 422


def test_openapi_publishes_success_and_error_contracts():
    schema = api_request("GET", "/openapi.json").json()
    predict_operation = schema["paths"]["/predict"]["post"]
    assert "PredictionResponse" in json.dumps(predict_operation["responses"]["200"])
    assert "ErrorResponse" in json.dumps(predict_operation["responses"]["422"])
    assert "ErrorResponse" in json.dumps(predict_operation["responses"]["500"])

    payload = request_payload()
    payload["followup_months"] = 6
    response = api_request("POST", "/predict", json=payload)
    assert response.status_code == 422
