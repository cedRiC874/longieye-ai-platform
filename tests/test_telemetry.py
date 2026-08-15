import json
import logging

from longieye.telemetry import (
    JsonFormatter,
    normalize_request_id,
    reset_request_id,
    set_request_id,
)


def test_request_id_rejects_log_injection_characters():
    generated = normalize_request_id("unsafe\nforged-log")
    assert generated != "unsafe\nforged-log"
    assert "\n" not in generated


def test_json_formatter_adds_trace_context_without_custom_payloads():
    token = set_request_id("trace-123")
    try:
        record = logging.LogRecord(
            "longieye",
            logging.INFO,
            __file__,
            1,
            "request finished",
            (),
            None,
        )
        record.event = "http_request_completed"
        record.status_code = 200
        payload = json.loads(JsonFormatter().format(record))
    finally:
        reset_request_id(token)

    assert payload["request_id"] == "trace-123"
    assert payload["event"] == "http_request_completed"
    assert payload["status_code"] == 200
