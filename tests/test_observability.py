import json
import logging

from evidencepilot.observability import JsonFormatter


def test_json_formatter_redacts_sensitive_field_names():
    record = logging.LogRecord("test", logging.INFO, "", 0, "provider_call", (), None)
    record.event_fields = {"task_id": "T1", "api_key": "must-not-leak", "token": "x"}
    payload = json.loads(JsonFormatter().format(record))
    assert payload["task_id"] == "T1"
    assert payload["api_key"] == "[REDACTED]"
    assert payload["token"] == "[REDACTED]"
    assert "must-not-leak" not in json.dumps(payload)
