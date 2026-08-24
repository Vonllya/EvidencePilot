from __future__ import annotations

import json
import logging
import re
from typing import Any

SENSITIVE_NAME = re.compile(r"(api[_-]?key|authorization|token|secret|reasoning_content)", re.I)


def _safe(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: "[REDACTED]" if SENSITIVE_NAME.search(key) else value
        for key, value in fields.items()
    }


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"level": record.levelname, "logger": record.name, "event": record.getMessage()}
        payload.update(_safe(getattr(record, "event_fields", {})))
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(event, extra={"event_fields": _safe(fields)})
