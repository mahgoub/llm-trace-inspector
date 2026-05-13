from __future__ import annotations

import json
from pathlib import Path

from llm_trace_inspector.models import Trace


def load_trace(path: str | Path) -> Trace:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Trace.model_validate(data)


def write_json(path: str | Path, data: object) -> None:
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

