from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from .types import MoveLogEntry


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    try:
        return float(value)
    except Exception:
        return str(value)


class RefinementLogger:
    def __init__(
        self,
        output_dir: Optional[str],
        log_name: str,
        diagnostics_name: str = "smart_ligand_refine_diagnostics.json",
    ):
        self.output_dir = output_dir or os.getcwd()
        self.log_name = log_name
        self.diagnostics_name = diagnostics_name
        self.entries: List[MoveLogEntry] = []
        self.diagnostic_events: List[Dict[str, Any]] = []
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def path(self) -> str:
        return os.path.join(self.output_dir, self.log_name)

    @property
    def diagnostics_path(self) -> str:
        return os.path.join(self.output_dir, self.diagnostics_name)

    def log(self, entry: MoveLogEntry) -> None:
        self.entries.append(entry)

    def diagnostic(self, event_type: str, **payload: Any) -> None:
        event = {"type": str(event_type)}
        event.update(payload)
        self.diagnostic_events.append(event)

    def write(self) -> str:
        rows = [_jsonable(asdict(entry)) for entry in self.entries]
        with open(self.path, "w") as handle:
            json.dump(rows, handle, indent=2)
        return self.path

    def write_diagnostics(self, summary: Optional[Dict[str, Any]] = None) -> str:
        payload = {
            "summary": _jsonable(summary or {}),
            "events": _jsonable(self.diagnostic_events),
        }
        with open(self.diagnostics_path, "w") as handle:
            json.dump(payload, handle, indent=2)
        return self.diagnostics_path
