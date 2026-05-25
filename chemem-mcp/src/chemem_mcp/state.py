from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .paths import safe_child
from .schemas import (
    SCHEMA_VERSION,
    AcceptedIon,
    AcceptedLigand,
    BindingSiteRecord,
    ControllerDecision,
    FinalStatus,
    IonCandidate,
    LigandLibraryEntry,
    PoseRecord,
    RefinementOutcome,
    SegmentEvidence,
    Stage,
)

STATE_FILENAME = "state.json"
DECISION_LOG_FILENAME = "decision_log.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _tweak_key(tool_name: str, scope: str) -> str:
    if "::" in tool_name or "::" in scope:
        raise ValueError("tool_name and scope must not contain '::'")
    return f"{tool_name}::{scope}"


class CaseState(BaseModel):
    """Persistent per-case state.

    Lives at ``<run_dir>/state.json``. The companion ``decision_log.jsonl``
    is append-only and never rewritten.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    run_id: str
    run_dir: str
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: str = Field(default_factory=_utc_now_iso)
    stage: Stage = "preparation"
    final_status: Optional[FinalStatus] = None

    binding_sites: dict[str, BindingSiteRecord] = Field(default_factory=dict)
    ligand_library: dict[str, LigandLibraryEntry] = Field(default_factory=dict)
    segments: dict[str, SegmentEvidence] = Field(default_factory=dict)
    poses: dict[str, PoseRecord] = Field(default_factory=dict)
    ion_candidates: dict[str, IonCandidate] = Field(default_factory=dict)
    refinements: dict[str, RefinementOutcome] = Field(default_factory=dict)
    accepted_ligands: list[AcceptedLigand] = Field(default_factory=list)
    accepted_ions: list[AcceptedIon] = Field(default_factory=list)

    # Tier-3 escalation tracking: keyed by f"{tool}::{scope}".
    tweak_depth: dict[str, int] = Field(default_factory=dict)

    controller_step_count: int = 0
    tool_call_count: int = 0

    # Most recent tool dispatch signature: f"{tool_name}::{canonical_json(args)}".
    # Used by the agent to refuse identical consecutive tool calls.
    last_tool_signature: Optional[str] = None

    # Compact summary of the most recent successful tool call, surfaced back
    # to the model when it tries to repeat the same call. Helps the model
    # see "you already ran this; here's what you got" instead of just being
    # told "rejected".
    last_tool_result_summary: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load_or_create(cls, run_id: str, run_dir: Path) -> "CaseState":
        run_dir = Path(run_dir).resolve()
        state_path = safe_child(run_dir, STATE_FILENAME)
        if state_path.exists():
            data = json.loads(state_path.read_text())
            return cls.model_validate(data)
        state = cls(run_id=run_id, run_dir=str(run_dir))
        state.save()
        return state

    @property
    def state_path(self) -> Path:
        return safe_child(Path(self.run_dir), STATE_FILENAME)

    @property
    def decision_log_path(self) -> Path:
        return safe_child(Path(self.run_dir), DECISION_LOG_FILENAME)

    def save(self) -> Path:
        """Atomically write state.json."""
        self.updated_at = _utc_now_iso()
        target = self.state_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=str(target.parent),
            prefix=".state.",
            suffix=".tmp",
        ) as tmp:
            json.dump(payload, tmp, indent=2, sort_keys=True)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, target)
        return target

    # ------------------------------------------------------------------
    # Decision log — append-only
    # ------------------------------------------------------------------

    def _append_log(self, kind: str, payload: dict[str, Any]) -> None:
        target = self.decision_log_path
        target.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": _utc_now_iso(), "kind": kind, "payload": payload}
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True))
            fh.write("\n")

    def log_decision(self, decision: ControllerDecision, *, persist: bool = True) -> None:
        self.controller_step_count += 1
        self.stage = decision.stage
        self._append_log("decision", decision.model_dump(mode="json"))
        if persist:
            self.save()

    def log_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result_summary: dict[str, Any],
        *,
        persist: bool = True,
    ) -> None:
        self.tool_call_count += 1
        self._append_log(
            "tool_call",
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "result_summary": result_summary,
            },
        )
        if persist:
            self.save()

    # ------------------------------------------------------------------
    # Tier-3 escalation tracking
    # ------------------------------------------------------------------

    def get_tweak_depth(self, tool_name: str, scope: str) -> int:
        return self.tweak_depth.get(_tweak_key(tool_name, scope), 0)

    def bump_tweak(self, tool_name: str, scope: str, *, persist: bool = True) -> int:
        key = _tweak_key(tool_name, scope)
        depth = self.tweak_depth.get(key, 0) + 1
        self.tweak_depth[key] = depth
        if persist:
            self.save()
        return depth

    def reset_tweak(self, tool_name: str, scope: str, *, persist: bool = True) -> None:
        self.tweak_depth.pop(_tweak_key(tool_name, scope), None)
        if persist:
            self.save()

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def set_stage(self, stage: Stage, *, persist: bool = True) -> None:
        self.stage = stage
        if persist:
            self.save()

    def add_binding_site(
        self, site: BindingSiteRecord, *, persist: bool = True
    ) -> None:
        self.binding_sites[site.site_id] = site
        if persist:
            self.save()

    def add_ligand(
        self, entry: LigandLibraryEntry, *, persist: bool = True
    ) -> None:
        self.ligand_library[entry.ligand_id] = entry
        if persist:
            self.save()

    def add_pose(self, pose: PoseRecord, *, persist: bool = True) -> None:
        self.poses[pose.pose_id] = pose
        if persist:
            self.save()

    def add_segment(self, segment: SegmentEvidence, *, persist: bool = True) -> None:
        self.segments[segment.segment_id] = segment
        if persist:
            self.save()

    def add_ion_candidate(self, candidate: IonCandidate, *, persist: bool = True) -> None:
        self.ion_candidates[candidate.ion_candidate_id] = candidate
        if persist:
            self.save()

    def add_refinement(self, outcome: RefinementOutcome, *, persist: bool = True) -> None:
        self.refinements[outcome.refinement_id] = outcome
        if persist:
            self.save()
