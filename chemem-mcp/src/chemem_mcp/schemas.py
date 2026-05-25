from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1"

Stage = Literal[
    "preparation",
    "density_assignment",
    "ion_analysis",
    "smart_refine_2",
    "final_report",
]

NextAction = Literal[
    "run_tool",
    "advance_stage",
    "accept_assignments",
    "stop_ambiguous",
    "fail_case",
    "cleanup_intermediate_files",
]

AssignmentStatus = Literal[
    "single_candidate",
    "multiple_candidates",
    "noise_solvent",
    "ion_candidate",
    "unexplained",
]

FinalStatus = Literal[
    "accepted",
    "accepted_with_warnings",
    "ambiguous",
    "failed",
]

RefinementStatus = Literal["ok", "failed", "unstable"]


class ToolTier(str, Enum):
    MONOLITHIC = "monolithic"
    ATOMIC = "atomic"
    PARAMETER_TWEAK = "parameter_tweak"


class _Versioned(BaseModel):
    """Common base for top-level persisted artifacts."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Stage 1 — segmentation + candidate evidence
# ---------------------------------------------------------------------------


class AlphaMaskParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threshold_sigma: float
    smoothing: float
    min_blob_volume: float


class BindingSiteRecord(BaseModel):
    """Lightweight per-site record persisted to CaseState by generate_binding_sites."""

    model_config = ConfigDict(extra="forbid")
    site_id: str
    center_xyz: tuple[float, float, float]
    radius_angstrom: float


class LigandLibraryEntry(BaseModel):
    """One ligand registered with the case by prepare_ligand_library."""

    model_config = ConfigDict(extra="forbid")
    ligand_id: str
    source: str


class CandidateScores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dock_score: Optional[float] = None
    minimised_score: Optional[float] = None
    ccc: Optional[float] = None
    mi: Optional[float] = None
    sci: Optional[float] = None
    mapq_mean: Optional[float] = None
    mapq_min: Optional[float] = None
    qscore_mean: Optional[float] = None
    density_coverage: Optional[float] = None
    segment_ligand_volume_ratio: Optional[float] = None
    clash_count: Optional[int] = None
    severe_clash_count: Optional[int] = None
    ligand_strain_kcal_mol: Optional[float] = None
    mmgbsa_kcal_mol: Optional[float] = None


class PoseRecord(BaseModel):
    """One docked pose registered with the case by dock_ligands_to_segment."""

    model_config = ConfigDict(extra="forbid")
    pose_id: str
    segment_id: str
    ligand_id: str
    scores: CandidateScores = Field(default_factory=CandidateScores)


class InteractionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hbonds: int = 0
    salt_bridges: int = 0
    pi_stacking: int = 0
    metal_coordination: int = 0
    key_residues: list[str] = Field(default_factory=list)


class CandidateAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assignment_id: str
    segment_id: str
    ligand_id: str
    pose_id: str
    status: Literal["ok", "failed", "warning"] = "ok"
    scores: CandidateScores = Field(default_factory=CandidateScores)
    interactions: InteractionSummary = Field(default_factory=InteractionSummary)
    warnings: list[str] = Field(default_factory=list)


class NonLigandHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["solvent_or_noise", "ion_or_metal", "unexplained"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class SegmentEvidence(_Versioned):
    segment_id: str
    center_xyz: tuple[float, float, float]
    volume_angstrom3: float
    peak_density: float
    mean_density: float
    local_background_mean: Optional[float] = None
    local_background_std: Optional[float] = None
    nearby_residues: list[str] = Field(default_factory=list)
    possible_binding_site_ids: list[str] = Field(default_factory=list)
    alpha_mask_parameters: Optional[AlphaMaskParameters] = None
    candidate_assignments: list[CandidateAssignment] = Field(default_factory=list)
    non_ligand_hypotheses: list[NonLigandHypothesis] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 3 — ion candidates
# ---------------------------------------------------------------------------


class IonDensity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    peak_density: float
    shape: Literal["compact_spherical", "elongated", "diffuse", "irregular"]
    volume_angstrom3: float


class IonCoordination(BaseModel):
    model_config = ConfigDict(extra="forbid")
    geometry: str
    coordination_number: int
    donor_atoms: list[str]
    mean_distance_angstrom: float
    distance_std_angstrom: float


class IonCandidate(_Versioned):
    ion_candidate_id: str
    segment_id: str
    proposed_identity: str
    confidence: float = Field(ge=0.0, le=1.0)
    density: IonDensity
    coordination: IonCoordination
    red_flags: list[str] = Field(default_factory=list)
    recommended_next_checks: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Controller decisions
# ---------------------------------------------------------------------------


class ControllerDecision(_Versioned):
    controller_step_id: str
    stage: Stage
    next_action: NextAction
    tool_name: Optional[str] = None
    tool_tier: Optional[ToolTier] = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str
    evidence_used: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)

    # Single-LLM self-critique compensation (residual pushback point 1 in the plan).
    self_critique: str
    red_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Stage 4 — refinement outcomes
# ---------------------------------------------------------------------------


class RefinementInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pose_ids: list[str]
    ion_candidate_ids: list[str] = Field(default_factory=list)


class RefinementStability(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nan_detected: bool
    max_ligand_rmsd_from_start: float
    max_protein_local_rmsd: float
    energy_exploded: bool
    coordination_restraints_stable: bool


class RefinementScoreDeltas(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mean_ligand_ccc_delta: Optional[float] = None
    mean_mapq_delta: Optional[float] = None
    severe_clash_delta: Optional[int] = None
    ligand_strain_delta: Optional[float] = None


class RefinementFiles(BaseModel):
    model_config = ConfigDict(extra="forbid")
    refined_model_pdb: Optional[str] = None
    refined_ligands_sdf: Optional[str] = None
    summary_json: Optional[str] = None


class RefinementOutcome(_Versioned):
    refinement_id: str
    status: RefinementStatus
    inputs: RefinementInputs
    stability: RefinementStability
    score_deltas: RefinementScoreDeltas = Field(default_factory=RefinementScoreDeltas)
    files: RefinementFiles = Field(default_factory=RefinementFiles)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 5 — final report
# ---------------------------------------------------------------------------


class AcceptedLigand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ligand_id: str
    segment_id: str
    final_pose_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AcceptedIon(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ion_id: str
    segment_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FinalReport(_Versioned):
    run_id: str
    final_status: FinalStatus
    accepted_ligands: list[AcceptedLigand] = Field(default_factory=list)
    accepted_ions: list[AcceptedIon] = Field(default_factory=list)
    noise_or_solvent_segments: list[str] = Field(default_factory=list)
    unexplained_segments: list[str] = Field(default_factory=list)
    recommended_human_checks: list[str] = Field(default_factory=list)
    files_to_keep: list[str] = Field(default_factory=list)
    files_safe_to_clean: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Block 4 decision-case fixtures
# ---------------------------------------------------------------------------


class ExpectedDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    next_action: NextAction
    tool_name: Optional[str] = None
    tool_tier: Optional[ToolTier] = None
    stage_after: Optional[Stage] = None
    rationale: str


class DecisionCase(_Versioned):
    """Pre-baked agent input + expected decision used in Block 4 CI battery."""

    case_id: str
    description: str
    stage: Stage
    input_evidence: dict[str, Any]
    expected_decision: ExpectedDecision
    notes: str = ""
