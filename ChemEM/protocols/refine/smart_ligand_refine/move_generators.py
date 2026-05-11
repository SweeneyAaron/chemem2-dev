from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .torsion_profiles import (
    candidate_angles_from_profile,
    dihedral_angle_deg,
    set_torsion_angle,
    signed_angle_delta_deg,
)
from .types import (
    AtomClass,
    CandidatePose,
    LigandMapMetrics,
    SmartLigandRefinementConfig,
    TorsionInfo,
    TorsionProfile,
)

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional at import time
    Chem = None


_REPAIR_LIKE_CLASSES = (
    AtomClass.REPAIR,
    AtomClass.WEAK_OR_ABSENT,
    AtomClass.UNCERTAIN,
)


def _class_weight(cls: Optional[AtomClass]) -> float:
    if cls == AtomClass.REPAIR:
        return 1.0
    if cls == AtomClass.WEAK_OR_ABSENT:
        return 0.75
    if cls == AtomClass.UNCERTAIN:
        return 0.5
    return 0.0


def _atom_fit_badness(
    atom_idx: int,
    metrics: LigandMapMetrics,
    atom_classes: Dict[int, AtomClass],
    config: SmartLigandRefinementConfig,
    *,
    max_q: Optional[float] = None,
) -> float:
    cls = atom_classes.get(int(atom_idx))
    weight = _class_weight(cls)
    relative = bool(getattr(config, "branch_relative_q_seeding", False))
    if relative and bool(getattr(config, "branch_use_anchor_seeds", False)) and weight <= 0.0:
        weight = 0.25
    if weight <= 0.0:
        return 0.0
    atom_metric = metrics.atom_metrics.get(int(atom_idx))
    if atom_metric is None:
        return 0.0
    q = float(atom_metric.q_score)
    floor = float(config.target_q)
    if relative and max_q is not None and float(max_q) > floor:
        floor = float(max_q)
    q_gap = max(0.0, floor - q)
    ccc_gap = max(0.0, float(config.anchor_local_ccc_min) - float(atom_metric.local_ccc))
    relative_term = 0.0
    if relative and max_q is not None and float(max_q) > 1e-6:
        relative_term = max(0.0, 1.0 - q / float(max_q))
    return float(weight * (q_gap + 0.5 * ccc_gap + 0.5 * relative_term))


def _rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    theta = np.radians(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    ux, uy, uz = [float(x) for x in axis]
    k_mat = np.array(
        [[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]],
        dtype=np.float64,
    )
    return (np.eye(3) * c) + (s * k_mat) + ((1.0 - c) * np.outer(axis, axis))


def anchor_indices(atom_classes: Dict[int, AtomClass]) -> List[int]:
    return [
        int(idx)
        for idx, cls in atom_classes.items()
        if cls == AtomClass.ANCHOR
    ]


def anchor_rmsd(
    old_coords_A: np.ndarray,
    new_coords_A: np.ndarray,
    atom_classes: Dict[int, AtomClass],
) -> float:
    idx = anchor_indices(atom_classes)
    if not idx:
        return 0.0
    old = np.asarray(old_coords_A, dtype=np.float64)[idx]
    new = np.asarray(new_coords_A, dtype=np.float64)[idx]
    d2 = np.sum((old - new) * (old - new), axis=1)
    return float(np.sqrt(np.mean(d2))) if d2.size else 0.0


def get_torsion_axis(
    coords_A: np.ndarray,
    torsion: TorsionInfo,
) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords_A, dtype=np.float64)
    a, b = [int(x) for x in torsion.bond_atoms]
    origin = coords[a]
    axis = coords[b] - origin
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return origin, np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return origin, axis / norm


def distance_from_torsion_axis(
    coords_A: np.ndarray,
    atom_idx: int,
    torsion: TorsionInfo,
) -> float:
    origin, axis = get_torsion_axis(coords_A, torsion)
    r = np.asarray(coords_A, dtype=np.float64)[int(atom_idx)] - origin
    return float(np.linalg.norm(np.cross(axis, r)))


def rank_torsions_by_badness(
    torsions: Sequence[TorsionInfo],
    coords_A: np.ndarray,
    metrics: LigandMapMetrics,
    atom_classes: Dict[int, AtomClass],
    config: Optional[SmartLigandRefinementConfig] = None,
) -> List[TorsionInfo]:
    config = config or SmartLigandRefinementConfig()
    scored = []
    for torsion in torsions:
        score = 0.0
        for atom_idx in torsion.downstream_atoms:
            atom_idx = int(atom_idx)
            badness = _atom_fit_badness(atom_idx, metrics, atom_classes, config)
            if badness <= 0.0:
                continue
            leverage = distance_from_torsion_axis(coords_A, atom_idx, torsion)
            score += badness * leverage
        scored.append((float(score), torsion))
    scored.sort(reverse=True, key=lambda item: item[0])
    return [
        torsion
        for score, torsion in scored
        if score > float(config.min_torsion_badness)
    ]


def projected_torsion_gradient(
    coords_A: np.ndarray,
    torsion: TorsionInfo,
    metrics: LigandMapMetrics,
    atom_classes: Dict[int, AtomClass],
    config: Optional[SmartLigandRefinementConfig] = None,
) -> float:
    config = config or SmartLigandRefinementConfig()
    axis_origin, axis_unit = get_torsion_axis(coords_A, torsion)
    slope = 0.0
    coords = np.asarray(coords_A, dtype=np.float64)

    for atom_idx in torsion.downstream_atoms:
        atom_idx = int(atom_idx)
        if atom_classes.get(atom_idx) not in _REPAIR_LIKE_CLASSES:
            continue
        atom_metric = metrics.atom_metrics.get(atom_idx)
        if atom_metric is None:
            continue
        q_weight = _atom_fit_badness(atom_idx, metrics, atom_classes, config)
        if q_weight <= 0.0:
            continue
        r = coords[atom_idx]
        grad = np.asarray(atom_metric.map_gradient, dtype=np.float64)
        torsion_motion = np.cross(axis_unit, r - axis_origin)
        slope += q_weight * float(np.dot(grad, torsion_motion))
    return float(slope)


class RigidMicroJiggleMoveGenerator:
    def __init__(
        self,
        config: Optional[SmartLigandRefinementConfig] = None,
        translations_A: Optional[Sequence[float]] = None,
        rotations_deg: Optional[Sequence[float]] = None,
    ):
        self.config = config or SmartLigandRefinementConfig()
        self.translations_A = list(translations_A or [0.1, 0.2, 0.4])
        self.rotations_deg = list(rotations_deg or [2.0, 5.0, 10.0])

    def generate(
        self,
        pose: CandidatePose,
        metrics: LigandMapMetrics,
        atom_classes: Dict[int, AtomClass],
    ) -> List[CandidatePose]:
        coords = np.asarray(pose.coords, dtype=np.float64)
        if coords.size == 0:
            return []
        centroid = np.mean(coords, axis=0)
        candidates: List[CandidatePose] = []
        axes = np.eye(3, dtype=np.float64)

        for mag in self.translations_A:
            for axis_idx, axis in enumerate(axes):
                for sign in (-1.0, 1.0):
                    shift = axis * float(mag) * sign
                    candidates.append(
                        CandidatePose(
                            coords=coords + shift[None, :],
                            move_type="rigid_micro_jiggle",
                            move_metadata={
                                "kind": "translation",
                                "axis": int(axis_idx),
                                "magnitude_A": float(mag) * sign,
                            },
                        )
                    )

        gradient_shift = self._repair_gradient_direction(metrics, atom_classes)
        if gradient_shift is not None:
            for mag in self.translations_A:
                candidates.append(
                    CandidatePose(
                        coords=coords + gradient_shift[None, :] * float(mag),
                        move_type="rigid_micro_jiggle",
                        move_metadata={
                            "kind": "translation",
                            "axis": "repair_gradient",
                            "magnitude_A": float(mag),
                        },
                    )
                )

        for angle in self.rotations_deg:
            for axis_idx, axis in enumerate(axes):
                for sign in (-1.0, 1.0):
                    rot = _rotation_matrix(axis, float(angle) * sign)
                    new_coords = (coords - centroid) @ rot.T + centroid
                    candidates.append(
                        CandidatePose(
                            coords=new_coords,
                            move_type="rigid_micro_jiggle",
                            move_metadata={
                                "kind": "rotation",
                                "axis": int(axis_idx),
                                "angle_deg": float(angle) * sign,
                            },
                        )
                    )

        return candidates[: int(self.config.max_candidates_per_stage)]

    @staticmethod
    def _repair_gradient_direction(
        metrics: LigandMapMetrics,
        atom_classes: Dict[int, AtomClass],
    ) -> Optional[np.ndarray]:
        vec = np.zeros(3, dtype=np.float64)
        for atom_idx, cls in atom_classes.items():
            if cls not in _REPAIR_LIKE_CLASSES:
                continue
            metric = metrics.atom_metrics.get(int(atom_idx))
            if metric is None:
                continue
            vec += np.asarray(metric.map_gradient, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return None
        return vec / norm


class TorsionMinimaMoveGenerator:
    def __init__(
        self,
        config: Optional[SmartLigandRefinementConfig] = None,
        offsets_deg: Optional[Sequence[float]] = None,
    ):
        self.config = config or SmartLigandRefinementConfig()
        self.offsets_deg = list(offsets_deg or [0.0, -5.0, 5.0, -10.0, 10.0])

    def generate(
        self,
        pose: CandidatePose,
        torsion: TorsionInfo,
        torsion_profile: TorsionProfile,
        metrics: LigandMapMetrics,
        atom_classes: Dict[int, AtomClass],
    ) -> List[CandidatePose]:
        coords = np.asarray(pose.coords, dtype=np.float64)
        slope = projected_torsion_gradient(
            coords, torsion, metrics, atom_classes, self.config
        )
        current = dihedral_angle_deg(coords, torsion.atom_indices)
        angles = candidate_angles_from_profile(torsion_profile, self.offsets_deg)

        def sort_key(angle: float):
            delta = signed_angle_delta_deg(angle, current)
            preferred = -np.sign(slope) * delta
            return (preferred, abs(delta))

        if abs(float(slope)) > 1e-12:
            angles = sorted(angles, key=sort_key)
        else:
            angles = sorted(angles, key=lambda a: abs(signed_angle_delta_deg(a, current)))

        if bool(self.config.enable_directed_torsion_proposals) and abs(float(slope)) > 1e-12:
            step = abs(float(self.config.directed_torsion_step_deg))
            sign = 1.0 if float(slope) > 0.0 else -1.0
            for multiplier in (1.0, 2.0):
                angle = (float(current) + sign * step * multiplier) % 360.0
                if not any(abs(signed_angle_delta_deg(angle, old)) < 1e-6 for old in angles):
                    angles.append(float(angle))

        candidates = []
        for angle in angles:
            new_coords = set_torsion_angle(coords, torsion, angle)
            candidates.append(
                CandidatePose(
                    coords=new_coords,
                    move_type="torsion_minima",
                    move_metadata={
                        "torsion_id": int(torsion.torsion_id),
                        "old_angle_deg": float(current),
                        "new_angle_deg": float(angle),
                        "projected_gradient": float(slope),
                        "profile_source": str(torsion_profile.source),
                        "profile_minima_deg": list(torsion_profile.minima_deg),
                    },
                )
            )
        return candidates[: int(self.config.max_candidates_per_stage)]


@dataclass(frozen=True)
class _BranchStep:
    torsion: TorsionInfo
    target_atom: int


@dataclass(frozen=True)
class _BranchPath:
    seed_atom: int
    steps: Tuple[_BranchStep, ...]


class BranchRebuildMoveGenerator:
    """
    Targeted worst-atom branch rebuild hook.

    This generator finds directional torsion paths leading to the worst
    repair-like atoms, then walks each path from proximal torsions to distal
    torsions. Intermediate beam ranking uses only the current step target atom,
    while final candidates are rescored by the normal SmartLigandRefinement
    scorer and accepted or rejected by the caller.
    """

    def __init__(
        self,
        config: Optional[SmartLigandRefinementConfig] = None,
        *,
        ligand=None,
    ):
        self.config = config or SmartLigandRefinementConfig()
        self.ligand = ligand
        self._progress_callback: Optional[Callable[..., None]] = None

    def _emit(self, event_type: str, **payload) -> None:
        cb = self._progress_callback
        if cb is None:
            return
        if not bool(getattr(self.config, "branch_rebuild_verbose", False)):
            return
        try:
            cb(event_type, **payload)
        except Exception:
            pass

    def generate(
        self,
        pose: CandidatePose,
        torsions: Sequence[TorsionInfo],
        torsion_profiles: Dict[int, TorsionProfile],
        metrics: LigandMapMetrics,
        atom_classes: Dict[int, AtomClass],
    ) -> List[CandidatePose]:
        branch_paths = self._targeted_branch_paths(
            pose,
            torsions,
            torsion_profiles,
            metrics,
            atom_classes,
        )
        if not branch_paths:
            return []

        out: List[CandidatePose] = []
        for branch in branch_paths:
            beam: List[CandidatePose] = [
                CandidatePose(
                    coords=np.asarray(pose.coords, dtype=np.float64),
                    move_type=pose.move_type,
                    move_metadata={
                        "seed_atom": int(branch.seed_atom),
                        "branch_path": self._branch_path_metadata(branch),
                        "torsion_sequence": [],
                    },
                    map_metrics=pose.map_metrics,
                    geometry_metrics=pose.geometry_metrics,
                    score=pose.score,
                )
            ]
            for depth, step in enumerate(branch.steps, start=1):
                profile = torsion_profiles.get(int(step.torsion.torsion_id))
                if profile is None:
                    continue
                new_beam: List[CandidatePose] = []
                for state in beam:
                    new_beam.extend(
                        self._generate_step_candidates(
                            state,
                            branch,
                            step,
                            profile,
                            depth,
                        )
                    )
                beam = new_beam[: max(1, int(self.config.branch_beam_width))]
            out.extend(beam)
            if len(out) >= int(self.config.max_candidates_per_stage):
                break
        return out[: int(self.config.max_candidates_per_stage)]

    def generate_scored(
        self,
        pose: CandidatePose,
        torsions: Sequence[TorsionInfo],
        torsion_profiles: Dict[int, TorsionProfile],
        metrics: LigandMapMetrics,
        atom_classes: Dict[int, AtomClass],
        baseline_geometry: Optional[dict],
        scorer,
        *,
        clean_fn: Optional[Callable[[CandidatePose], CandidatePose]] = None,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> List[CandidatePose]:
        self._progress_callback = progress_callback
        branch_paths = self._targeted_branch_paths(
            pose,
            torsions,
            torsion_profiles,
            metrics,
            atom_classes,
        )
        if not branch_paths:
            self._progress_callback = None
            return []

        final_candidates: List[CandidatePose] = []
        for branch in branch_paths:
            branch_pose = CandidatePose(
                coords=np.asarray(pose.coords, dtype=np.float64),
                move_type=pose.move_type,
                move_metadata={
                    "seed_atom": int(branch.seed_atom),
                    "branch_path": self._branch_path_metadata(branch),
                    "torsion_sequence": [],
                },
                map_metrics=metrics,
                geometry_metrics=baseline_geometry,
                score=pose.score,
            )
            beam: List[CandidatePose] = [branch_pose]
            for depth, step in enumerate(branch.steps, start=1):
                profile = torsion_profiles.get(int(step.torsion.torsion_id))
                if profile is None:
                    continue
                generated: List[Tuple[CandidatePose, LigandMapMetrics]] = []
                for state in beam:
                    state_metrics = state.map_metrics or metrics
                    for candidate in self._generate_step_candidates(
                        state,
                        branch,
                        step,
                        profile,
                        depth,
                    ):
                        generated.append(
                            (
                                candidate,
                                state_metrics,
                            )
                        )

                evaluated: List[CandidatePose] = []
                for candidate, state_metrics in generated:
                    cand = clean_fn(candidate) if clean_fn is not None else candidate
                    try:
                        cand = self._score_intermediate_candidate(
                            cand,
                            pose,
                            metrics,
                            state_metrics,
                            atom_classes,
                            baseline_geometry,
                            scorer,
                            target_atom=int(step.target_atom),
                        )
                        if cand.score is not None and np.isfinite(float(cand.score)):
                            evaluated.append(cand)
                    except Exception as exc:
                        cand.score = float("-inf")
                        cand.move_metadata["beam_error"] = type(exc).__name__
                        evaluated.append(cand)

                evaluated.sort(
                    key=lambda cand: float(cand.score)
                    if cand.score is not None and np.isfinite(float(cand.score))
                    else float("-inf"),
                    reverse=True,
                )
                beam = evaluated[: max(1, int(self.config.branch_beam_width))]
                if progress_callback is not None:
                    rejection_counts: Dict[str, int] = {
                        "anchor_rmsd": 0,
                        "geometry": 0,
                        "clash": 0,
                        "target_score_negative": 0,
                        "non_finite": 0,
                    }
                    accepted_for_beam = 0
                    for cand in evaluated:
                        terms = (cand.move_metadata or {}).get(
                            "target_score_terms", {}
                        )
                        if bool(terms.get("intermediate_rejected", False)):
                            if float(terms.get("anchor_disruption", 0.0)) > float(
                                self.config.max_anchor_rmsd
                            ):
                                rejection_counts["anchor_rmsd"] += 1
                            elif float(terms.get("clash_penalty", 0.0)) > float(
                                self.config.max_clash_score_increase
                            ):
                                rejection_counts["clash"] += 1
                            else:
                                rejection_counts["geometry"] += 1
                        elif (
                            cand.score is None
                            or not np.isfinite(float(cand.score))
                        ):
                            rejection_counts["non_finite"] += 1
                        elif float(cand.score) < 0.0:
                            rejection_counts["target_score_negative"] += 1
                        else:
                            accepted_for_beam += 1
                    progress_callback(
                        "branch_beam",
                        beam_depth=int(depth),
                        beam_width=int(self.config.branch_beam_width),
                        seed_atom=int(branch.seed_atom),
                        target_atom=int(step.target_atom),
                        torsion_id=int(step.torsion.torsion_id),
                        candidates_generated=int(len(generated)),
                        candidates_scored=int(len(evaluated)),
                        candidates_kept=int(len(beam)),
                        candidates_target_positive=int(accepted_for_beam),
                        rejection_counts=rejection_counts,
                        best_score=(
                            None
                            if not beam or beam[0].score is None
                            else float(beam[0].score)
                        ),
                        profile_source=str(profile.source),
                        profile_minima_deg=list(profile.minima_deg),
                    )
            final_candidates.extend(beam)
            if len(final_candidates) >= int(self.config.max_candidates_per_stage):
                break

        try:
            return self._score_final_candidates(
                final_candidates,
                pose,
                metrics,
                atom_classes,
                baseline_geometry,
                scorer,
            )
        finally:
            self._progress_callback = None

    def _targeted_branch_paths(
        self,
        pose: CandidatePose,
        torsions: Sequence[TorsionInfo],
        torsion_profiles: Dict[int, TorsionProfile],
        metrics: LigandMapMetrics,
        atom_classes: Dict[int, AtomClass],
    ) -> List[_BranchPath]:
        class_counts: Dict[str, int] = {
            AtomClass.ANCHOR.value: 0,
            AtomClass.REPAIR.value: 0,
            AtomClass.WEAK_OR_ABSENT.value: 0,
            AtomClass.UNCERTAIN.value: 0,
        }
        for cls in atom_classes.values():
            key = cls.value if hasattr(cls, "value") else str(cls)
            class_counts[key] = class_counts.get(key, 0) + 1

        max_q = 0.0
        if metrics.atom_metrics:
            max_q = max(
                float(am.q_score) for am in metrics.atom_metrics.values()
            )

        use_anchor_seeds = bool(
            getattr(self.config, "branch_use_anchor_seeds", False)
        )
        relative_seeding = bool(
            getattr(self.config, "branch_relative_q_seeding", False)
        )

        repair_like = {
            int(idx)
            for idx, cls in atom_classes.items()
            if cls in _REPAIR_LIKE_CLASSES
        }

        anchor_percentile_seeds: List[int] = []
        if use_anchor_seeds and metrics.atom_metrics:
            sorted_by_q = sorted(
                metrics.atom_metrics.items(),
                key=lambda item: float(item[1].q_score),
            )
            pct = float(getattr(self.config, "branch_relative_q_percentile", 0.25))
            pct = min(max(pct, 0.0), 1.0)
            n_pct = max(1, int(round(pct * len(sorted_by_q))))
            anchor_percentile_seeds = [int(idx) for idx, _ in sorted_by_q[:n_pct]]

        candidate_pool = set(repair_like) | set(anchor_percentile_seeds)
        if not candidate_pool:
            self._emit(
                "branch_seed_summary",
                class_counts=dict(class_counts),
                worst_atom_indices=list(metrics.worst_atom_indices),
                seeds=[],
                used_badness_fallback=False,
                seed_pool_empty=True,
                max_q=float(max_q),
                anchor_seeds_added=0,
            )
            return []

        from_worst = [
            int(idx)
            for idx in metrics.worst_atom_indices
            if int(idx) in candidate_pool and int(idx) in metrics.atom_metrics
        ]
        used_badness_fallback = False
        if from_worst and not use_anchor_seeds:
            seeds = from_worst
        else:
            used_badness_fallback = True
            seeds = sorted(
                candidate_pool,
                key=lambda idx: _atom_fit_badness(
                    idx, metrics, atom_classes, self.config, max_q=max_q
                ),
                reverse=True,
            )

        seed_records: List[Dict[str, object]] = []
        anchor_seeds_added = 0
        for sidx in seeds:
            atom_metric = metrics.atom_metrics.get(int(sidx))
            cls = atom_classes.get(int(sidx))
            is_anchor_seed = (
                int(sidx) not in repair_like and int(sidx) in anchor_percentile_seeds
            )
            if is_anchor_seed:
                anchor_seeds_added += 1
            seed_records.append(
                {
                    "atom_index": int(sidx),
                    "q_score": (
                        None if atom_metric is None else float(atom_metric.q_score)
                    ),
                    "local_ccc": (
                        None if atom_metric is None else float(atom_metric.local_ccc)
                    ),
                    "atom_class": (
                        cls.value if hasattr(cls, "value") else str(cls)
                    ),
                    "anchor_seed": bool(is_anchor_seed),
                    "badness": float(
                        _atom_fit_badness(
                            int(sidx),
                            metrics,
                            atom_classes,
                            self.config,
                            max_q=max_q,
                        )
                    ),
                }
            )
        self._emit(
            "branch_seed_summary",
            class_counts=dict(class_counts),
            worst_atom_indices=list(metrics.worst_atom_indices),
            seeds=seed_records,
            used_badness_fallback=bool(used_badness_fallback),
            seed_pool_empty=False,
            max_q=float(max_q),
            anchor_seeds_added=int(anchor_seeds_added),
            relative_q_seeding=bool(relative_seeding),
            use_anchor_seeds=bool(use_anchor_seeds),
        )

        adjacency = self._adjacency(int(np.asarray(pose.coords).shape[0]))
        max_depth = max(1, int(self.config.max_branch_torsions))
        paths: List[_BranchPath] = []
        seen = set()
        relax_repair_like_filter = use_anchor_seeds
        for seed in seeds:
            path_torsions = self._path_torsions_for_seed(
                int(seed),
                torsions,
                torsion_profiles,
                metrics,
                atom_classes,
                adjacency,
                relax_repair_like_filter=relax_repair_like_filter,
                max_q=max_q,
            )
            if not path_torsions:
                continue
            steps = tuple(
                _BranchStep(
                    torsion=torsion,
                    target_atom=self._target_atom_for_step(torsion, int(seed), metrics),
                )
                for torsion in path_torsions[:max_depth]
            )
            key = tuple(int(step.torsion.torsion_id) for step in steps)
            if not key or key in seen:
                continue
            seen.add(key)
            paths.append(_BranchPath(seed_atom=int(seed), steps=steps))
        return paths

    def _path_torsions_for_seed(
        self,
        seed_atom: int,
        torsions: Sequence[TorsionInfo],
        torsion_profiles: Dict[int, TorsionProfile],
        metrics: LigandMapMetrics,
        atom_classes: Dict[int, AtomClass],
        adjacency: Dict[int, set],
        *,
        relax_repair_like_filter: bool = False,
        max_q: Optional[float] = None,
    ) -> List[TorsionInfo]:
        candidates = []
        skipped_no_profile = 0
        skipped_seed_not_downstream = 0
        skipped_no_repair_like = 0
        for torsion in torsions:
            if int(torsion.torsion_id) not in torsion_profiles:
                skipped_no_profile += 1
                continue
            downstream = {int(i) for i in torsion.downstream_atoms}
            if int(seed_atom) not in downstream:
                skipped_seed_not_downstream += 1
                continue
            if not relax_repair_like_filter and not any(
                atom_classes.get(int(idx)) in _REPAIR_LIKE_CLASSES
                for idx in downstream
            ):
                skipped_no_repair_like += 1
                continue
            dist = self._graph_distance(int(torsion.bond_atoms[1]), int(seed_atom), adjacency)
            badness = sum(
                _atom_fit_badness(
                    int(idx), metrics, atom_classes, self.config, max_q=max_q
                )
                for idx in downstream
            )
            candidates.append(
                (
                    -len(downstream),
                    -dist,
                    -float(badness),
                    int(torsion.torsion_id),
                    torsion,
                    downstream,
                )
            )

        candidates.sort(key=lambda item: item[:4])
        path: List[TorsionInfo] = []
        diagnostic_records: List[Dict[str, object]] = []
        previous_downstream: Optional[set] = None
        for *prefix, torsion, downstream in candidates:
            keep = True
            drop_reason: Optional[str] = None
            if previous_downstream is not None:
                if not downstream.issubset(previous_downstream):
                    keep = False
                    drop_reason = "not_subset_of_previous"
                elif downstream == previous_downstream:
                    keep = False
                    drop_reason = "equal_to_previous"
            diagnostic_records.append(
                {
                    "torsion_id": int(torsion.torsion_id),
                    "downstream_size": int(len(downstream)),
                    "distance_to_seed": int(-prefix[1]),
                    "sum_badness": float(-prefix[2]),
                    "kept": bool(keep),
                    "drop_reason": drop_reason,
                }
            )
            if keep:
                path.append(torsion)
                previous_downstream = downstream

        self._emit(
            "branch_path_candidates",
            seed_atom=int(seed_atom),
            skipped_no_profile=int(skipped_no_profile),
            skipped_seed_not_downstream=int(skipped_seed_not_downstream),
            skipped_no_repair_like=int(skipped_no_repair_like),
            considered=len(diagnostic_records),
            path_torsion_ids=[int(t.torsion_id) for t in path],
            candidates=diagnostic_records,
        )
        return path

    def _generate_step_candidates(
        self,
        state: CandidatePose,
        branch: _BranchPath,
        step: _BranchStep,
        profile: TorsionProfile,
        depth: int,
    ) -> List[CandidatePose]:
        torsion = step.torsion
        current = dihedral_angle_deg(state.coords, torsion.atom_indices)
        previous_sequence = list((state.move_metadata or {}).get("torsion_sequence", []))
        out: List[CandidatePose] = []
        for angle, proposal_kind in self._angle_options(torsion, profile, current):
            sequence_entry = {
                "torsion_id": int(torsion.torsion_id),
                "old_angle_deg": float(current),
                "new_angle_deg": float(angle),
                "profile_source": str(profile.source),
                "target_atom": int(step.target_atom),
                "downstream_size": int(len(torsion.downstream_atoms)),
                "proposal_kind": str(proposal_kind),
            }
            sequence = previous_sequence + [sequence_entry]
            out.append(
                CandidatePose(
                    coords=set_torsion_angle(state.coords, torsion, angle),
                    move_type="branch_rebuild",
                    move_metadata={
                        "torsion_id": int(torsion.torsion_id),
                        "old_angle_deg": float(current),
                        "new_angle_deg": float(angle),
                        "beam_depth": int(depth),
                        "seed_atom": int(branch.seed_atom),
                        "target_atom": int(step.target_atom),
                        "proposal_kind": str(proposal_kind),
                        "branch_path": self._branch_path_metadata(branch),
                        "torsion_sequence": sequence,
                        "profile_minima_deg": list(profile.minima_deg),
                    },
                )
            )
        return out

    def _angle_options(
        self,
        torsion: TorsionInfo,
        profile: TorsionProfile,
        current_angle_deg: float,
    ) -> List[Tuple[float, str]]:
        aggressive = bool(
            getattr(self.config, "branch_aggressive_angle_search", False)
        )
        options: List[Tuple[float, str]] = []
        seen: List[float] = []

        def _try_add(angle: float, kind: str) -> bool:
            wrapped = float(angle) % 360.0
            for old in seen:
                if abs(signed_angle_delta_deg(wrapped, old)) < 2.0:
                    return False
            options.append((float(wrapped), str(kind)))
            seen.append(float(wrapped))
            return True

        for angle in candidate_angles_from_profile(
            profile,
            self.config.branch_minimum_offsets_deg,
        ):
            _try_add(float(angle), "torsion_minimum")

        ring_flip_eligible = self._is_exocyclic_ring_flip_torsion(torsion)
        ring_flip_used = False
        if bool(getattr(self.config, "enable_ring_flip_proposals", True)) and (
            ring_flip_eligible or aggressive
        ):
            flip_angle = (float(current_angle_deg) + 180.0) % 360.0
            if _try_add(flip_angle, "ring_flip"):
                ring_flip_used = True

        grid_added = 0
        flip_added = 0
        if aggressive:
            step = float(getattr(self.config, "branch_aggressive_grid_step_deg", 30.0))
            if step > 0.0:
                n_steps = max(1, int(round(360.0 / step)))
                for i in range(n_steps):
                    if _try_add(i * step, "grid_coarse"):
                        grid_added += 1
            for offset in getattr(self.config, "branch_aggressive_flip_offsets_deg", []):
                if _try_add(float(current_angle_deg) + float(offset), "flip"):
                    flip_added += 1

        self._emit(
            "branch_angle_options",
            torsion_id=int(torsion.torsion_id),
            current_angle_deg=float(current_angle_deg),
            profile_minima_deg=list(profile.minima_deg),
            offsets_deg=list(self.config.branch_minimum_offsets_deg),
            aggressive=bool(aggressive),
            ring_flip_eligible=bool(ring_flip_eligible),
            ring_flip_proposed=bool(ring_flip_used),
            grid_added=int(grid_added),
            flip_added=int(flip_added),
            angle_options=[
                {"angle_deg": float(a), "kind": str(kind)} for a, kind in options
            ],
        )
        return options

    def _score_intermediate_candidate(
        self,
        candidate: CandidatePose,
        baseline_pose: CandidatePose,
        baseline_metrics: LigandMapMetrics,
        state_metrics: LigandMapMetrics,
        atom_classes: Dict[int, AtomClass],
        baseline_geometry: Optional[dict],
        scorer,
        *,
        target_atom: int,
    ) -> CandidatePose:
        map_evaluator = getattr(scorer, "map_metric_evaluator", None)
        geometry_oracle = getattr(scorer, "geometry_oracle", None)
        if map_evaluator is None or geometry_oracle is None:
            candidate = scorer.score(
                candidate,
                baseline_pose,
                baseline_metrics,
                atom_classes,
                baseline_geometry,
            )
            candidate.move_metadata["target_score_terms"] = {}
            return candidate

        candidate.map_metrics = map_evaluator.evaluate(candidate.coords)
        candidate.geometry_metrics = geometry_oracle.evaluate(candidate)

        target_score, terms = self._target_atom_score(
            state_metrics,
            candidate.map_metrics,
            int(target_atom),
        )
        anchor_disruption = anchor_rmsd(
            baseline_pose.coords,
            candidate.coords,
            atom_classes,
        )
        clash_penalty = self._clash_increase(candidate.geometry_metrics, baseline_geometry)
        rejected = False
        if anchor_disruption > float(self.config.max_anchor_rmsd):
            target_score = float("-inf")
            rejected = True
        if not bool(geometry_oracle.is_acceptable(candidate)):
            target_score = float("-inf")
            rejected = True
        if clash_penalty > float(self.config.max_clash_score_increase):
            target_score = float("-inf")
            rejected = True

        candidate.score = float(target_score)
        candidate.move_metadata["target_score_terms"] = {
            **terms,
            "anchor_disruption": float(anchor_disruption),
            "clash_penalty": float(clash_penalty),
            "intermediate_rejected": bool(rejected),
        }
        return candidate

    def _score_final_candidates(
        self,
        candidates: Sequence[CandidatePose],
        pose: CandidatePose,
        metrics: LigandMapMetrics,
        atom_classes: Dict[int, AtomClass],
        baseline_geometry: Optional[dict],
        scorer,
    ) -> List[CandidatePose]:
        evaluated: List[CandidatePose] = []
        for candidate in candidates:
            target_score = candidate.score
            candidate.move_metadata["branch_target_score"] = (
                None if target_score is None else float(target_score)
            )
            try:
                cand = scorer.score(
                    candidate,
                    pose,
                    metrics,
                    atom_classes,
                    baseline_geometry,
                )
                evaluated.append(cand)
            except Exception as exc:
                candidate.score = float("-inf")
                candidate.move_metadata["final_score_error"] = type(exc).__name__
                evaluated.append(candidate)
        evaluated.sort(
            key=lambda cand: float(cand.score)
            if cand.score is not None and np.isfinite(float(cand.score))
            else float("-inf"),
            reverse=True,
        )
        return evaluated[: int(self.config.max_candidates_per_stage)]

    @staticmethod
    def _target_atom_score(
        old_metrics: LigandMapMetrics,
        new_metrics: LigandMapMetrics,
        target_atom: int,
    ) -> Tuple[float, Dict[str, float]]:
        old = old_metrics.atom_metrics.get(int(target_atom))
        new = new_metrics.atom_metrics.get(int(target_atom))
        if old is None or new is None:
            return float("-inf"), {
                "delta_target_q": float("-inf"),
                "delta_target_local_ccc": float("-inf"),
            }
        delta_q = float(new.q_score) - float(old.q_score)
        delta_local = float(new.local_ccc) - float(old.local_ccc)
        score = float(delta_q + 0.5 * delta_local)
        return score, {
            "delta_target_q": float(delta_q),
            "delta_target_local_ccc": float(delta_local),
        }

    @staticmethod
    def _clash_increase(
        new_geometry: Optional[dict],
        old_geometry: Optional[dict],
    ) -> float:
        if not new_geometry:
            return 0.0
        old_clash = 0.0 if old_geometry is None else float(
            old_geometry.get("protein_ligand_clash_score", 0.0)
        )
        new_clash = float(new_geometry.get("protein_ligand_clash_score", 0.0))
        return float(max(0.0, new_clash - old_clash))

    def _target_atom_for_step(
        self,
        torsion: TorsionInfo,
        seed_atom: int,
        metrics: LigandMapMetrics,
    ) -> int:
        direct = int(torsion.atom_indices[3])
        if direct in metrics.atom_metrics:
            return direct
        if int(seed_atom) in metrics.atom_metrics:
            return int(seed_atom)
        for atom_idx in torsion.downstream_atoms:
            if int(atom_idx) in metrics.atom_metrics:
                return int(atom_idx)
        return int(seed_atom)

    @staticmethod
    def _branch_path_metadata(branch: _BranchPath) -> List[dict]:
        return [
            {
                "torsion_id": int(step.torsion.torsion_id),
                "bond_atoms": tuple(int(i) for i in step.torsion.bond_atoms),
                "atom_indices": tuple(int(i) for i in step.torsion.atom_indices),
                "target_atom": int(step.target_atom),
                "downstream_size": int(len(step.torsion.downstream_atoms)),
            }
            for step in branch.steps
        ]

    def _adjacency(self, n_atoms: int) -> Dict[int, set]:
        mol = _ligand_mol(self.ligand)
        adjacency: Dict[int, set] = {int(i): set() for i in range(int(n_atoms))}
        if mol is None or not hasattr(mol, "GetBonds"):
            return adjacency
        for bond in mol.GetBonds():
            i = int(bond.GetBeginAtomIdx())
            j = int(bond.GetEndAtomIdx())
            if i >= int(n_atoms) or j >= int(n_atoms):
                continue
            adjacency.setdefault(i, set()).add(j)
            adjacency.setdefault(j, set()).add(i)
        return adjacency

    @staticmethod
    def _graph_distance(start: int, goal: int, adjacency: Dict[int, set]) -> int:
        if int(start) == int(goal):
            return 0
        visited = {int(start)}
        queue: List[Tuple[int, int]] = [(int(start), 0)]
        while queue:
            atom_idx, dist = queue.pop(0)
            for nb in adjacency.get(atom_idx, set()):
                nb = int(nb)
                if nb == int(goal):
                    return int(dist + 1)
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return 0

    def _is_exocyclic_ring_flip_torsion(self, torsion: TorsionInfo) -> bool:
        mol = _ligand_mol(self.ligand)
        if Chem is None or mol is None or not hasattr(mol, "GetAtomWithIdx"):
            return False
        try:
            work = Chem.RemoveHs(mol)
            i, j, k, l = [int(x) for x in torsion.atom_indices]
            if max(i, j, k, l) >= int(work.GetNumAtoms()):
                return False
            ring_atoms = {
                int(atom.GetIdx())
                for atom in work.GetAtoms()
                if bool(atom.IsInRing())
            }
            ranks = list(Chem.CanonicalRankAtoms(work, breakTies=False))
            if k in ring_atoms and l in ring_atoms and j not in ring_atoms and i not in ring_atoms:
                a, b, c = l, k, j
            elif i in ring_atoms and j in ring_atoms and k not in ring_atoms and l not in ring_atoms:
                a, b, c = i, j, k
            else:
                return False

            neighbours = [
                int(nb.GetIdx())
                for nb in work.GetAtomWithIdx(int(b)).GetNeighbors()
                if int(nb.GetIdx()) not in (int(a), int(c))
            ]
            rank_a = ranks[int(a)]
            if any(ranks[int(nb)] == rank_a for nb in neighbours):
                return False
            return True
        except Exception:
            return False


def _best_rigid_transform_for_cluster(
    cluster_coords_A: np.ndarray,
    target_dirs: np.ndarray,
    epsilon_A: float,
) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(cluster_coords_A, dtype=np.float64)
    dirs = np.asarray(target_dirs, dtype=np.float64)
    desired = coords + float(epsilon_A) * dirs
    if coords.shape[0] <= 1:
        return np.eye(3, dtype=np.float64), (desired - coords).reshape((-1, 3))[0]

    c_src = coords.mean(axis=0)
    c_dst = desired.mean(axis=0)
    p = coords - c_src
    q = desired - c_dst
    try:
        u, _, vt = np.linalg.svd(p.T @ q)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64), c_dst - c_src
    sign = float(np.sign(np.linalg.det(vt.T @ u.T))) or 1.0
    d = np.diag([1.0, 1.0, sign])
    r = vt.T @ d @ u.T
    t = c_dst - r @ c_src
    return r, t


def _ligand_mol(ligand):
    return getattr(ligand, "mol", ligand)


def _connected_repair_like_clusters(
    ligand,
    bad_atoms: Sequence[int],
    n_atoms: int,
) -> List[np.ndarray]:
    mol = _ligand_mol(ligand)
    bad = set(int(i) for i in bad_atoms if 0 <= int(i) < int(n_atoms))
    if not bad:
        return []
    if mol is None or not hasattr(mol, "GetBonds"):
        return [np.asarray(sorted(bad), dtype=int)]

    adjacency: Dict[int, set] = {int(i): set() for i in range(int(n_atoms))}
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        if i >= int(n_atoms) or j >= int(n_atoms):
            continue
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)

    clusters: List[np.ndarray] = []
    visited = set()
    for seed in sorted(bad):
        if seed in visited:
            continue
        stack = [seed]
        component = []
        while stack:
            atom_idx = int(stack.pop())
            if atom_idx in visited:
                continue
            visited.add(atom_idx)
            component.append(atom_idx)
            for nb in adjacency.get(atom_idx, set()):
                if nb in bad and nb not in visited:
                    stack.append(int(nb))
        clusters.append(np.asarray(sorted(component), dtype=int))
    return clusters


class SubregionRigidBodyMoveGenerator:
    def __init__(self, config: Optional[SmartLigandRefinementConfig] = None):
        self.config = config or SmartLigandRefinementConfig()

    def generate(
        self,
        ligand,
        pose: CandidatePose,
        metrics: LigandMapMetrics,
        atom_classes: Dict[int, AtomClass],
    ) -> List[CandidatePose]:
        if not bool(self.config.enable_subregion_proposals):
            return []
        coords = np.asarray(pose.coords, dtype=np.float64)
        bad_atoms = [
            int(idx)
            for idx in metrics.atom_metrics
            if _atom_fit_badness(int(idx), metrics, atom_classes, self.config) > 0.0
        ]
        clusters = _connected_repair_like_clusters(ligand, bad_atoms, coords.shape[0])
        valid = [
            cluster
            for cluster in clusters
            if int(self.config.subregion_min_size)
            <= int(cluster.size)
            <= int(self.config.subregion_max_size)
        ]
        valid.sort(
            key=lambda cluster: -sum(
                _atom_fit_badness(int(idx), metrics, atom_classes, self.config)
                for idx in cluster
            )
        )

        candidates: List[CandidatePose] = []
        for cluster in valid:
            grad_rows = []
            for atom_idx in cluster:
                metric = metrics.atom_metrics.get(int(atom_idx))
                grad = (
                    np.zeros(3, dtype=np.float64)
                    if metric is None
                    else np.asarray(metric.map_gradient, dtype=np.float64)
                )
                norm = float(np.linalg.norm(grad))
                grad_rows.append(grad / norm if norm > 1e-12 else np.zeros(3))
            dirs = np.asarray(grad_rows, dtype=np.float64)
            if not np.any(np.linalg.norm(dirs, axis=1) > 1e-12):
                continue
            r_mat, t_vec = _best_rigid_transform_for_cluster(
                coords[cluster],
                dirs,
                float(self.config.subregion_step_A),
            )
            new_coords = coords.copy()
            new_coords[cluster] = coords[cluster] @ r_mat.T + t_vec
            candidates.append(
                CandidatePose(
                    coords=new_coords,
                    move_type="subregion_rigid_body",
                    move_metadata={
                        "cluster_atoms": [int(i) for i in cluster.tolist()],
                        "cluster_size": int(cluster.size),
                        "step_A": float(self.config.subregion_step_A),
                    },
                )
            )
            if len(candidates) >= int(self.config.max_candidates_per_stage):
                break
        return candidates


class LocalPolishMoveGenerator:
    def __init__(self, config: Optional[SmartLigandRefinementConfig] = None):
        self.config = config or SmartLigandRefinementConfig()

    def generate(
        self,
        pose: CandidatePose,
        torsions: Sequence[TorsionInfo],
    ) -> List[CandidatePose]:
        coords = np.asarray(pose.coords, dtype=np.float64)
        candidates: List[CandidatePose] = []
        for torsion in torsions:
            current = dihedral_angle_deg(coords, torsion.atom_indices)
            for offset in (-5.0, 5.0):
                angle = current + offset
                candidates.append(
                    CandidatePose(
                        coords=set_torsion_angle(coords, torsion, angle),
                        move_type="local_polish",
                        move_metadata={
                            "torsion_id": int(torsion.torsion_id),
                            "old_angle_deg": float(current),
                            "new_angle_deg": float(angle),
                        },
                    )
                )
        candidates.extend(
            RigidMicroJiggleMoveGenerator(
                self.config,
                translations_A=[0.1],
                rotations_deg=[2.0],
            ).generate(pose, LigandMapMetrics({}, 0.0, 0.0, 0.0, []), {})
        )
        return candidates[: int(self.config.max_candidates_per_stage)]
