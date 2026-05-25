# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Per-gate ranking and selection.

Default orchestrator scoring is absolute coverage-first map fit. Legacy
coverage and qscore modes keep the old z-score/Q-score behavior.

Gate 1: keep top-K1 per site.
Gate 2: keep top-K2 per site after cheap refinement.
Gate 3: per-site winner of final-refined candidates.

Sign convention: higher Q-score is better; lower (more negative) MMGBSA
deltaG is better. The composite is built so that higher = better.
Capacity-aware: the same ligand may be assigned to multiple sites.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .state import AssignmentRejection, FinalAssignment, PoseCandidate


ABSOLUTE_WEIGHTS = {
    "density_coverage": 5.0,
    "density_overlap": 1.0,
    "density_ccc": 1.0,
    "qscore": 0.5,
    "q_low_tail": 0.25,
}

DEFAULT_MIN_ASSIGNMENT_SCORE = 3.25
DEFAULT_MIN_DENSITY_COVERAGE = 0.30
DEFAULT_MIN_ASSIGNMENT_MARGIN = 0.15
DEFAULT_SAME_LIGAND_MMGSA_WINDOW = 0.15


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    mu = float(arr[finite].mean())
    sd = float(arr[finite].std())
    if sd < 1e-12:
        return np.zeros_like(arr)
    out = (arr - mu) / sd
    out[~finite] = 0.0
    return out


def _metric_array(candidates: List[PoseCandidate], key: str) -> np.ndarray:
    values = []
    for c in candidates:
        value = None
        if key == "qscore":
            value = c.qscore
        elif key == "mmgbsa":
            value = c.mmgbsa
        else:
            value = c.metrics.get(key)
        values.append(np.nan if value is None else value)
    return np.asarray(values, dtype=float)


def _finite_metric(candidate: PoseCandidate, key: str, default: float = 0.0) -> float:
    if key == "qscore":
        value = candidate.qscore
    elif key == "mmgbsa":
        value = candidate.mmgbsa
    else:
        value = candidate.metrics.get(key)
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _absolute_weights(
    *,
    w_qscore: float = 0.5,
    w_qtail: float = 0.25,
    w_density_coverage: float = 5.0,
    w_density_overlap: float = 1.0,
    w_density_ccc: float = 1.0,
) -> dict:
    return {
        "density_coverage": float(w_density_coverage),
        "density_overlap": float(w_density_overlap),
        "density_ccc": float(w_density_ccc),
        "qscore": float(w_qscore),
        "q_low_tail": float(w_qtail),
    }


def absolute_assignment_score(
    candidate: PoseCandidate,
    weights: Optional[dict] = None,
    *,
    update_candidate: bool = True,
) -> float:
    """Coverage-first raw-magnitude assignment score; higher is better."""
    weights = dict(ABSOLUTE_WEIGHTS if weights is None else weights)
    terms = {
        key: float(weight) * _finite_metric(candidate, key, 0.0)
        for key, weight in weights.items()
    }
    score = float(sum(terms.values()))
    if update_candidate:
        candidate.metrics["assignment_score"] = score
        candidate.metrics["assignment_score_terms"] = dict(terms)
        candidate.rank_score = score
    return score


def _absolute_scores(candidates: List[PoseCandidate], weights: Optional[dict]) -> np.ndarray:
    return np.asarray(
        [absolute_assignment_score(c, weights, update_candidate=True) for c in candidates],
        dtype=float,
    )


def _absolute_sort_key(candidate: PoseCandidate, score: Optional[float] = None) -> tuple:
    if score is None:
        score = candidate.rank_score
    try:
        score = float(score)
    except Exception:
        score = float("-inf")
    if not np.isfinite(score):
        score = float("-inf")
    return (
        -score,
        -_finite_metric(candidate, "density_coverage"),
        -_finite_metric(candidate, "density_ccc"),
        -_finite_metric(candidate, "density_overlap"),
        -_finite_metric(candidate, "q_low_tail"),
        int(candidate.ligand_idx),
        int(candidate.pose_idx),
    )


def _same_ligand_pose_key(candidate: PoseCandidate) -> tuple:
    mm = candidate.mmgbsa if candidate.mmgbsa is not None else float("inf")
    try:
        mm = float(mm)
    except Exception:
        mm = float("inf")
    if not np.isfinite(mm):
        mm = float("inf")
    return (
        mm,
        *_absolute_sort_key(candidate),
    )


def _by_ligand(candidates: List[PoseCandidate]) -> Dict[int, List[PoseCandidate]]:
    grouped: Dict[int, List[PoseCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(int(candidate.ligand_idx), []).append(candidate)
    return grouped


def _best_pose_for_ligand(
    candidates: List[PoseCandidate],
    *,
    weights: Optional[dict],
    use_mmgbsa: bool,
    mmgbsa_window: float,
) -> Optional[PoseCandidate]:
    if not candidates:
        return None
    scores = _absolute_scores(candidates, weights)
    best_score = float(np.max(scores)) if scores.size else float("-inf")
    close = [
        c
        for c, score in zip(candidates, scores)
        if np.isfinite(score) and score >= best_score - float(mmgbsa_window)
    ]
    if use_mmgbsa:
        with_mmgbsa = [
            c
            for c in close
            if c.mmgbsa is not None and np.isfinite(float(c.mmgbsa))
        ]
        if with_mmgbsa:
            return sorted(with_mmgbsa, key=_same_ligand_pose_key)[0]
    return sorted(candidates, key=lambda c: _absolute_sort_key(c, c.rank_score))[0]


def _absolute_ligand_representatives(
    candidates: List[PoseCandidate],
    *,
    weights: Optional[dict],
    use_mmgbsa_within_ligand: bool,
    mmgbsa_window: float,
) -> List[PoseCandidate]:
    reps = []
    for grouped in _by_ligand(candidates).values():
        rep = _best_pose_for_ligand(
            grouped,
            weights=weights,
            use_mmgbsa=use_mmgbsa_within_ligand,
            mmgbsa_window=mmgbsa_window,
        )
        if rep is not None:
            reps.append(rep)
    return sorted(reps, key=lambda c: _absolute_sort_key(c, c.rank_score))


def composite(
    candidates: List[PoseCandidate],
    w_qscore: float,
    w_mmgbsa: float,
    *,
    score_mode: str = "qscore",
    w_qtail: float = 0.5,
    w_density_coverage: float = 2.0,
    w_density_precision: float = 0.5,
    w_density_ccc: float = 0.5,
    w_density_overlap: float = 1.0,
) -> np.ndarray:
    """Return per-candidate ranking scores; higher is better.

    MMGBSA is sign-flipped because lower deltaG = stronger binding = better.
    Coverage mode adds low-tail Q-score and density coverage/precision/CCC.
    Candidates with missing metrics get 0 contribution from that channel.
    """
    if not candidates:
        return np.zeros((0,), dtype=float)
    if str(score_mode).lower() == "absolute":
        weights = _absolute_weights(
            w_qscore=w_qscore,
            w_qtail=w_qtail,
            w_density_coverage=w_density_coverage,
            w_density_overlap=w_density_overlap,
            w_density_ccc=w_density_ccc,
        )
        return _absolute_scores(candidates, weights)

    q = _metric_array(candidates, "qscore")
    m = _metric_array(candidates, "mmgbsa")
    qz = _zscore(q)
    mz = _zscore(-m)  # sign-flip
    score = w_qscore * qz + w_mmgbsa * mz

    if str(score_mode).lower() == "coverage":
        q_tail = _metric_array(candidates, "q_low_tail")
        coverage = _metric_array(candidates, "density_coverage")
        precision = _metric_array(candidates, "density_precision")
        ccc = _metric_array(candidates, "density_ccc")
        score = (
            score
            + w_qtail * _zscore(q_tail)
            + w_density_coverage * _zscore(coverage)
            + w_density_precision * _zscore(precision)
            + w_density_ccc * _zscore(ccc)
        )

    return score


def _stable_sort_key(c: PoseCandidate, score: float) -> tuple:
    """Higher score wins; ties broken by tighter mmgbsa, then ligand_idx, pose_idx."""
    try:
        score = float(score)
    except Exception:
        score = float("-inf")
    if not np.isfinite(score):
        score = float("-inf")
    mm = c.mmgbsa if c.mmgbsa is not None else float("inf")
    return (-score, mm, int(c.ligand_idx), int(c.pose_idx))


def assign_rank_scores(
    candidates_by_site: Dict[str, List[PoseCandidate]],
    *,
    score_mode: str = "qscore",
    w_qscore: float = 1.0,
    w_mmgbsa: float = 0.5,
    w_qtail: float = 0.5,
    w_density_coverage: float = 2.0,
    w_density_precision: float = 0.5,
    w_density_ccc: float = 0.5,
    w_density_overlap: float = 1.0,
) -> Dict[str, np.ndarray]:
    scores_by_site: Dict[str, np.ndarray] = {}
    for site_id, candidates in candidates_by_site.items():
        mode = str(score_mode).lower()
        if mode == "absolute":
            scores = composite(
                candidates,
                w_qscore,
                0.0,
                score_mode="absolute",
                w_qtail=w_qtail,
                w_density_coverage=w_density_coverage,
                w_density_ccc=w_density_ccc,
                w_density_overlap=w_density_overlap,
            )
            scores_by_site[site_id] = scores
            continue
        if mode != "coverage" and float(w_mmgbsa) == 0.0:
            scores = _metric_array(candidates, "qscore")
            for c, score in zip(candidates, scores):
                c.rank_score = None if not np.isfinite(score) else float(score)
            scores_by_site[site_id] = scores
            continue
        scores = composite(
            candidates,
            w_qscore,
            w_mmgbsa,
            score_mode=score_mode,
            w_qtail=w_qtail,
            w_density_coverage=w_density_coverage,
            w_density_precision=w_density_precision,
            w_density_ccc=w_density_ccc,
            w_density_overlap=w_density_overlap,
        )
        for c, score in zip(candidates, scores):
            c.rank_score = float(score)
        scores_by_site[site_id] = scores
    return scores_by_site


def gate1_select(
    candidates_by_site: Dict[str, List[PoseCandidate]],
    top_k: int,
    *,
    score_mode: str = "qscore",
    w_qscore: float = 1.0,
    w_mmgbsa: float = 0.0,
    w_qtail: float = 0.5,
    w_density_coverage: float = 2.0,
    w_density_precision: float = 0.5,
    w_density_ccc: float = 0.5,
    w_density_overlap: float = 1.0,
) -> Dict[str, List[PoseCandidate]]:
    """Per site, keep the top_k poses by the requested score mode.

    Candidates with qscore=None are dropped (out of map / no heavy atoms).
    """
    _ = w_mmgbsa  # Gate 1 happens before MMGBSA, but accepts shared rank kwargs.
    out: Dict[str, List[PoseCandidate]] = {}
    for site_id, candidates in candidates_by_site.items():
        scored = [c for c in candidates if c.qscore is not None]
        mode = str(score_mode).lower()
        if mode == "absolute":
            weights = _absolute_weights(
                w_qscore=w_qscore,
                w_qtail=w_qtail,
                w_density_coverage=w_density_coverage,
                w_density_overlap=w_density_overlap,
                w_density_ccc=w_density_ccc,
            )
            _absolute_scores(scored, weights)
            selected = []
            selected_ids = set()
            reps = _absolute_ligand_representatives(
                scored,
                weights=weights,
                use_mmgbsa_within_ligand=False,
                mmgbsa_window=0.0,
            )
            for candidate in reps:
                if len(selected) >= max(0, int(top_k)):
                    break
                selected.append(candidate)
                selected_ids.add(id(candidate))
            if len(selected) < max(0, int(top_k)):
                for candidate in sorted(scored, key=lambda c: _absolute_sort_key(c, c.rank_score)):
                    if id(candidate) in selected_ids:
                        continue
                    selected.append(candidate)
                    selected_ids.add(id(candidate))
                    if len(selected) >= max(0, int(top_k)):
                        break
            out[site_id] = selected
        elif mode == "coverage":
            scores = composite(
                scored,
                w_qscore,
                0.0,
                score_mode="coverage",
                w_qtail=w_qtail,
                w_density_coverage=w_density_coverage,
                w_density_precision=w_density_precision,
                w_density_ccc=w_density_ccc,
                w_density_overlap=w_density_overlap,
            )
            for c, score in zip(scored, scores):
                c.rank_score = float(score)
            scored.sort(key=lambda c: _stable_sort_key(c, c.rank_score))
            out[site_id] = scored[: max(0, int(top_k))]
        else:
            for c in scored:
                c.rank_score = float(c.qscore)
            scored.sort(key=lambda c: _stable_sort_key(c, c.qscore))
            out[site_id] = scored[: max(0, int(top_k))]
    return out


def gate2_select(
    candidates_by_site: Dict[str, List[PoseCandidate]],
    top_k: int,
    w_qscore: float,
    w_mmgbsa: float,
    *,
    score_mode: str = "qscore",
    w_qtail: float = 0.5,
    w_density_coverage: float = 2.0,
    w_density_precision: float = 0.5,
    w_density_ccc: float = 0.5,
    w_density_overlap: float = 1.0,
    same_ligand_mmgbsa_window: float = DEFAULT_SAME_LIGAND_MMGSA_WINDOW,
) -> Dict[str, List[PoseCandidate]]:
    """Per site, rank by composite and keep top_k."""
    out: Dict[str, List[PoseCandidate]] = {}
    for site_id, candidates in candidates_by_site.items():
        if not candidates:
            out[site_id] = []
            continue
        if str(score_mode).lower() == "absolute":
            weights = _absolute_weights(
                w_qscore=w_qscore,
                w_qtail=w_qtail,
                w_density_coverage=w_density_coverage,
                w_density_overlap=w_density_overlap,
                w_density_ccc=w_density_ccc,
            )
            reps = _absolute_ligand_representatives(
                candidates,
                weights=weights,
                use_mmgbsa_within_ligand=True,
                mmgbsa_window=same_ligand_mmgbsa_window,
            )
            out[site_id] = reps[: max(0, int(top_k))]
            continue
        scores = composite(
            candidates,
            w_qscore,
            w_mmgbsa,
            score_mode=score_mode,
            w_qtail=w_qtail,
            w_density_coverage=w_density_coverage,
            w_density_precision=w_density_precision,
            w_density_ccc=w_density_ccc,
            w_density_overlap=w_density_overlap,
        )
        for c, score in zip(candidates, scores):
            c.rank_score = float(score)
        order = sorted(
            range(len(candidates)),
            key=lambda i: _stable_sort_key(candidates[i], float(scores[i])),
        )
        out[site_id] = [candidates[i] for i in order[: max(0, int(top_k))]]
    return out


def gate3_select(
    candidates_by_site: Dict[str, List[PoseCandidate]],
    w_qscore: float,
    w_mmgbsa: float,
    *,
    score_mode: str = "qscore",
    w_qtail: float = 0.5,
    w_density_coverage: float = 2.0,
    w_density_precision: float = 0.5,
    w_density_ccc: float = 0.5,
    w_density_overlap: float = 1.0,
    min_assignment_score: float = DEFAULT_MIN_ASSIGNMENT_SCORE,
    min_density_coverage: float = DEFAULT_MIN_DENSITY_COVERAGE,
    min_assignment_margin: float = DEFAULT_MIN_ASSIGNMENT_MARGIN,
    same_ligand_mmgbsa_window: float = DEFAULT_SAME_LIGAND_MMGSA_WINDOW,
    return_rejections: bool = False,
) -> List[FinalAssignment] | tuple[List[FinalAssignment], List[AssignmentRejection]]:
    """Per site, pick the single best candidate.

    Capacity-aware: same ligand may end up in multiple sites' winners.
    Distinct chain IDs are assigned per copy of a ligand across sites
    (A, B, C, ...) in deterministic site_id order.
    """
    assignments: List[FinalAssignment] = []
    rejections: List[AssignmentRejection] = []
    chain_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    used_per_ligand: Dict[int, int] = {}

    for site_id in sorted(candidates_by_site.keys(), key=str):
        candidates = candidates_by_site[site_id]
        if not candidates:
            continue
        if str(score_mode).lower() == "absolute":
            weights = _absolute_weights(
                w_qscore=w_qscore,
                w_qtail=w_qtail,
                w_density_coverage=w_density_coverage,
                w_density_overlap=w_density_overlap,
                w_density_ccc=w_density_ccc,
            )
            reps = _absolute_ligand_representatives(
                candidates,
                weights=weights,
                use_mmgbsa_within_ligand=True,
                mmgbsa_window=same_ligand_mmgbsa_window,
            )
            if not reps:
                continue
            winner = reps[0]
            winner_score = float(winner.rank_score)
            runner_score = float(reps[1].rank_score) if len(reps) > 1 else None
            margin = (
                None
                if runner_score is None
                else float(winner_score - runner_score)
            )
            winner.metrics["assignment_score"] = winner_score
            winner.metrics["assignment_margin"] = margin

            rejection_reason = None
            if winner_score < float(min_assignment_score):
                rejection_reason = "assignment_score_below_threshold"
            elif _finite_metric(winner, "density_coverage") < float(min_density_coverage):
                rejection_reason = "density_coverage_below_threshold"
            elif margin is not None and margin < float(min_assignment_margin):
                rejection_reason = "assignment_margin_below_threshold"

            if rejection_reason is not None:
                winner.metrics["rejection_reason"] = rejection_reason
                rejections.append(
                    AssignmentRejection(
                        site_id=str(site_id),
                        best_ligand_idx=int(winner.ligand_idx),
                        best_pose_idx=int(winner.pose_idx),
                        qscore=None if winner.qscore is None else float(winner.qscore),
                        mmgbsa=None if winner.mmgbsa is None else float(winner.mmgbsa),
                        assignment_score=winner_score,
                        assignment_margin=margin,
                        rejection_reason=rejection_reason,
                        stage=winner.stage,
                        metrics=dict(winner.metrics),
                        rank_score=winner.rank_score,
                    )
                )
                continue

            copy_idx = used_per_ligand.get(winner.ligand_idx, 0)
            used_per_ligand[winner.ligand_idx] = copy_idx + 1
            chain_id = chain_alphabet[copy_idx % len(chain_alphabet)]
            assignments.append(
                FinalAssignment(
                    site_id=str(site_id),
                    ligand_idx=int(winner.ligand_idx),
                    pose_idx=int(winner.pose_idx),
                    coords=np.asarray(winner.coords, dtype=float),
                    qscore=(
                        float(winner.qscore)
                        if winner.qscore is not None
                        else float("nan")
                    ),
                    mmgbsa=None if winner.mmgbsa is None else float(winner.mmgbsa),
                    composite=winner_score,
                    stage=winner.stage,
                    chain_id=chain_id,
                    metrics=dict(winner.metrics),
                    rank_score=winner.rank_score,
                )
            )
            continue

        scores = composite(
            candidates,
            w_qscore,
            w_mmgbsa,
            score_mode=score_mode,
            w_qtail=w_qtail,
            w_density_coverage=w_density_coverage,
            w_density_precision=w_density_precision,
            w_density_ccc=w_density_ccc,
            w_density_overlap=w_density_overlap,
        )
        for c, score in zip(candidates, scores):
            c.rank_score = float(score)
        best_i = min(
            range(len(candidates)),
            key=lambda i: _stable_sort_key(candidates[i], float(scores[i])),
        )
        winner = candidates[best_i]
        copy_idx = used_per_ligand.get(winner.ligand_idx, 0)
        used_per_ligand[winner.ligand_idx] = copy_idx + 1
        chain_id = chain_alphabet[copy_idx % len(chain_alphabet)]
        assignments.append(
            FinalAssignment(
                site_id=str(site_id),
                ligand_idx=int(winner.ligand_idx),
                pose_idx=int(winner.pose_idx),
                coords=np.asarray(winner.coords, dtype=float),
                qscore=float(winner.qscore) if winner.qscore is not None else float("nan"),
                mmgbsa=None if winner.mmgbsa is None else float(winner.mmgbsa),
                composite=float(scores[best_i]),
                stage=winner.stage,
                chain_id=chain_id,
                metrics=dict(winner.metrics),
                rank_score=winner.rank_score,
            )
        )
    if return_rejections:
        return assignments, rejections
    return assignments


def assignment_evaluation_rows(
    assignments: List[FinalAssignment],
    rejections: List[AssignmentRejection],
    expected_assignments: Optional[Dict[str, int]] = None,
) -> tuple[List[dict], dict]:
    """Build labelled assignment-evaluation rows and confusion counts."""
    expected = {
        str(site_id): int(ligand_idx)
        for site_id, ligand_idx in (expected_assignments or {}).items()
    }
    accepted_by_site = {str(a.site_id): a for a in assignments}
    rejected_by_site = {str(r.site_id): r for r in rejections}
    site_ids = sorted(
        set(expected) | set(accepted_by_site) | set(rejected_by_site),
        key=str,
    )
    counts = {
        "true_positive": 0,
        "wrong_ligand": 0,
        "false_positive_site": 0,
        "false_negative_site": 0,
        "true_negative_rejected": 0,
        "unlabeled_accepted": 0,
        "unlabeled_rejected": 0,
    }
    rows: List[dict] = []
    for site_id in site_ids:
        expected_ligand = expected.get(site_id)
        assignment = accepted_by_site.get(site_id)
        rejection = rejected_by_site.get(site_id)
        accepted = assignment is not None
        if accepted:
            actual_ligand = int(assignment.ligand_idx)
            actual_pose = int(assignment.pose_idx)
            score = assignment.rank_score
            margin = assignment.metrics.get("assignment_margin")
            reason = None
            metrics = assignment.metrics
            qscore = assignment.qscore
            mmgbsa = assignment.mmgbsa
        elif rejection is not None:
            actual_ligand = rejection.best_ligand_idx
            actual_pose = rejection.best_pose_idx
            score = rejection.assignment_score
            margin = rejection.assignment_margin
            reason = rejection.rejection_reason
            metrics = rejection.metrics
            qscore = rejection.qscore
            mmgbsa = rejection.mmgbsa
        else:
            actual_ligand = None
            actual_pose = None
            score = None
            margin = None
            reason = "no_candidate"
            metrics = {}
            qscore = None
            mmgbsa = None

        if expected_ligand is None and accepted:
            label = "false_positive_site" if expected else "unlabeled_accepted"
        elif expected_ligand is None and not accepted:
            label = "true_negative_rejected" if expected else "unlabeled_rejected"
        elif accepted and actual_ligand == expected_ligand:
            label = "true_positive"
        elif accepted:
            label = "wrong_ligand"
        else:
            label = "false_negative_site"
        counts[label] = int(counts.get(label, 0)) + 1

        rows.append(
            {
                "site_id": site_id,
                "expected_ligand_idx": expected_ligand,
                "actual_ligand_idx": actual_ligand,
                "actual_pose_idx": actual_pose,
                "accepted": bool(accepted),
                "label": label,
                "assignment_score": score,
                "assignment_margin": margin,
                "rejection_reason": reason,
                "qscore": qscore,
                "density_coverage": metrics.get("density_coverage"),
                "density_overlap": metrics.get("density_overlap"),
                "density_ccc": metrics.get("density_ccc"),
                "density_precision": metrics.get("density_precision"),
                "density_mi": metrics.get("density_mi"),
                "density_normalized_mi": metrics.get("density_normalized_mi"),
                "density_mi_nbins": metrics.get("density_mi_nbins"),
                "density_sci": metrics.get("density_sci"),
                "shape_zernike_similarity": metrics.get("shape_zernike_similarity"),
                "shape_zernike_distance": metrics.get("shape_zernike_distance"),
                "shape_spharm_similarity": metrics.get("shape_spharm_similarity"),
                "shape_spharm_distance": metrics.get("shape_spharm_distance"),
                "shape_radial_profile_similarity": metrics.get(
                    "shape_radial_profile_similarity"
                ),
                "shape_volume_ratio": metrics.get("shape_volume_ratio"),
                "shape_exp_component_count": metrics.get("shape_exp_component_count"),
                "shape_selected_component_fraction": metrics.get(
                    "shape_selected_component_fraction"
                ),
                "shape_selected_component_mode": metrics.get(
                    "shape_selected_component_mode"
                ),
                "shape_zernike_backend": metrics.get("shape_zernike_backend"),
                "density_skeleton_precision": metrics.get(
                    "density_skeleton_precision"
                ),
                "density_skeleton_recall": metrics.get("density_skeleton_recall"),
                "density_skeleton_f1": metrics.get("density_skeleton_f1"),
                "density_skeleton_mean_distance_A": metrics.get(
                    "density_skeleton_mean_distance_A"
                ),
                "density_skeleton_unmatched_density_length_A": metrics.get(
                    "density_skeleton_unmatched_density_length_A"
                ),
                "density_skeleton_unmatched_ligand_length_A": metrics.get(
                    "density_skeleton_unmatched_ligand_length_A"
                ),
                "density_skeleton_exp_branch_count": metrics.get(
                    "density_skeleton_exp_branch_count"
                ),
                "density_skeleton_sim_branch_count": metrics.get(
                    "density_skeleton_sim_branch_count"
                ),
                "density_skeleton_branch_count_diff": metrics.get(
                    "density_skeleton_branch_count_diff"
                ),
                "density_skeleton_exp_point_count": metrics.get(
                    "density_skeleton_exp_point_count"
                ),
                "density_skeleton_sim_point_count": metrics.get(
                    "density_skeleton_sim_point_count"
                ),
                "density_skeleton_component_mode": metrics.get(
                    "density_skeleton_component_mode"
                ),
                "density_skeleton_backend": metrics.get("density_skeleton_backend"),
                "mmgbsa": mmgbsa,
            }
        )

    summary = {
        "expected_assignments": dict(expected),
        "counts": counts,
        "n_expected": len(expected),
        "n_accepted": len(assignments),
        "n_rejected": len(rejections),
    }
    return rows, summary
