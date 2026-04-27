# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Per-gate ranking and selection.

Gate 1: Q-score only — keep top-K1 per site.
Gate 2: composite of Q-score + MMGBSA — keep top-K2 per site.
Gate 3: per-site winner of post-search-refine candidates.

Sign convention: higher Q-score is better; lower (more negative) MMGBSA
deltaG is better. The composite is built so that higher = better.
Capacity-aware: the same ligand may be assigned to multiple sites.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .state import FinalAssignment, PoseCandidate


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


def composite(
    candidates: List[PoseCandidate],
    w_qscore: float,
    w_mmgbsa: float,
) -> np.ndarray:
    """composite_i = w_q * z(qscore_i) + w_m * z(-mmgbsa_i).

    MMGBSA is sign-flipped because lower deltaG = stronger binding = better.
    Candidates with missing metrics get 0 contribution from that channel.
    """
    if not candidates:
        return np.zeros((0,), dtype=float)
    q = np.array([np.nan if c.qscore is None else c.qscore for c in candidates])
    m = np.array([np.nan if c.mmgbsa is None else c.mmgbsa for c in candidates])
    qz = _zscore(q)
    mz = _zscore(-m)  # sign-flip
    return w_qscore * qz + w_mmgbsa * mz


def _stable_sort_key(c: PoseCandidate, score: float) -> tuple:
    """Higher score wins; ties broken by tighter mmgbsa, then ligand_idx, pose_idx."""
    mm = c.mmgbsa if c.mmgbsa is not None else float("inf")
    return (-score, mm, int(c.ligand_idx), int(c.pose_idx))


def gate1_select(
    candidates_by_site: Dict[str, List[PoseCandidate]],
    top_k: int,
) -> Dict[str, List[PoseCandidate]]:
    """Per site, keep the top_k poses by qscore (descending).

    Candidates with qscore=None are dropped (out of map / no heavy atoms).
    """
    out: Dict[str, List[PoseCandidate]] = {}
    for site_id, candidates in candidates_by_site.items():
        scored = [c for c in candidates if c.qscore is not None]
        scored.sort(key=lambda c: _stable_sort_key(c, c.qscore))
        out[site_id] = scored[: max(0, int(top_k))]
    return out


def gate2_select(
    candidates_by_site: Dict[str, List[PoseCandidate]],
    top_k: int,
    w_qscore: float,
    w_mmgbsa: float,
) -> Dict[str, List[PoseCandidate]]:
    """Per site, rank by composite and keep top_k."""
    out: Dict[str, List[PoseCandidate]] = {}
    for site_id, candidates in candidates_by_site.items():
        if not candidates:
            out[site_id] = []
            continue
        scores = composite(candidates, w_qscore, w_mmgbsa)
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
) -> List[FinalAssignment]:
    """Per site, pick the single best candidate.

    Capacity-aware: same ligand may end up in multiple sites' winners.
    Distinct chain IDs are assigned per copy of a ligand across sites
    (A, B, C, ...) in deterministic site_id order.
    """
    assignments: List[FinalAssignment] = []
    chain_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    used_per_ligand: Dict[int, int] = {}

    for site_id in sorted(candidates_by_site.keys(), key=str):
        candidates = candidates_by_site[site_id]
        if not candidates:
            continue
        scores = composite(candidates, w_qscore, w_mmgbsa)
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
            )
        )
    return assignments
