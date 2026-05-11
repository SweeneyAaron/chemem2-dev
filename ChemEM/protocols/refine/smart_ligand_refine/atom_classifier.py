from __future__ import annotations

from typing import Dict, Optional

from .types import (
    AtomClass,
    AtomMapMetrics,
    LigandMapMetrics,
    SmartLigandRefinementConfig,
)


def classify_atom(
    metric: AtomMapMetrics,
    config: Optional[SmartLigandRefinementConfig] = None,
) -> AtomClass:
    config = config or SmartLigandRefinementConfig()
    q = float(metric.q_score)
    ccc = float(metric.local_ccc)

    half_ok = (
        metric.halfmap_agreement is None
        or float(metric.halfmap_agreement) >= float(config.min_halfmap_agreement)
    )

    density_supported = (
        float(metric.map_value) >= float(config.min_density_value)
        or (
            metric.positive_diff_density is not None
            and float(metric.positive_diff_density)
            >= float(config.min_positive_diff_density)
        )
    )

    if (
        q >= float(config.anchor_q_min)
        and ccc >= float(config.anchor_local_ccc_min)
        and half_ok
    ):
        return AtomClass.ANCHOR

    if q <= float(config.repair_q_max) and density_supported and half_ok:
        return AtomClass.REPAIR

    if q <= float(config.repair_q_max) and not density_supported:
        return AtomClass.WEAK_OR_ABSENT

    return AtomClass.UNCERTAIN


class AtomClassifier:
    def __init__(self, config: Optional[SmartLigandRefinementConfig] = None):
        self.config = config or SmartLigandRefinementConfig()

    def classify(self, metrics: LigandMapMetrics) -> Dict[int, AtomClass]:
        return {
            int(atom_idx): classify_atom(metric, self.config)
            for atom_idx, metric in metrics.atom_metrics.items()
        }

