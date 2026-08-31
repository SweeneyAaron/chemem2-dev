#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Density-fit metrics for a pose (``--score --score-with density``).

Coverage, precision, CCC, overlap, envelope IoU, mutual information, and optionally
SCI and the shape/skeleton descriptors.

The region these are computed against matters more than any of the parameters:

  * ``full`` (default) -- the whole map. Every term is well defined and the coverage
    denominator is a per-case constant, so poses are comparable across a case.
  * ``box``  -- a fixed-size cube around each pose centroid. Coverage is normalised
    per pose and it is faster, at the cost of cross-pose comparability.
  * ``site`` -- the segmented binding-site map, which is what ``--orchestrate``
    scores against. That map has been multiplied by a boundary EDT, so its
    amplitudes are on a completely different scale from the other two regions.
    Requires segmentation to have run, which is why this is the one region that
    pulls in the binding_site/alpha_mask/confidence_map chain.
"""

from __future__ import annotations

import numpy as np

from ChemEM.protocols.orchestrator.scoring import density_fit_metrics

from ..sites import site_maps_for
from .base import PoseScorer

SEGMENTATION = ("binding_site", "alpha_mask", "confidence_map")

# Nested dicts / per-feature lists that do not belong in a flat numeric CSV.
JSON_ONLY = ("density_feature_metrics", "density_sci_terms")


class DensityScorer(PoseScorer):
    NAME = "density"
    HELP = "Density fit: coverage, precision, CCC, MI, SCI, shape"
    DEPS = ()
    HEADLINE = "density_overlap"
    HIGHER_IS_BETTER = True

    COLUMNS = (
        "density_region", "density_coverage", "density_precision", "density_ccc",
        "density_overlap", "density_envelope_iou", "density_excess_fraction",
        "density_mi", "density_normalized_mi", "density_sci",
        "density_feature_score", "density_threshold", "density_total_weight",
        "selected_feature_idx", "ligand_heavy_atom_count",
    )

    @classmethod
    def deps_for(cls, opts):
        # Only the segmented-site region needs segmentation; full and box read the
        # map the config already loaded.
        if getattr(opts, "score_density_region", "full") == "site":
            return SEGMENTATION
        return ()

    def needs_site(self) -> bool:
        return getattr(self.opts, "score_density_region", "full") == "site"

    def setup_run(self, ctx) -> None:
        self.region = str(self._opt("score_density_region", "full"))
        self.box_size = float(self._opt("score_density_box_size", 24.0))
        self.threshold_frac = float(self._opt("score_density_threshold_frac", 0.05))
        self.compute_sci = not bool(self._opt("score_density_no_sci", False))
        self.compute_shape = not bool(self._opt("score_density_no_shape", False))

        if self.region != "site" and getattr(self.system, "density_map", None) is None:
            raise ValueError(
                f"--score-density-region {self.region} requires a density map. Set "
                "`densmap =` in the config, and do not pass --no-map."
            )
        self.system.log(
            f"[score:density] region={self.region} "
            f"threshold_frac={self.threshold_frac} sci={self.compute_sci} "
            f"shape={self.compute_shape}"
            + (f" box_size={self.box_size}" if self.region == "box" else "")
        )
        if self.region == "site":
            self.system.log(
                "[score:density] NOTE: the segmented site map is rescaled by a "
                "boundary EDT, so these numbers are not on the same scale as "
                "region full/box."
            )

    def _region_map(self, pose, row):
        """The map object the density terms are scored against for this pose."""
        if self.region == "site":
            site_id, _bs = pose.site()
            maps = site_maps_for(self.system, site_id)
            if maps is None:
                row["density_failed"] = "no_site_map"
            return maps

        full = getattr(self.system, "density_map", None)
        if full is None:
            row["density_failed"] = "no_density_map"
            return None
        if self.region == "full":
            return full

        # box: a fixed-size cube cropped from the full map, centred on the pose.
        from ChemEM.parsers.EMMap import EMMap
        from ChemEM.tools.density import crop_map_around_point

        heavy = np.asarray(pose.coords, dtype=float)
        centroid = heavy.mean(axis=0)
        apix = np.asarray(full.apix, dtype=float).reshape(-1)
        if apix.size == 1:
            apix = np.repeat(apix, 3)
        try:
            sub, new_origin = crop_map_around_point(
                np.asarray(full.density_map, dtype=float),
                np.asarray(full.origin, dtype=float),
                apix,
                centroid,
                self.box_size,
            )
        except Exception as exc:  # centre off-map, degenerate crop, ...
            row["density_failed"] = f"box_empty:{type(exc).__name__}"
            return None
        if sub.size == 0:
            row["density_failed"] = "box_empty"
            return None
        row["box_size"] = self.box_size
        if float(np.max(np.abs(heavy - centroid))) > self.box_size / 2.0:
            # Some atoms fall outside the box, so they are scored against no density.
            row["ligand_exceeds_box"] = 1
        return EMMap(tuple(float(x) for x in new_origin),
                     tuple(float(a) for a in apix), sub,
                     getattr(full, "resolution", 3.0))

    def score(self, pose, row) -> None:
        row["density_region"] = self.region
        region_map = self._region_map(pose, row)
        if region_map is None:
            return

        metrics = density_fit_metrics(
            pose.coords,
            pose.mol,
            region_map,
            threshold_frac=self.threshold_frac,
            compute_sci=self.compute_sci,
            compute_shape=self.compute_shape,
        )
        if metrics is None:
            row["density_failed"] = "no_metrics"
            return

        for key, value in metrics.items():
            if key in JSON_ONLY:
                row.setdefault("_json", {})[key] = value
            elif isinstance(value, (list, tuple)):
                row[key] = "x".join(str(v) for v in value)
            else:
                row[key] = value

    def finish_run(self, ctx, rows) -> dict:
        return {
            "region": self.region,
            "box_size": self.box_size if self.region == "box" else None,
            "threshold_frac": self.threshold_frac,
            "sci": self.compute_sci,
            "shape": self.compute_shape,
        }
