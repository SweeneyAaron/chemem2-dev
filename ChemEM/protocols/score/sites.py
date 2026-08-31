#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Binding-site selection for pose scoring.

Docking picks a site and puts a pose in it. Scoring gets the pose already placed and
has to work out backwards which site it belongs to -- that is what ``site_for_pose``
does, and it is the only thing here that has no counterpart in the docking protocol.
"""

from __future__ import annotations

import numpy as np


def iter_sites(system):
    """The sites scoring should consider, mirroring ``Docking._iter_sites``.

    Kept in step with that method deliberately: a pose's ECHO total is only the
    number ``--dock`` optimised if it is scored against the same site the docking
    run would have used.
    """
    binding_sites = getattr(system, "binding_sites", None) or {}
    opts = system.options

    no_map = getattr(opts, "no_map", False)
    if no_map or getattr(system, "density_map", None) is None:
        return list(binding_sites.items())

    # --dock-full-map builds its map on demand from the BindingSite geometry, and
    # --manual-site is a user-defined volume, so in both cases a site that
    # segmentation dropped is still valid and must not be filtered out here.
    if (getattr(opts, "dock_full_map", False)
            or getattr(opts, "manual_site", False)):
        return list(binding_sites.items())

    site_maps = getattr(system, "binding_site_maps", None) or {}
    return [
        (key, binding_site)
        for key, binding_site in binding_sites.items()
        if binding_site.key in site_maps
    ]


def site_for_pose(sites, coords):
    """Pick the site a pose belongs to.

    Poses arrive with fixed coordinates, so unlike docking we have to work out which
    site they sit in: prefer a site whose bounding box contains the pose centroid,
    else fall back to the nearest site centroid.
    """
    if not sites:
        return None, None
    if len(sites) == 1:
        return sites[0]

    centroid = np.asarray(coords, dtype=float).mean(axis=0)

    for site_id, binding_site in sites:
        lo = np.asarray(binding_site.min_coords, dtype=float)
        hi = np.asarray(binding_site.max_coords, dtype=float)
        if np.all(centroid >= lo) and np.all(centroid <= hi):
            return site_id, binding_site

    def distance(entry):
        site_centroid = np.asarray(entry[1].binding_site_centroid, dtype=float)
        return float(np.linalg.norm(centroid - site_centroid))

    return min(sites, key=distance)


def site_maps_for(system, site_id):
    """The segmented density blobs for one site, tolerating str/int key mismatch.

    ``binding_site_maps`` is keyed by whatever type the segmentation protocol used,
    while site ids reach us as strings once they have been through a CSV column.
    """
    maps = getattr(system, "binding_site_maps", None) or {}
    if site_id in maps:
        return maps[site_id]
    try:
        int_id = int(site_id)
    except (TypeError, ValueError):
        int_id = None
    if int_id is not None and int_id in maps:
        return maps[int_id]
    for key, value in maps.items():
        if str(key) == str(site_id):
            return value
    return None
