#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""The ``--score`` protocol: one place to score already-placed ligand poses.

Replaces the four things that used to do this separately -- ``--rescore-poses``
(ECHO), ``--mapq-score`` (Q-score), and the unregistered ``score_poses`` /
``echo_terms`` scripts.

``score_poses.ScorePoses`` is deliberately not imported here: ``ChemEM.protocol_spec``
imports ``.cli`` while building the parser, and pulling the protocol in at that point
would drag rdkit and the compiled engines into ``--help``.
"""
