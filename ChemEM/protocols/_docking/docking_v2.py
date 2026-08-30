# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>
"""Experimental, sampling-efficient docking engine (v2).

Selected on the CLI with ``--dock2``. Orchestration is inherited verbatim from the
baseline :class:`~ChemEM.protocols._docking.docking.Docking`; the ONLY difference is
that every compiled-extension call (ACO search, ECHO rescore, weight construction) is
routed to the isolated ``docking_v2`` module instead of ``docking`` — see
:meth:`Docking._echo_ext`.

Isolation guarantee: this never imports or rebuilds the baseline ``docking`` module,
and the baseline ``--dock`` path never imports ``docking_v2``. The two engines share no
compiled code, so ``python -m ChemEM <conf> --dock`` is unaffected by anything here.
The experimental sampling improvements (numerical-gradient L-BFGS local search,
cluster-then-refine gate, smoother coarse clash) live in ``ChemEM/cpp/docking_v2``.
"""
from __future__ import annotations

from .docking import Docking


class DockingV2(Docking):
    """Docking driven by the isolated ``docking_v2`` extension (see module docstring)."""

    ENGINE_NAME = "docking_v2"

    def _echo_ext(self):
        # Route ALL ext calls (search / rescore / weights) to the v2 module. Because the
        # base class funnels every compiled-extension access through _echo_ext(), this
        # single override makes DockingV2 use docking_v2 exclusively — the baseline
        # `docking` module is never imported in a --dock2 process.
        from ChemEM import docking_v2
        return docking_v2

    def _warn_unsupported_engine_features(self):
        """Flag the features this engine silently drops.

        `cpp/docking_v2` contains no covalent handling at all — the five `covalent_*`
        attributes that `_apply_covalent_anchor` sets on the precompute object are
        ingested by `cpp/docking_refactor/PreComputedData.cpp` and simply not read here.
        A covalent ligand therefore docks as if it were non-covalent, with no tether to
        the protein anchor, and the resulting poses look perfectly reasonable. Say so.
        """
        covalent = [i for i, lig in enumerate(getattr(self.system, "ligand", []) or [])
                    if getattr(lig, "covalent_link", None) is not None]
        if covalent:
            self.system.log(
                f"[dock2] WARNING: ligand(s) {covalent} carry a covalent link, but the "
                "docking_v2 engine has no covalent support — the warhead will NOT be "
                "tethered to the protein anchor and these poses are effectively "
                "non-covalent. Use --dock for covalent ligands.")

        # --return-n is not ingested by docking_v2's config block, so topN keeps its
        # compiled default of 20 and the flag does nothing.
        return_n = getattr(self.system.options, "return_n", None)
        if return_n is not None and int(return_n) != 20:
            self.system.log(
                f"[dock2] WARNING: --return-n {return_n} is ignored by this engine "
                "(topN is compiled at 20); the run will return at most 20 poses.")
