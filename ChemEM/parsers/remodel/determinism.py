# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Deterministic settings for protein preparation.

Protein prep used to produce different coordinates in every process, which put a
floor of ~1.9 ECHO score units under every result -- enough to swamp the gap
between competing docked poses. Two independent causes, both inside the OpenMM
minimisations run during prep:

  * the ``Context`` was built with ``platform=None``, so OpenMM auto-selected the
    fastest platform (OpenCL here), which is single-precision and explicitly
    allowed to be non-reproducible;
  * ``PDBFixer.addMissingAtoms`` used a ``LangevinIntegrator`` whose seed
    defaulted to 0, which OpenMM documents as "a unique seed is chosen when a
    Context is created". When a rebuilt heavy atom lands within 1.3 A of a
    neighbour, PDBFixer runs up to 2000 fs of 300 K Langevin dynamics with that
    seed -- which is what moved atoms up to 9.8 A between runs. Minimisation
    alone could not.

Seeding Python's ``random`` / ``numpy.random`` does nothing here: the RNG lives in
OpenMM's C++ kernel.

CPU, not Reference, is the default platform: both are deterministic, but Reference
is ~10x slower (platform speeds are Reference 1, CPU 10, OpenCL 50) and prep runs
over the whole receptor.

Scope of the guarantee: repeated runs *on one machine* agree. Identical
coordinates across machines, OpenMM builds or pdbfixer versions are NOT promised
-- CPU SIMD dispatch differs. Use ``Reference`` if cross-machine identity matters.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass

# Bump whenever the prep pipeline changes shape in a way that alters output but
# is not captured by any other cache-key field (e.g. adding a remodelling step).
#   1 -> 2: bounded clash relief + GB-free hydrogen placement (both move atoms).
PREP_SCHEMA_VERSION = 2

# MUST stay non-zero: OpenMM reads a seed of 0 as "choose a fresh random seed",
# which silently reverts the whole fix. Guarded in PrepOptions below.
DEFAULT_PREP_SEED = 1234567

DEFAULT_PREP_PLATFORM = "CPU"

# Properties that make a platform reproducible. Only CPU exposes them; Reference
# is deterministic already and OpenCL cannot be made so.
_DETERMINISTIC_PROPERTIES = (("Threads", "1"), ("DeterministicForces", "true"))

# PDBFixer relieves clashes among rebuilt atoms with up to 10 rounds of 200 x 5 fs
# Langevin dynamics, keeping the best snapshot and stopping early if the closest
# contact ever reaches 1.3 A. On a heavily-repaired receptor it never gets there,
# and it dominates preparation cost: 362 s of a 492 s prep on 9e26.
#
# It is tempting to cap it, and that IS a large win -- but it is NOT safe as a
# default, because the metric is non-monotonic and the useful snapshot lands at a
# structure-dependent iteration. Closest contact vs steps on 9e26, in Angstrom:
#
#   steps:     0     200    400    600    800   1000  1200  1400  1600  1800  2000
#   nearest: 0.821  0.564  0.188  0.952  0.574 0.712 0.733 0.756 0.855 0.893 0.890
#
# It gets *worse* before better and peaks at 600. Capping there reproduces the
# uncapped structure exactly on 9e26 -- but on 7bxu the same 600-step cap leaves a
# 0.655 A worst contact against 1.052 A uncapped, plus ~39 more sub-2 A contacts.
# A "stop when it stops improving" rule fails too: on 9e26 the peak arrives only
# after two consecutive worsening rounds.
#
# So the default keeps PDBFixer's behaviour and the cap is opt-in. If you set it,
# check the resulting contacts -- see the README. The unbounded
# LocalEnergyMinimizer after the loop always runs either way.
DEFAULT_CLASH_RELIEF_STEPS = None


@dataclass(frozen=True, slots=True)
class PrepOptions:
    """How to prepare a protein. `deterministic=False` restores the old behaviour."""

    platform: str = DEFAULT_PREP_PLATFORM
    threads: int = 1
    seed: int = DEFAULT_PREP_SEED
    pH: float = 7.4
    deterministic: bool = True
    # Langevin step budget for PDBFixer's clash relief; 0 disables the dynamics
    # and leaves only the minimisations, None restores PDBFixer's own 2000.
    clash_relief_steps: int = DEFAULT_CLASH_RELIEF_STEPS
    # Keep implicit solvent in the force field used for hydrogen placement.
    #
    # Dropping GBn2 here is tempting: it is a CustomGBForce over every atom with no
    # interaction group, minimised 50x, and it costs 105 s of a 236 s preparation
    # on 9e26. But it is NOT score-neutral. Measured: dropping it leaves hbond_raw
    # and every non-electrostatic term bit-identical, yet moves echo_total by up to
    # 0.63 units -- because the ECHO electrostatic grid is built with
    # `collapse_hydrogens=False`, i.e. from per-atom charges including hydrogens,
    # so any change in H positions rewrites it.
    #
    # 0.63 units is comparable to the run-to-run noise this whole effort removed,
    # so it would invalidate the fitted default_v1 weights. Default therefore keeps
    # GB; `--no-prep-h-implicit` opts into the faster, score-shifting path.
    h_placement_implicit: bool = True

    def __post_init__(self):
        if self.deterministic and int(self.seed) == 0:
            raise ValueError(
                "protein prep seed must be non-zero: OpenMM treats seed 0 as "
                "'choose a fresh random seed per Context', which silently makes "
                "preparation non-reproducible. Pass a non-zero seed, or set "
                "deterministic=False if you want the old behaviour."
            )

    def key_fields(self):
        """The parts that change the prepared coordinates, for a cache key."""
        return {
            "platform": self.platform,
            "threads": int(self.threads),
            "seed": int(self.seed),
            "pH": float(self.pH),
            "deterministic": bool(self.deterministic),
            "clash_relief_steps": (None if self.clash_relief_steps is None
                                   else int(self.clash_relief_steps)),
            "h_placement_implicit": bool(self.h_placement_implicit),
        }


def check_pdbfixer_support():
    """Fail loudly if the installed pdbfixer cannot be made deterministic.

    Both hooks landed well before pdbfixer 1.7, but silently producing
    irreproducible output because a hook is missing is the one outcome worth
    an explicit error.
    """
    from pdbfixer import PDBFixer

    missing = []
    if "platform" not in inspect.signature(PDBFixer.__init__).parameters:
        missing.append("PDBFixer(platform=...)")
    if "seed" not in inspect.signature(PDBFixer.addMissingAtoms).parameters:
        missing.append("PDBFixer.addMissingAtoms(seed=...)")
    if missing:
        raise RuntimeError(
            "Deterministic protein preparation needs pdbfixer >= 1.7; this "
            f"install is missing {', '.join(missing)}. Upgrade pdbfixer, or run "
            "with --no-deterministic-prep to accept irreproducible coordinates."
        )


@contextmanager
def bounded_clash_relief(max_steps):
    """Cap the Langevin dynamics PDBFixer runs to relieve clashes among rebuilt atoms.

    `PDBFixer.addMissingAtoms` ends with, in effect::

        if nearest < 0.13:                    # 1.3 A
            for i in range(10):
                context.setParameter('C', 0.15*(i+1))
                integrator.step(200)          # 200 x 5 fs
                ...
                if nearest >= 0.13: break
            context.setState(best)
            LocalEnergyMinimizer.minimize(context)

    On a heavily-repaired receptor that loop routinely exhausts its budget without
    ever reaching 1.3 A. Measured on 9e26: all 2000 steps, 362 s of a 492 s prep,
    and it still exits by running out of iterations rather than by succeeding.

    Capping the step budget keeps the early rounds, which do most of the useful
    work, and drops the tail that is not converging. The loop still keeps its best
    snapshot, and the unbounded `LocalEnergyMinimizer.minimize` *after* the loop
    still runs, so preparation always ends on a fully minimised structure.

    Implemented by wrapping `LangevinIntegrator.step` rather than by calling
    PDBFixer's private `_addAtomsToTopology` / `_createForceField` ourselves: those
    are internals whose signatures move between releases, while `step` is public
    and stable. The patch is process-global for its duration, so it also catches
    any other Langevin stepping in the same window -- preparation is
    single-threaded and does none, but keep the scope tight.

    `max_steps=None` leaves PDBFixer's own budget alone; 0 skips the dynamics.
    """
    if max_steps is None:
        yield
        return

    try:
        from openmm import LangevinIntegrator
    except Exception:
        yield
        return

    original = LangevinIntegrator.step
    remaining = {"steps": max(0, int(max_steps))}

    def capped(self, steps):
        allowed = min(int(steps), remaining["steps"])
        remaining["steps"] -= allowed
        if allowed > 0:
            return original(self, allowed)
        return None

    LangevinIntegrator.step = capped
    try:
        yield
    finally:
        LangevinIntegrator.step = original


@contextmanager
def prep_platform(opts: PrepOptions):
    """Yield an OpenMM Platform pinned for reproducible preparation.

    Platform property defaults are **process-global**, so setting Threads=1 here
    without restoring it would silently serialise every docking and refinement
    Context created later in the run. The previous values are therefore saved and
    restored on exit, including on exception.

    Yields None when determinism is off or the platform is unavailable, which
    makes callers fall back to OpenMM's auto-selection.
    """
    if not opts.deterministic or not opts.platform or opts.platform == "inherit":
        yield None
        return

    check_pdbfixer_support()

    try:
        from openmm import Platform
        platform = Platform.getPlatformByName(str(opts.platform))
    except Exception:
        print(
            f"[prep] WARNING: OpenMM platform {opts.platform!r} is unavailable; "
            "falling back to auto-selection. Preparation will NOT be reproducible."
        )
        yield None
        return

    available = set(platform.getPropertyNames())
    wanted = [(name, value) for name, value in _DETERMINISTIC_PROPERTIES if name in available]
    if "Threads" in available:
        wanted = [("Threads", str(int(opts.threads))) if n == "Threads" else (n, v)
                  for n, v in wanted]

    saved = []
    try:
        for name, value in wanted:
            try:
                saved.append((name, platform.getPropertyDefaultValue(name)))
                platform.setPropertyDefaultValue(name, value)
            except Exception:
                pass
        yield platform
    finally:
        for name, value in saved:
            try:
                platform.setPropertyDefaultValue(name, value)
            except Exception:
                pass
