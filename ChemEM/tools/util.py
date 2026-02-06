# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

from typing import Iterable, Sequence, Optional


def _get_platforms():
    # Import lazily so non-OpenMM workflows can still import ChemEM
    from openmm import Platform
    return Platform


def platform_ok(platform, *, verbose: bool = False) -> bool:
    """
    Run a tiny 2-particle test to confirm kernels/context/integration work.
    Returns True if it runs, False otherwise.
    """
    try:
        from openmm import unit, Context, VerletIntegrator
        from openmm import System as OpenMMSystem
        from openmm.openmm import CustomNonbondedForce

        sys = OpenMMSystem()
        sys.addParticle(12.0 * unit.amu)
        sys.addParticle(12.0 * unit.amu)

        nb = CustomNonbondedForce("0.0")
        nb.addParticle([])
        nb.addParticle([])
        sys.addForce(nb)

        integrator = VerletIntegrator(1.0 * unit.femtoseconds)
        ctx = Context(sys, integrator, platform)

        ctx.setPositions([[0, 0, 0], [0.1, 0, 0]] * unit.nanometer)
        ctx.getState(getEnergy=True)  # force kernel compile

        ctx.setVelocitiesToTemperature(300 * unit.kelvin)
        integrator.step(10)

        return True
    
    except Exception as e:
        if verbose:
            try:
                name = platform.getName()
            except Exception:
                name = "<unknown>"
            print(f"[{name}] failed self-test → {e}")
        return False


def _platform_by_name(name: str):
    Platform = _get_platforms()
    for idx in range(Platform.getNumPlatforms()):
        plat = Platform.getPlatform(idx)
        if plat.getName().upper() == name.upper():
            return plat
    return None


def resolve_platform_name(
    requested: Optional[str],
    *,
    preferred: Sequence[str] = ("CUDA", "OpenCL", "CPU"),
    self_test: bool = True,
    verbose: bool = True,
) -> str:
    """
    requested:
      - "auto" / "best" / None : choose first working platform in `preferred`
      - "CPU" / "CUDA" / "OpenCL" : force that platform (optionally self-tested)
    Returns the platform name string to store in your System.
    """
    if requested is None:
        mode = "auto"
    else:
        mode = str(requested).strip()

    if mode == "":
        mode = "auto"

    mode_upper = mode.upper()
    auto = mode_upper in ("AUTO", "BEST")

    if auto:
        for name in preferred:
            plat = _platform_by_name(name)
            if plat is None:
                continue
            if (not self_test) or platform_ok(plat, verbose=verbose):
                if verbose:
                    print(f"Using {plat.getName()} platform")
                return plat.getName()
        raise RuntimeError("No working OpenMM platform found.")
    else:
        plat = _platform_by_name(mode_upper)
        if plat is None:
            raise ValueError(
                f"Requested OpenMM platform '{mode}' not found. "
                f"Available are: {available_platform_names()}"
            )
        if self_test and not platform_ok(plat, verbose=verbose):
            raise RuntimeError(f"Requested platform '{mode}' failed self-test.")
        if verbose:
            print(f"Using {plat.getName()} platform (forced)")
        return plat.getName()


def available_platform_names() -> list[str]:
    Platform = _get_platforms()
    return [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
