# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Any, Mapping, Optional


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

CPU_ATTRS = ("ncpu", "n_cpu", "n_cpus")
DEVICE_ATTRS = ("openmm_device_index", "device_index", "platform_device_index")


def default_cpu_budget() -> int:
    return max(1, (os.cpu_count() or 1) - 2)


def _coerce_cpu(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        ncpu = int(value)
    except (TypeError, ValueError):
        return None
    return max(1, ncpu)


def _value_from_mapping(source: Mapping[str, Any], names: tuple[str, ...]) -> Optional[Any]:
    for name in names:
        if name in source and source[name] is not None:
            return source[name]
    return None


def _value_from_object(source: Any, names: tuple[str, ...]) -> Optional[Any]:
    if isinstance(source, Mapping):
        return _value_from_mapping(source, names)
    for name in names:
        value = getattr(source, name, None)
        if value is not None:
            return value
    return None


def resolve_cpu_budget(source: Any = None, default: Any = None) -> int:
    """Return the canonical ChemEM CPU budget.

    ``source`` may be an integer, a config/options/system object, or a mapping.
    The modern name is ``ncpu``; legacy ``n_cpu`` and ``n_cpus`` are accepted.
    """
    direct = _coerce_cpu(source)
    if direct is not None and (
        isinstance(source, (int, float, str, bytes)) or source is None
    ):
        return direct

    for candidate in (source, getattr(source, "options", None)):
        if candidate is None:
            continue
        ncpu = _coerce_cpu(_value_from_object(candidate, CPU_ATTRS))
        if ncpu is not None:
            return ncpu

    ncpu = _coerce_cpu(default)
    if ncpu is not None:
        return ncpu
    return default_cpu_budget()


def apply_cpu_budget(target: Any, ncpu: Any) -> int:
    """Set canonical and legacy CPU aliases on an object and its options."""
    budget = resolve_cpu_budget(ncpu)
    for obj in (target, getattr(target, "options", None)):
        if obj is None:
            continue
        for attr in CPU_ATTRS:
            try:
                setattr(obj, attr, budget)
            except Exception:
                pass
    return budget


def resolve_device_index(source: Any = None) -> Optional[str]:
    for candidate in (source, getattr(source, "options", None)):
        if candidate is None:
            continue
        value = _value_from_object(candidate, DEVICE_ATTRS)
        if value is not None and str(value).strip() != "":
            return str(value)
    return None


def _platform_name(platform: Any) -> str:
    if hasattr(platform, "getName"):
        return str(platform.getName())
    return str(platform or "")


def _openmm_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def openmm_platform_properties(
    platform: Any,
    ncpu: Any = None,
    *,
    source: Any = None,
    precision: str = "single",
    device_index: Any = None,
    use_cpu_pme: Any = None,
    extra_properties: Optional[Mapping[str, Any]] = None,
) -> dict[str, str]:
    """Build platform properties without importing OpenMM.

    CPU supports the ``Threads`` property. CUDA/OpenCL do not support a generic
    CPU-thread property, so they get GPU-oriented properties only.
    """
    name = _platform_name(platform).upper()
    budget = resolve_cpu_budget(ncpu if ncpu is not None else source)
    props: dict[str, str] = {}

    if name == "CPU":
        props["Threads"] = str(budget)
    elif name in {"CUDA", "OPENCL"}:
        if precision:
            props["Precision"] = str(precision)
        if use_cpu_pme is None:
            use_cpu_pme = False
        props["UseCpuPme"] = "true" if _openmm_bool(use_cpu_pme) else "false"
        device = device_index if device_index is not None else resolve_device_index(source)
        if device is not None and str(device).strip() != "":
            props["DeviceIndex"] = str(device)

    if extra_properties:
        props.update({str(k): str(v) for k, v in extra_properties.items() if v is not None})
    return props


def openmm_platform_and_properties(
    platform_name: Any,
    *,
    source: Any = None,
    ncpu: Any = None,
    platform_properties: Optional[Mapping[str, Any]] = None,
):
    from openmm import Platform

    platform = Platform.getPlatformByName(str(platform_name))
    props = openmm_platform_properties(
        platform.getName(),
        ncpu=ncpu,
        source=source,
        extra_properties=platform_properties,
    )
    return platform, props


def make_openmm_simulation(
    topology,
    system,
    integrator,
    *,
    platform_name: Any = None,
    source: Any = None,
    ncpu: Any = None,
    platform_properties: Optional[Mapping[str, Any]] = None,
):
    from openmm import app

    if platform_name is None:
        return app.Simulation(topology, system, integrator)
    platform, props = openmm_platform_and_properties(
        platform_name,
        source=source,
        ncpu=ncpu,
        platform_properties=platform_properties,
    )
    return app.Simulation(topology, system, integrator, platform, props)


def make_openmm_context(
    system,
    integrator,
    *,
    platform_name: Any = None,
    source: Any = None,
    ncpu: Any = None,
    platform_properties: Optional[Mapping[str, Any]] = None,
):
    from openmm import Context

    if platform_name is None:
        return Context(system, integrator)
    platform, props = openmm_platform_and_properties(
        platform_name,
        source=source,
        ncpu=ncpu,
        platform_properties=platform_properties,
    )
    return Context(system, integrator, platform, props)


def set_native_thread_limit(ncpu: Any) -> int:
    budget = resolve_cpu_budget(ncpu)
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(budget)

    try:
        from ChemEM import grid_maps

        setter = getattr(grid_maps, "set_omp_num_threads", None)
        if setter is not None:
            setter(int(budget))
    except Exception:
        pass
    return budget


@contextmanager
def thread_limit_env(ncpu: Any):
    budget = resolve_cpu_budget(ncpu)
    old_values = {name: os.environ.get(name) for name in THREAD_ENV_VARS}
    old_grid_threads = None
    grid_thread_setter = None
    try:
        from ChemEM import grid_maps

        grid_thread_setter = getattr(grid_maps, "set_omp_num_threads", None)
        getter = getattr(grid_maps, "get_omp_max_threads", None)
        if getter is not None:
            old_grid_threads = int(getter())
    except Exception:
        pass

    set_native_thread_limit(budget)
    try:
        yield budget
    finally:
        for name, old in old_values.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
        if grid_thread_setter is not None and old_grid_threads is not None:
            try:
                grid_thread_setter(old_grid_threads)
            except Exception:
                pass
