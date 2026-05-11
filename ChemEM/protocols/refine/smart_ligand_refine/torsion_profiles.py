from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .types import (
    CandidatePose,
    SmartLigandRefinementConfig,
    TorsionInfo,
    TorsionProfile,
)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D
except Exception:  # pragma: no cover - optional at import time
    Chem = None
    AllChem = None
    Point3D = None

try:
    from openmm import unit
except Exception:  # pragma: no cover - optional at import time
    unit = None


def _ligand_mol(ligand):
    return getattr(ligand, "mol", ligand)


def wrap_angle_deg(angle: float) -> float:
    out = float(angle) % 360.0
    if out < 0.0:
        out += 360.0
    return out


def signed_angle_delta_deg(target: float, current: float) -> float:
    return ((float(target) - float(current) + 180.0) % 360.0) - 180.0


def dihedral_angle_deg(coords_A: np.ndarray, atom_indices: Sequence[int]) -> float:
    i, j, k, l = [int(x) for x in atom_indices]
    p0, p1, p2, p3 = np.asarray(coords_A, dtype=np.float64)[[i, j, k, l]]

    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    n1 = float(np.linalg.norm(b1))
    if n1 < 1e-12:
        return 0.0
    b1 = b1 / n1

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    nv = float(np.linalg.norm(v))
    nw = float(np.linalg.norm(w))
    if nv < 1e-12 or nw < 1e-12:
        return 0.0
    v = v / nv
    w = w / nw

    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return wrap_angle_deg(np.degrees(np.arctan2(y, x)))


def rotate_atoms_around_bond(
    coords_A: np.ndarray,
    bond_atoms: Tuple[int, int],
    atom_indices: Iterable[int],
    delta_angle_deg: float,
) -> np.ndarray:
    coords = np.asarray(coords_A, dtype=np.float64)
    out = coords.copy()
    atoms = np.asarray(list(atom_indices), dtype=int)
    if atoms.size == 0:
        return out

    a, b = [int(x) for x in bond_atoms]
    origin = coords[a]
    axis = coords[b] - origin
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return out
    axis = axis / norm

    theta = np.radians(float(delta_angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    ux, uy, uz = [float(x) for x in axis]
    k_mat = np.array(
        [[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]],
        dtype=np.float64,
    )
    rot = (np.eye(3) * c) + (s * k_mat) + ((1.0 - c) * np.outer(axis, axis))
    out[atoms] = (coords[atoms] - origin) @ rot.T + origin
    return out


def set_torsion_angle(
    coords_A: np.ndarray,
    torsion: TorsionInfo,
    target_angle_deg: float,
) -> np.ndarray:
    current = dihedral_angle_deg(coords_A, torsion.atom_indices)
    delta = signed_angle_delta_deg(target_angle_deg, current)
    return rotate_atoms_around_bond(
        coords_A,
        torsion.bond_atoms,
        torsion.downstream_atoms,
        delta,
    )


def _heavy_degree(mol, n_coords: int) -> List[int]:
    degree = [0] * int(n_coords)
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        if i >= n_coords or j >= n_coords:
            continue
        if mol.GetAtomWithIdx(i).GetAtomicNum() <= 1:
            continue
        if mol.GetAtomWithIdx(j).GetAtomicNum() <= 1:
            continue
        degree[i] += 1
        degree[j] += 1
    return degree


def _bond_side_atoms(mol, start: int, blocked: int, n_coords: int) -> List[int]:
    visited = set()
    stack = [int(start)]
    while stack:
        atom_idx = int(stack.pop())
        if atom_idx in visited or atom_idx >= n_coords:
            continue
        visited.add(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        for nb in atom.GetNeighbors():
            nb_idx = int(nb.GetIdx())
            if nb_idx >= n_coords:
                continue
            if atom_idx == start and nb_idx == blocked:
                continue
            if atom_idx == blocked and nb_idx == start:
                continue
            if nb_idx not in visited:
                stack.append(nb_idx)
    return sorted(visited)


def _first_neighbor_on_side(mol, center: int, excluded: int, side: set) -> Optional[int]:
    for nb in mol.GetAtomWithIdx(int(center)).GetNeighbors():
        idx = int(nb.GetIdx())
        if idx == int(excluded):
            continue
        if idx in side:
            return idx
    return None


def detect_rotatable_torsions(
    ligand,
    coords_A: Optional[np.ndarray] = None,
) -> List[TorsionInfo]:
    mol = _ligand_mol(ligand)
    if mol is None or Chem is None or not hasattr(mol, "GetBonds"):
        return []
    n_coords = int(np.asarray(coords_A).shape[0]) if coords_A is not None else mol.GetNumAtoms()
    degree = _heavy_degree(mol, n_coords)

    torsions: List[TorsionInfo] = []
    torsion_id = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if bond.GetIsAromatic() or bond.IsInRing():
            continue
        b = int(bond.GetBeginAtomIdx())
        c = int(bond.GetEndAtomIdx())
        if b >= n_coords or c >= n_coords:
            continue
        if mol.GetAtomWithIdx(b).GetAtomicNum() <= 1:
            continue
        if mol.GetAtomWithIdx(c).GetAtomicNum() <= 1:
            continue
        if degree[b] <= 1 or degree[c] <= 1:
            continue

        side_b = _bond_side_atoms(mol, b, c, n_coords)
        side_c = _bond_side_atoms(mol, c, b, n_coords)
        if c in side_b or b in side_c:
            continue  # ring or otherwise still connected

        for left, right, downstream in ((b, c, side_c), (c, b, side_b)):
            down_set = set(int(x) for x in downstream)
            up_set = set(range(n_coords)) - down_set
            a = _first_neighbor_on_side(mol, left, right, up_set)
            d = _first_neighbor_on_side(mol, right, left, down_set)
            if a is None or d is None:
                continue
            current = 0.0
            if coords_A is not None:
                current = dihedral_angle_deg(coords_A, (a, left, right, d))
            torsions.append(
                TorsionInfo(
                    torsion_id=int(torsion_id),
                    atom_indices=(int(a), int(left), int(right), int(d)),
                    bond_atoms=(int(left), int(right)),
                    downstream_atoms=[int(x) for x in downstream],
                    current_angle_deg=float(current),
                )
            )
            torsion_id += 1

    return torsions


def _mol_with_coords(mol, coords_A: np.ndarray):
    if Chem is None:
        return None
    work = Chem.Mol(mol)
    coords = np.asarray(coords_A, dtype=np.float64)
    if work.GetNumConformers() == 0:
        conf = Chem.Conformer(work.GetNumAtoms())
        work.AddConformer(conf, assignId=True)
    conf = work.GetConformer(0)
    for idx in range(min(work.GetNumAtoms(), coords.shape[0])):
        x, y, z = coords[idx]
        conf.SetAtomPosition(int(idx), Point3D(float(x), float(y), float(z)))
    return work


def _rdkit_energy_kcal(mol, coords_A: np.ndarray) -> float:
    if Chem is None or AllChem is None or mol is None:
        return 0.0
    try:
        work = _mol_with_coords(mol, coords_A)
        props = AllChem.MMFFGetMoleculeProperties(work, mmffVariant="MMFF94s")
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(work, props, confId=0)
        else:
            ff = None
        if ff is None:
            ff = AllChem.UFFGetMoleculeForceField(work, confId=0)
        if ff is None:
            return 0.0
        return float(ff.CalcEnergy())
    except Exception:
        return 0.0


def _local_minima(angles: List[float], energies: List[float]) -> List[int]:
    if not energies:
        return []
    n = len(energies)
    minima = []
    for i in range(n):
        prev_e = energies[(i - 1) % n]
        next_e = energies[(i + 1) % n]
        e = energies[i]
        if e <= prev_e and e <= next_e:
            minima.append(i)
    if not minima:
        minima = [int(np.argmin(energies))]
    return minima


def build_profile_from_scan(
    torsion: TorsionInfo,
    angles: Sequence[float],
    energies: Sequence[float],
    config: Optional[SmartLigandRefinementConfig] = None,
    *,
    source: str = "scan",
) -> TorsionProfile:
    config = config or SmartLigandRefinementConfig()
    scan_angles = [wrap_angle_deg(float(a)) for a in angles]
    finite = np.asarray(energies, dtype=np.float64)
    finite = np.where(np.isfinite(finite), finite, np.inf)
    e_min = float(np.min(finite)) if finite.size else 0.0
    relative = [float(e - e_min) if np.isfinite(e) else float("inf") for e in finite]
    minima_idx = _local_minima(scan_angles, relative)

    filtered = [
        idx
        for idx in minima_idx
        if relative[idx] <= float(config.max_relative_minimum_energy_kcal)
    ]
    if not filtered and relative:
        filtered = [int(np.argmin(relative))]

    return TorsionProfile(
        torsion_id=int(torsion.torsion_id),
        minima_deg=[wrap_angle_deg(scan_angles[i]) for i in filtered],
        relative_energies=[float(relative[i]) for i in filtered],
        scan_angles_deg=[wrap_angle_deg(a) for a in scan_angles],
        scan_energies=[float(e) for e in relative],
        source=str(source),
        atom_indices=tuple(int(i) for i in torsion.atom_indices),
    )


def _build_rdkit_profile_worker(payload):
    torsion, base_coords, mol, config = payload
    builder = TorsionProfileBuilder(config)
    profile = builder._build_scanned_profile(
        torsion,
        np.asarray(base_coords, dtype=np.float64),
        lambda coords: _rdkit_energy_kcal(mol, coords),
        source="rdkit",
    )
    return int(torsion.torsion_id), profile, builder.timing_report()


def _periodic_torsion_force(openmm_system):
    if openmm_system is None or not hasattr(openmm_system, "getNumForces"):
        return None
    for idx in range(int(openmm_system.getNumForces())):
        force = openmm_system.getForce(idx)
        if force.__class__.__name__ == "PeriodicTorsionForce":
            return force
    return None


class TorsionProfileBuilder:
    def __init__(
        self,
        config: Optional[SmartLigandRefinementConfig] = None,
        geometry_oracle=None,
    ):
        self.config = config or SmartLigandRefinementConfig()
        self.geometry_oracle = geometry_oracle
        self._timings: Dict[str, float] = {}
        self._call_counts: Dict[str, int] = {}
        self._profile_timings = bool(getattr(self.config, "profile_timings", False))

    def _add_timing(self, name: str, seconds: float) -> None:
        if not self._profile_timings:
            return
        self._timings[name] = float(self._timings.get(name, 0.0) + seconds)

    def _count_call(self, name: str) -> None:
        if not self._profile_timings:
            return
        self._call_counts[name] = int(self._call_counts.get(name, 0) + 1)

    def _merge_timing_report(self, report: dict) -> None:
        if not self._profile_timings or not report:
            return
        for key, value in dict(report.get("timings_s", {})).items():
            self._timings[str(key)] = float(
                self._timings.get(str(key), 0.0) + float(value)
            )
        for key, value in dict(report.get("call_counts", {})).items():
            self._call_counts[str(key)] = int(
                self._call_counts.get(str(key), 0) + int(value)
            )

    def timing_report(self) -> dict:
        return {
            "timings_s": {str(k): float(v) for k, v in sorted(self._timings.items())},
            "call_counts": {str(k): int(v) for k, v in sorted(self._call_counts.items())},
        }

    def _parallel_worker_count(self, n_jobs: int) -> int:
        if int(n_jobs) < 2 or bool(getattr(self.config, "no_para", False)):
            return 1
        requested = getattr(self.config, "max_parallel_workers", None)
        if requested is None:
            cpus_per_site = max(1, int(getattr(self.config, "cpus_per_site", 1) or 1))
            requested = max(1, (os.cpu_count() or 1) // cpus_per_site)
        return max(1, min(int(requested), int(n_jobs)))

    def build_profiles(
        self,
        ligand,
        torsions: Sequence[TorsionInfo],
        openmm_system=None,
        context=None,
    ) -> Dict[int, TorsionProfile]:
        mol = _ligand_mol(ligand)
        base_coords = self._coords_from_ligand(ligand)
        profiles: Dict[int, TorsionProfile] = {}
        max_count = max(0, int(self.config.max_torsion_profile_count))
        torsion_list = list(torsions)[:max_count]
        source_policy = str(getattr(self.config, "torsion_profile_source", "auto")).lower()
        openmm_energy, cleanup_openmm = None, None
        if source_policy in ("auto", "openmm", "openff"):
            openmm_energy, cleanup_openmm = self._make_openmm_torsion_energy(
                openmm_system,
                context,
            )

        total_t0 = time.perf_counter()
        try:
            used_parallel = False
            if openmm_energy is None and source_policy not in ("openmm", "openff"):
                max_workers = self._parallel_worker_count(len(torsion_list))
                if max_workers > 1:
                    parallel_t0 = time.perf_counter()
                    try:
                        worker_config = replace(self.config, debug_printer=None)
                        with ProcessPoolExecutor(max_workers=max_workers) as pool:
                            futures = [
                                pool.submit(
                                    _build_rdkit_profile_worker,
                                    (torsion, base_coords, mol, worker_config),
                                )
                                for torsion in torsion_list
                            ]
                            for fut in as_completed(futures):
                                torsion_id, profile, timing_report = fut.result()
                                self._merge_timing_report(timing_report)
                                if profile is not None:
                                    profiles[int(torsion_id)] = profile
                        used_parallel = True
                        self._add_timing("scan.parallel_rdkit", time.perf_counter() - parallel_t0)
                    except Exception as exc:
                        self._count_call(f"scan.parallel_fallback.{type(exc).__name__}")

            if used_parallel:
                return profiles

            for torsion in torsion_list:
                profile = None
                if source_policy in ("auto", "openmm", "openff") and openmm_energy is not None:
                    profile = self._build_scanned_profile(
                        torsion,
                        base_coords,
                        openmm_energy,
                        source="openff_openmm",
                    )
                if profile is None and source_policy not in ("openmm", "openff"):
                    profile = self._build_scanned_profile(
                        torsion,
                        base_coords,
                        lambda coords: self._energy_kcal(mol, coords),
                        source="rdkit",
                    )
                if profile is not None:
                    profiles[int(torsion.torsion_id)] = profile
        finally:
            if cleanup_openmm is not None:
                cleanup_openmm()
            self._add_timing("scan.total", time.perf_counter() - total_t0)

        return profiles

    def _build_scanned_profile(
        self,
        torsion: TorsionInfo,
        base_coords: np.ndarray,
        energy_fn,
        *,
        source: str,
    ) -> Optional[TorsionProfile]:
        angles = list(
            np.arange(
                0.0,
                360.0,
                max(float(self.config.torsion_scan_step_deg), 1.0),
                dtype=np.float64,
            )
        )
        energies = []
        try:
            for angle in angles:
                self._count_call("scan.angle")
                set_t0 = time.perf_counter()
                coords = set_torsion_angle(base_coords, torsion, float(angle))
                self._add_timing("scan.set_torsion_angle", time.perf_counter() - set_t0)
                energy_t0 = time.perf_counter()
                energies.append(float(energy_fn(coords)))
                self._add_timing("scan.energy_fn", time.perf_counter() - energy_t0)
            return build_profile_from_scan(
                torsion,
                angles,
                energies,
                self.config,
                source=source,
            )
        except Exception:
            return None

    def _coords_from_ligand(self, ligand) -> np.ndarray:
        if hasattr(ligand, "get_positions"):
            coords = ligand.get_positions()
            if np.asarray(coords).size:
                return np.asarray(coords, dtype=np.float64)
        mol = _ligand_mol(ligand)
        if mol is not None and hasattr(mol, "GetConformer") and mol.GetNumConformers():
            return np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)
        raise ValueError("Cannot build torsion profiles without ligand coordinates")

    def _energy_kcal(self, mol, coords_A: np.ndarray) -> float:
        if self.geometry_oracle is not None:
            try:
                cand = CandidatePose(
                    coords=np.asarray(coords_A, dtype=np.float64),
                    move_type="torsion_scan",
                    move_metadata={},
                )
                metrics = self.geometry_oracle.evaluate(cand)
                e = float(metrics.get("ligand_internal_energy", metrics.get("openmm_energy", 0.0)))
                if np.isfinite(e):
                    return e
            except Exception:
                pass
        return _rdkit_energy_kcal(mol, coords_A)

    def _make_openmm_torsion_energy(self, openmm_system, context):
        if openmm_system is None or context is None or unit is None:
            return None, None
        torsion_force = _periodic_torsion_force(openmm_system)
        if torsion_force is None:
            return None, None
        if not hasattr(openmm_system, "getNumForces"):
            return None, None

        saved_groups = []
        try:
            for idx in range(int(openmm_system.getNumForces())):
                force = openmm_system.getForce(idx)
                saved_groups.append((force, int(force.getForceGroup())))
                force.setForceGroup(30)
            torsion_force.setForceGroup(31)
            if hasattr(context, "reinitialize"):
                context.reinitialize(preserveState=True)
        except Exception:
            for force, group in saved_groups:
                try:
                    force.setForceGroup(group)
                except Exception:
                    pass
            return None, None

        def cleanup():
            for force, group in saved_groups:
                try:
                    force.setForceGroup(group)
                except Exception:
                    pass
            try:
                if hasattr(context, "reinitialize"):
                    context.reinitialize(preserveState=True)
            except Exception:
                pass

        def energy(coords_A: np.ndarray) -> float:
            if not self._set_openmm_positions(context, np.asarray(coords_A, dtype=np.float64)):
                raise ValueError("Cannot map ligand coordinates into OpenMM context")
            state = context.getState(getEnergy=True, groups={31})
            return float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

        return energy, cleanup

    def _set_openmm_positions(self, context, coords_A: np.ndarray) -> bool:
        if self.geometry_oracle is not None and hasattr(
            self.geometry_oracle, "_set_context_ligand_positions"
        ):
            try:
                self.geometry_oracle._set_context_ligand_positions(coords_A)
                return True
            except Exception:
                pass
        try:
            if hasattr(context, "getSystem"):
                n_particles = int(context.getSystem().getNumParticles())
                if n_particles == int(coords_A.shape[0]):
                    context.setPositions(np.asarray(coords_A, dtype=np.float64) * unit.angstrom)
                    return True
        except Exception:
            pass
        return False


def candidate_angles_from_profile(
    profile: TorsionProfile,
    offsets_deg: Sequence[float],
) -> List[float]:
    seen = set()
    out: List[float] = []
    for minimum in profile.minima_deg:
        for offset in offsets_deg:
            angle = round(wrap_angle_deg(float(minimum) + float(offset)), 6)
            if angle in seen:
                continue
            seen.add(angle)
            out.append(float(angle))
    return out
