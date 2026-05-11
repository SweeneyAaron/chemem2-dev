from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .types import CandidatePose, SmartLigandRefinementConfig

try:
    from openmm import LocalEnergyMinimizer, unit
except Exception:  # pragma: no cover - optional in light unit tests
    LocalEnergyMinimizer = None
    unit = None

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D
except Exception:  # pragma: no cover - optional at import time
    Chem = None
    AllChem = None
    Point3D = None


def _ligand_mol(ligand):
    return getattr(ligand, "mol", ligand)


def _structure_coords_A(structure) -> np.ndarray:
    if structure is None:
        return np.zeros((0, 3), dtype=np.float64)
    if isinstance(structure, np.ndarray):
        arr = np.asarray(structure, dtype=np.float64)
        return arr.reshape((-1, 3)) if arr.size else np.zeros((0, 3), dtype=np.float64)
    if hasattr(structure, "complex_structure"):
        structure = structure.complex_structure
    if hasattr(structure, "positions") and structure.positions is not None:
        try:
            pos = structure.positions
            if hasattr(pos, "value_in_unit") and unit is not None:
                return np.asarray(pos.value_in_unit(unit.angstrom), dtype=np.float64)
            return np.asarray(pos, dtype=np.float64)
        except Exception:
            pass
    if hasattr(structure, "atoms"):
        coords = []
        for atom in structure.atoms:
            try:
                if getattr(atom, "element", 6) == 1:
                    continue
                coords.append([float(atom.xx), float(atom.xy), float(atom.xz)])
            except Exception:
                continue
        return np.asarray(coords, dtype=np.float64)
    return np.zeros((0, 3), dtype=np.float64)


def _mol_with_coords(mol, coords_A: np.ndarray):
    if Chem is None or mol is None:
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
        ff = AllChem.MMFFGetMoleculeForceField(work, props, confId=0) if props else None
        if ff is None:
            ff = AllChem.UFFGetMoleculeForceField(work, confId=0)
        if ff is None:
            return 0.0
        return float(ff.CalcEnergy())
    except Exception:
        return 0.0


class GeometryOracle:
    """
    OpenMM/OpenFF geometry oracle.

    This class can evaluate or clean a candidate in an OpenMM context, but it
    does not add or own any density forces. When OpenMM context/system objects
    are unavailable, it falls back to RDKit/UFF-MMFF and simple geometric
    checks so the search controller remains testable.
    """

    def __init__(
        self,
        protein=None,
        ligand=None,
        openmm_system=None,
        openmm_context=None,
        *,
        ligand_context_indices: Optional[Sequence[int]] = None,
        protein_coords_A: Optional[np.ndarray] = None,
        reference_coords_A: Optional[np.ndarray] = None,
        config: Optional[SmartLigandRefinementConfig] = None,
    ):
        self.protein = protein
        self.ligand = ligand
        self.mol = _ligand_mol(ligand)
        self.openmm_system = openmm_system
        self.context = openmm_context
        self.config = config or SmartLigandRefinementConfig()
        self.ligand_context_indices = (
            np.asarray(ligand_context_indices, dtype=int)
            if ligand_context_indices is not None
            else (
                np.asarray(self.config.ligand_context_indices, dtype=int)
                if self.config.ligand_context_indices is not None
                else None
            )
        )

        if protein_coords_A is None:
            protein_coords_A = self.config.protein_coords_A
        if protein_coords_A is None:
            protein_coords_A = _structure_coords_A(protein)
        self.protein_coords_A = np.asarray(protein_coords_A, dtype=np.float64)
        if self.protein_coords_A.ndim != 2 or self.protein_coords_A.shape[-1] != 3:
            self.protein_coords_A = np.zeros((0, 3), dtype=np.float64)

        self.reference_coords_A = None
        self.reference_bond_lengths = {}
        self.reference_angles = {}
        self.reference_energy_kcal = 0.0
        if reference_coords_A is not None:
            self.set_reference(reference_coords_A)

    def set_reference(self, coords_A: np.ndarray) -> None:
        coords = np.asarray(coords_A, dtype=np.float64)
        self.reference_coords_A = coords.copy()
        self.reference_bond_lengths = self._bond_lengths(coords)
        self.reference_angles = self._angles(coords)
        self.reference_energy_kcal = self._energy_kcal(coords)

    def evaluate(self, candidate: CandidatePose) -> Dict[str, float]:
        coords = np.asarray(candidate.coords, dtype=np.float64)
        if self.reference_coords_A is None:
            self.set_reference(coords)

        openmm_energy = self._energy_kcal(coords)
        bond_dev = self._max_bond_deviation(coords)
        angle_dev = self._max_angle_deviation(coords)
        ring_error = self._ring_planarity_error(coords)
        clash_score = self._protein_ligand_clash_score(coords)

        metrics = {
            "openmm_energy": float(openmm_energy),
            "ligand_internal_energy": float(openmm_energy),
            "delta_ligand_internal_energy": float(
                openmm_energy - self.reference_energy_kcal
            ),
            "max_bond_deviation": float(bond_dev),
            "max_angle_deviation": float(angle_dev),
            "has_chirality_error": False,
            "has_ring_planarity_error": bool(ring_error),
            "protein_ligand_clash_score": float(clash_score),
        }
        candidate.geometry_metrics = metrics
        return metrics

    def clean(self, candidate: CandidatePose) -> CandidatePose:
        coords = np.asarray(candidate.coords, dtype=np.float64)
        cleaned = coords.copy()
        if self.context is not None and unit is not None and LocalEnergyMinimizer is not None:
            try:
                self._set_context_ligand_positions(coords)
                LocalEnergyMinimizer.minimize(
                    self.context,
                    maxIterations=int(self.config.clean_max_iterations),
                )
                cleaned = self._get_context_ligand_positions_A(coords.shape[0])
            except Exception:
                cleaned = coords.copy()
        elif Chem is not None and AllChem is not None and self.mol is not None:
            try:
                work = _mol_with_coords(self.mol, coords)
                props = AllChem.MMFFGetMoleculeProperties(work, mmffVariant="MMFF94s")
                if props is not None:
                    AllChem.MMFFOptimizeMolecule(
                        work,
                        mmffVariant="MMFF94s",
                        maxIters=int(self.config.clean_max_iterations),
                    )
                else:
                    AllChem.UFFOptimizeMolecule(
                        work,
                        maxIters=int(self.config.clean_max_iterations),
                    )
                cleaned = np.asarray(work.GetConformer(0).GetPositions(), dtype=np.float64)
            except Exception:
                cleaned = coords.copy()

        out = CandidatePose(
            coords=cleaned,
            move_type=candidate.move_type,
            move_metadata=dict(candidate.move_metadata),
            map_metrics=candidate.map_metrics,
            geometry_metrics=None,
            score=candidate.score,
        )
        return out

    def is_acceptable(self, candidate: CandidatePose) -> bool:
        metrics = candidate.geometry_metrics or self.evaluate(candidate)
        if metrics.get("has_chirality_error", False):
            return False
        if metrics.get("has_ring_planarity_error", False):
            return False
        if (
            float(metrics.get("delta_ligand_internal_energy", 0.0))
            > float(self.config.max_ligand_strain_increase_kcal)
        ):
            return False
        if (
            float(metrics.get("max_bond_deviation", 0.0))
            > float(self.config.max_bond_deviation_A)
        ):
            return False
        if (
            float(metrics.get("max_angle_deviation", 0.0))
            > float(self.config.max_angle_deviation_deg)
        ):
            return False
        return True

    def _energy_kcal(self, coords_A: np.ndarray) -> float:
        if self.context is not None and unit is not None:
            try:
                self._set_context_ligand_positions(coords_A)
                state = self.context.getState(getEnergy=True)
                return float(
                    state.getPotentialEnergy().value_in_unit(
                        unit.kilocalories_per_mole
                    )
                )
            except Exception:
                pass
        return _rdkit_energy_kcal(self.mol, coords_A)

    def _ligand_indices_for_context(self, n_ligand_atoms: int) -> Optional[np.ndarray]:
        if self.context is None or unit is None:
            return None
        if self.ligand_context_indices is not None:
            idx = np.asarray(self.ligand_context_indices, dtype=int)
            if idx.size == n_ligand_atoms:
                return idx
        try:
            state = self.context.getState(getPositions=True)
            pos = np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=np.float64,
            )
            n_particles = int(pos.shape[0])
        except Exception:
            return None
        if n_particles == n_ligand_atoms:
            return np.arange(n_ligand_atoms, dtype=int)
        if n_particles > n_ligand_atoms:
            return np.arange(n_particles - n_ligand_atoms, n_particles, dtype=int)
        return None

    def _set_context_ligand_positions(self, coords_A: np.ndarray) -> None:
        idx = self._ligand_indices_for_context(int(coords_A.shape[0]))
        if idx is None:
            raise ValueError("Cannot map ligand coordinates into OpenMM context")
        state = self.context.getState(getPositions=True)
        pos_nm = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64,
        )
        pos_nm[idx] = np.asarray(coords_A, dtype=np.float64) * 0.1
        self.context.setPositions(pos_nm * unit.nanometer)

    def _get_context_ligand_positions_A(self, n_ligand_atoms: int) -> np.ndarray:
        idx = self._ligand_indices_for_context(int(n_ligand_atoms))
        if idx is None:
            raise ValueError("Cannot map ligand coordinates from OpenMM context")
        state = self.context.getState(getPositions=True)
        pos_nm = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64,
        )
        return np.asarray(pos_nm[idx], dtype=np.float64) * 10.0

    def _bond_lengths(self, coords_A: np.ndarray) -> Dict[tuple, float]:
        out = {}
        if self.mol is None or not hasattr(self.mol, "GetBonds"):
            return out
        coords = np.asarray(coords_A, dtype=np.float64)
        for bond in self.mol.GetBonds():
            i = int(bond.GetBeginAtomIdx())
            j = int(bond.GetEndAtomIdx())
            if i >= coords.shape[0] or j >= coords.shape[0]:
                continue
            out[(i, j)] = float(np.linalg.norm(coords[i] - coords[j]))
        return out

    def _angles(self, coords_A: np.ndarray) -> Dict[tuple, float]:
        out = {}
        if self.mol is None or not hasattr(self.mol, "GetAtoms"):
            return out
        coords = np.asarray(coords_A, dtype=np.float64)
        for atom in self.mol.GetAtoms():
            j = int(atom.GetIdx())
            if j >= coords.shape[0]:
                continue
            neighbors = [
                int(nb.GetIdx())
                for nb in atom.GetNeighbors()
                if int(nb.GetIdx()) < coords.shape[0]
            ]
            for a_idx, i in enumerate(neighbors):
                for k in neighbors[a_idx + 1 :]:
                    v1 = coords[i] - coords[j]
                    v2 = coords[k] - coords[j]
                    n = float(np.linalg.norm(v1) * np.linalg.norm(v2))
                    if n < 1e-12:
                        continue
                    cosang = np.clip(float(np.dot(v1, v2) / n), -1.0, 1.0)
                    out[(i, j, k)] = float(np.degrees(np.arccos(cosang)))
        return out

    def _max_bond_deviation(self, coords_A: np.ndarray) -> float:
        if not self.reference_bond_lengths:
            return 0.0
        current = self._bond_lengths(coords_A)
        vals = []
        for key, ref in self.reference_bond_lengths.items():
            if key in current:
                vals.append(abs(float(current[key]) - float(ref)))
        return float(max(vals)) if vals else 0.0

    def _max_angle_deviation(self, coords_A: np.ndarray) -> float:
        if not self.reference_angles:
            return 0.0
        current = self._angles(coords_A)
        vals = []
        for key, ref in self.reference_angles.items():
            if key in current:
                vals.append(abs(float(current[key]) - float(ref)))
        return float(max(vals)) if vals else 0.0

    def _ring_planarity_error(self, coords_A: np.ndarray) -> bool:
        if self.mol is None or not hasattr(self.mol, "GetRingInfo"):
            return False
        coords = np.asarray(coords_A, dtype=np.float64)
        try:
            for ring in self.mol.GetRingInfo().AtomRings():
                idx = [int(i) for i in ring if int(i) < coords.shape[0]]
                if len(idx) < 4:
                    continue
                pts = coords[idx]
                centred = pts - pts.mean(axis=0)
                _, svals, _ = np.linalg.svd(centred, full_matrices=False)
                if svals.size >= 3 and float(svals[-1]) > 0.25:
                    return True
        except Exception:
            return False
        return False

    def _protein_ligand_clash_score(self, coords_A: np.ndarray) -> float:
        prot = np.asarray(self.protein_coords_A, dtype=np.float64)
        if prot.size == 0:
            return 0.0
        lig = np.asarray(coords_A, dtype=np.float64)
        if lig.size == 0:
            return 0.0
        try:
            tree = cKDTree(prot)
            dists, _ = tree.query(lig, k=1)
            overlap = np.maximum(
                0.0,
                float(self.config.protein_ligand_clash_distance_A)
                - np.asarray(dists, dtype=np.float64),
            )
            return float(np.mean(overlap * overlap))
        except Exception:
            return 0.0
