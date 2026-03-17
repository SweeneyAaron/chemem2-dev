# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>
"""
Standalone helpers for building parameterised ion ParmEd structures
that can be merged into an existing protein/ligand complex.

The ion is parameterised with a supplied OpenMM ForceField object,
then converted to a ParmEd Structure for merging.

Notes
-----
- The coordination number is validated and carried as metadata, but it does
  NOT alter the standard nonbonded ion parameters assigned by the force field.
- Coordination geometry should be enforced later with custom restraints /
  custom forces in the ion-fixer workflow.
"""

from __future__ import annotations

from typing import Union
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import parmed
from openmm import Vec3, unit
from openmm import app
from itertools import combinations, permutations







ION_TEMPLATE_INFO = {
    # name passed by user -> (residue name in FF, atom name in FF, element symbol)
    "MG": ("MG", "MG", "Mg"),
    "ZN": ("ZN", "ZN", "Zn"),
    "CA": ("CA", "CA", "Ca"),
    "MN": ("MN", "MN", "Mn"),
    "FE2": ("FE2", "FE2", "Fe"),
    "FE": ("FE", "FE", "Fe"),   # in amber14/tip3pfb.xml FE is Fe3+, FE2 is Fe2+
    "NA": ("NA", "NA", "Na"),
    "K": ("K", "K", "K"),
    "CL": ("CL", "CL", "Cl"),
    "LI": ("LI", "LI", "Li"),
}
# Canonical geometry -> integer ID (stable ordering you can store in files)
COORD_GEOM_TO_INT = {
    "linear": 0,
    "trigonal planar": 1,
    "tetrahedral": 2,
    "square planar": 3,
    "square pyramidal": 4,
    "trigonal bipyramidal": 5,
    "octahedral": 6,
    "pentagonal bipyramidal": 7,
}

# Optional inverse map
INT_TO_COORD_GEOM = {v: k for k, v in COORD_GEOM_TO_INT.items()}

# Optional: alias map to normalize common typos/casing before lookup
COORD_GEOM_ALIASES = {
    "triganal planer": "trigonal planar",
    "triganal planar": "trigonal planar",
    "triganol planar": "trigonal planar",
    "square pyrimidal": "square pyramidal",
    "pentagonal bipyrimidal": "pentagonal bipyramidal",
    "octahedral": "octahedral",
    "square planar": "square planar",
    "linear": "linear",
    "trigonal bipyramidal": "trigonal bipyramidal",
    "tetrahedral": "tetrahedral",
    "pentagonal bipyramidal": "pentagonal bipyramidal",
}

def normalize_coordination_geometry(name: str) -> str:
    n = " ".join(str(name).strip().lower().split())
    return COORD_GEOM_ALIASES.get(n, n)

def coord_geom_to_int(name: str) -> int:
    g = normalize_coordination_geometry(name)
    if g not in COORD_GEOM_TO_INT:
        raise KeyError(f"Unknown coordination geometry {name!r} -> {g!r}")
    return COORD_GEOM_TO_INT[g]

def _norm_ion_name(name: str) -> str:
    s = str(name).strip().upper()
    # allow a few relaxed user inputs
    s = s.replace("+", "")
    s = s.replace("-", "")
    if s == "MG2":
        return "MG"
    if s == "ZN2":
        return "ZN"
    if s == "CA2":
        return "CA"
    if s == "MN2":
        return "MN"
    if s == "NA1":
        return "NA"
    return s


def _as_xyz_angstrom(position: Optional[Union[Sequence[float], np.ndarray]]) -> np.ndarray:
    if position is None:
        return np.zeros(3, dtype=float)
    arr = np.asarray(position, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"position must have shape (3,), got {arr.shape}")
    return arr


def create_parameterized_ion_structure(
    ion_name: str,
    coordination_number: int,
    forcefield: app.ForceField,
    position: Optional[Union[Sequence[float], np.ndarray]] = None,
    residue_id: str = "1",
    chain_id: str = "Z",
) -> parmed.Structure:
    """
    Return a parameterised one-ion ParmEd Structure ready to merge
    into a larger ParmEd complex.

    Notes
    -----
    coordination_number is accepted as metadata for later restraint setup.
    It does not alter the base nonbonded FF assignment.
    """
    ion_name = _norm_ion_name(ion_name)
    if ion_name not in ION_TEMPLATE_INFO:
        raise ValueError(
            f"Unsupported ion {ion_name!r}. Supported: {sorted(ION_TEMPLATE_INFO)}"
        )

    try:
        coordination_number = int(coordination_number)
    except Exception as e:
        raise ValueError("coordination_number must be an integer") from e

    if coordination_number < 1:
        raise ValueError("coordination_number must be >= 1")

    if not isinstance(forcefield, app.ForceField):
        raise TypeError("forcefield must be an openmm.app.ForceField")

    pos = _as_xyz_angstrom(position)

    resname, atomname, element_symbol = ION_TEMPLATE_INFO[ion_name]
    element = app.Element.getBySymbol(element_symbol)

    # Build a topology that exactly matches the XML residue template
    top = app.Topology()
    chain = top.addChain(chain_id)
    res = top.addResidue(resname, chain, id=str(residue_id))
    top.addAtom(atomname, element, res)

    positions = unit.Quantity(
        [Vec3(float(pos[0]), float(pos[1]), float(pos[2]))],
        unit.angstrom
    )

    try:
        system = forcefield.createSystem(top, nonbondedMethod=app.NoCutoff)
    except Exception as e:
        raise RuntimeError(
            f"Failed to parameterise ion {ion_name!r}. "
            f"Check that the ForceField really includes the matching ion template "
            f"and that the residue/atom names match exactly."
        ) from e

    structure = parmed.openmm.load_topology(top, system=system, xyz=positions)

    # keep identity tidy for later merging/writing
    if structure.residues:
        structure.residues[0].name = resname
        try:
            structure.residues[0].number = int(residue_id)
        except Exception:
            pass
        try:
            structure.residues[0].chain = chain_id
        except Exception:
            pass

    # metadata for later restraint building
    structure.ion_name = ion_name
    structure.coordination_number = coordination_number

    return structure




def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > eps else np.zeros(3, dtype=float)


def _kabsch_rotation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Return R (3x3) such that (A @ R.T) best matches B (least squares), row-vector convention.
    A,B shape (k,3) with vectors from origin.
    """
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    return R


def _rodrigues_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Rotation matrix R that maps unit vector a -> b (row-vector convention uses v @ R.T).
    """
    a = _unit(a)
    b = _unit(b)
    if np.allclose(a, 0) or np.allclose(b, 0):
        return np.eye(3)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < 1e-12:
        # a and b are parallel or anti-parallel
        if c > 0:
            return np.eye(3)
        # 180-degree flip around any axis orthogonal to a
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(ref, a))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a, ref))
        # Rotation by pi: R = I + 2*K^2 where K is skew(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=float)
        return np.eye(3) + 2.0 * (K @ K)

    axis = v / s
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=float)

    # Rodrigues: R = I + sinθ K + (1-cosθ) K^2 ; here cosθ=c, sinθ=s
    R = np.eye(3) + s * K + (1.0 - c) * (K @ K)
    return R


def _rot_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _unit(axis)
    if np.allclose(axis, 0):
        return np.eye(3)
    x, y, z = axis
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    C = 1.0 - c
    # Standard axis-angle rotation
    R = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
    ], dtype=float)
    return R


def _coord_geometry_templates() -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Return:
      templates[name] -> (CN,3) unit directions
      cn_map[name] -> CN
    Names are canonical lower-case keys.
    """
    s3 = np.sqrt(3.0)
    s5 = np.sin(np.deg2rad(72.0))
    c5 = np.cos(np.deg2rad(72.0))

    templates = {}

    # CN=6 Octahedral
    templates["octahedral"] = np.array([
        [ 1, 0, 0], [-1, 0, 0],
        [ 0, 1, 0], [ 0,-1, 0],
        [ 0, 0, 1], [ 0, 0,-1],
    ], dtype=float)

    # CN=4 Tetrahedral
    V = np.array([[ 1,  1,  1],
                  [ 1, -1, -1],
                  [-1,  1, -1],
                  [-1, -1,  1]], dtype=float)
    templates["tetrahedral"] = np.array([_unit(v) for v in V], dtype=float)

    # CN=4 Square Planar (in xy plane)
    templates["square planar"] = np.array([
        [ 1, 0, 0], [-1, 0, 0],
        [ 0, 1, 0], [ 0,-1, 0],
    ], dtype=float)

    # CN=2 Linear (+/- z)
    templates["linear"] = np.array([[0, 0, 1], [0, 0, -1]], dtype=float)

    # CN=3 Trigonal Planar (xy plane)
    templates["trigonal planar"] = np.array([
        [ 1, 0, 0],
        [-0.5,  0.5*s3, 0],
        [-0.5, -0.5*s3, 0],
    ], dtype=float)

    # CN=5 Trigonal Bipyramidal: axial +/-z, equatorial 120 in xy
    templates["trigonal bipyramidal"] = np.array([
        [ 0, 0,  1],
        [ 0, 0, -1],
        [ 1, 0,  0],
        [-0.5,  0.5*s3, 0],
        [-0.5, -0.5*s3, 0],
    ], dtype=float)

    # CN=5 Square Pyramidal: square planar + +z
    templates["square pyramidal"] = np.array([
        [ 1, 0, 0], [-1, 0, 0],
        [ 0, 1, 0], [ 0,-1, 0],
        [ 0, 0, 1],
    ], dtype=float)

    # CN=7 Pentagonal Bipyramidal: axial +/-z, 5 equatorial (72° spacing) in xy
    eq5 = []
    for k in range(5):
        ang = np.deg2rad(72.0 * k)
        eq5.append([np.cos(ang), np.sin(ang), 0.0])
    templates["pentagonal bipyramidal"] = np.array(
        [[0, 0,  1], [0, 0, -1]] + eq5, dtype=float
    )

    # Normalize everything
    for k, arr in list(templates.items()):
        templates[k] = np.array([_unit(v) for v in arr], dtype=float)

    cn_map = {k: templates[k].shape[0] for k in templates}
    return templates, cn_map


def _normalize_geom_name(name: str) -> str:
    n = " ".join(str(name).strip().lower().split())

    # support your common typos
    alias = {
        "triganal planer": "trigonal planar",
        "triganal planar": "trigonal planar",
        "triganol planar": "trigonal planar",

        "square pyrimidal": "square pyramidal",
        "square pyrimidal ": "square pyramidal",
        "square pyramidal": "square pyramidal",

        "pentagonal bipyrimidal": "pentagonal bipyramidal",
        "pentagonal bipyramidal": "pentagonal bipyramidal",

        "square planar": "square planar",
        "octahedral": "octahedral",
        "linear": "linear",
        "trigonal bipyramidal": "trigonal bipyramidal",
        "tetrahedral": "tetrahedral",
    }
    if n in alias:
        return alias[n]
    return n


def propose_dummy_water_oxygen_positions(
    ion_xyz: Sequence[float],
    donor_xyz: Sequence[Sequence[float]],
    coordination_geometry: str,
    ion_to_water_O_dist: float,
    obstacle_xyz: Sequence[Sequence[float]],
    min_obstacle_dist: float = 2.2,
    min_water_water_O_dist: float = 2.4,
    max_solutions: int = 10,
    n_axis_samples_if_underdetermined: int = 24,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """
    Propose dummy-water oxygen positions (Å) to complete the coordination shell.

    - Uses a coordination geometry template (Octahedral, Square Planar, etc.).
    - Uses existing donor atoms (protein/ligand) as fixed template sites.
    - Generates exactly CN - n_donors water oxygens.
    - Rejects candidates that clash with obstacle_xyz.

    Returns list of solutions sorted best-first.
    Each solution:
      - water_O_xyz: (n_missing,3)
      - cn: int
      - used_template_indices / missing_template_indices
      - fit_score: lower better (angular mismatch)
      - clearance: minimum distance from any proposed water-O to any obstacle
      - R: rotation matrix (row-vector convention v_rot = v @ R.T)
    """
    templates, cn_map = _coord_geometry_templates()
    geom = _normalize_geom_name(coordination_geometry)

    if geom not in templates:
        raise ValueError(f"Unknown coordination_geometry={coordination_geometry!r} (normalized to {geom!r}).")

    tmpl = templates[geom]
    cn = cn_map[geom]

    ion = np.asarray(ion_xyz, dtype=float).reshape(3)
    donors = np.asarray(donor_xyz, dtype=float).reshape(-1, 3) if len(donor_xyz) else np.zeros((0, 3), float)
    obstacles = np.asarray(obstacle_xyz, dtype=float).reshape(-1, 3) if len(obstacle_xyz) else np.zeros((0, 3), float)

    k = donors.shape[0]
    if k >= cn:
        # Nothing to add (or over-coordinated)
        return [{
            "water_O_xyz": np.zeros((0, 3), dtype=float),
            "cn": cn,
            "used_template_indices": tuple(range(min(k, cn))),
            "missing_template_indices": tuple(),
            "fit_score": 0.0,
            "clearance": float("inf"),
            "R": np.eye(3),
        }]

    donor_dirs = np.array([_unit(d - ion) for d in donors], dtype=float)  # (k,3)
    n_missing = cn - k

    rng = np.random.default_rng(seed)
    sols: List[Dict[str, Any]] = []

    def clash_ok(wO: np.ndarray) -> Tuple[bool, float]:
        # obstacle clearance
        clearance = float("inf")
        if len(obstacles):
            d = wO[:, None, :] - obstacles[None, :, :]
            clearance = float(np.min(np.linalg.norm(d, axis=2)))
            if clearance < min_obstacle_dist:
                return False, clearance

        # water-water O spacing
        if wO.shape[0] >= 2:
            for i in range(wO.shape[0]):
                for j in range(i + 1, wO.shape[0]):
                    if float(np.linalg.norm(wO[i] - wO[j])) < min_water_water_O_dist:
                        return False, clearance

        return True, clearance

    def emit_solution(R: np.ndarray, used_idx: Tuple[int, ...]) -> None:
        all_idx = tuple(range(cn))
        missing_idx = tuple(i for i in all_idx if i not in used_idx)

        miss_dirs = tmpl[list(missing_idx), :] @ R.T
        wO = ion[None, :] + ion_to_water_O_dist * miss_dirs

        ok, clearance = clash_ok(wO)
        if not ok:
            return

        if k > 0:
            fit_dirs = tmpl[list(used_idx), :] @ R.T
            dots = np.sum(fit_dirs * donor_dirs, axis=1)
            dots = np.clip(dots, -1.0, 1.0)
            fit_score = float(np.sum((1.0 - dots) ** 2))
        else:
            fit_score = 0.0

        sols.append({
            "water_O_xyz": wO,
            "cn": cn,
            "used_template_indices": used_idx,
            "missing_template_indices": missing_idx,
            "fit_score": fit_score,
            "clearance": clearance,
            "R": R,
        })

    # Case 0 donors: just try random rotations of the whole template
    if k == 0:
        # try identity + random rotations
        emit_solution(np.eye(3), ())
        for _ in range(max_solutions * 6):
            M = rng.normal(size=(3, 3))
            Q, _ = np.linalg.qr(M)
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            emit_solution(Q, ())
    # Case 1 donor: align one site to donor, then sample rotation around that axis
    elif k == 1:
        b = donor_dirs[0]
        for site in range(cn):
            a = tmpl[site]
            R0 = _rodrigues_align(a, b)  # maps a -> b
            used = (site,)

            # explore rotation around b to avoid clashes
            for t in range(max(1, n_axis_samples_if_underdetermined)):
                phi = 2.0 * np.pi * (t / max(1, n_axis_samples_if_underdetermined))
                Rax = _rot_about_axis(b, phi)
                R = Rax @ R0
                emit_solution(R, used)
    # Case >=2 donors: enumerate assignments and use Kabsch
    else:
        for used_sites in combinations(range(cn), k):
            for perm in permutations(used_sites, k):
                A = tmpl[list(perm), :]     # assigned template vectors
                B = donor_dirs              # donor directions
                R = _kabsch_rotation(A, B)
                emit_solution(R, tuple(perm))

    sols.sort(key=lambda s: (s["fit_score"], -s["clearance"]))
    return sols[:max_solutions]


_OH_DIST_A = 0.9572
_HOH_ANGLE_DEG = 104.52


def _unit_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros(3, dtype=float)
    return v / n


def _build_water_h_positions(
    o_xyz: np.ndarray,
    ion_xyz: Optional[np.ndarray] = None,
    oh_dist_a: float = _OH_DIST_A,
    hoh_angle_deg: float = _HOH_ANGLE_DEG,
) -> Tuple[np.ndarray, np.ndarray]:
    o_xyz = np.asarray(o_xyz, dtype=float).reshape(3)

    # Bisector direction: point Hs away from ion if provided
    if ion_xyz is not None:
        ion_xyz = np.asarray(ion_xyz, dtype=float).reshape(3)
        to_ion = _unit_vec(ion_xyz - o_xyz)
        bis = _unit_vec(-to_ion)
        if np.allclose(bis, 0.0):
            bis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        bis = np.array([0.0, 0.0, 1.0], dtype=float)

    # Build orthonormal frame around bisector
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(bis, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)

    a = _unit_vec(np.cross(bis, ref))
    c = _unit_vec(np.cross(a, bis))

    half = np.deg2rad(hoh_angle_deg) * 0.5
    v1 = bis * np.cos(half) + c * np.sin(half)
    v2 = bis * np.cos(half) - c * np.sin(half)

    h1 = o_xyz + oh_dist_a * v1
    h2 = o_xyz + oh_dist_a * v2
    return h1, h2


def create_parametrised_tip3p_water(
    o_xyz: Sequence[float],
    ion_xyz: Optional[Sequence[float]] = None,
    water_ff_xml: Optional[str] = None,
    residue_name: str = "HOH",
    chain_id: str = "W",
    resnum: int = 1,
    atom_names: Tuple[str, str, str] = ("O", "H1", "H2"),
) -> parmed.Structure:
    """
    Create a single TIP3P water as a ParmEd Structure, positioned at o_xyz (Å).

    The key detail is that the temporary OpenMM System used for ParmEd conversion
    must retain explicit bond/angle parameters for the water, otherwise the returned
    ParmEd Structure can end up with bond.type = None for the O-H bonds.
    """
    o_xyz = np.asarray(o_xyz, dtype=float).reshape(3)
    ion_xyz_arr = None if ion_xyz is None else np.asarray(ion_xyz, dtype=float).reshape(3)

    h1_xyz, h2_xyz = _build_water_h_positions(o_xyz, ion_xyz_arr)

    # Build an OpenMM Topology for one water
    top = app.Topology()
    chain = top.addChain(chain_id)
    res = top.addResidue(residue_name, chain, id=str(resnum))

    o_atom = top.addAtom(atom_names[0], app.element.oxygen, res)
    h1_atom = top.addAtom(atom_names[1], app.element.hydrogen, res)
    h2_atom = top.addAtom(atom_names[2], app.element.hydrogen, res)

    top.addBond(o_atom, h1_atom)
    top.addBond(o_atom, h2_atom)

    # Positions as Nx3 Quantity
    pos_a = np.vstack([o_xyz, h1_xyz, h2_xyz])   # Å, shape (3, 3)
    pos_nm = pos_a * 0.1                         # nm
    positions = pos_nm * unit.nanometer

    # Load a TIP3P FF definition
    ff_candidates = []
    if water_ff_xml is not None:
        ff_candidates.append(water_ff_xml)
    ff_candidates += [
        "tip3p.xml",
        "amber14/tip3p.xml",
    ]

    ff = None
    last_err = None
    for ffxml in ff_candidates:
        try:
            ff = app.ForceField(ffxml)
            break
        except Exception as e:
            last_err = e
            ff = None

    if ff is None:
        raise RuntimeError(
            f"Could not load a TIP3P ForceField XML. Tried: {ff_candidates}. "
            f"Last error: {last_err}"
        )

    # Build a temporary OpenMM System for parameter transfer into ParmEd.
    #
    # Preferred route:
    #   keep water rigid/constrained, but request explicit parameters for
    #   constrained DOFs so ParmEd can populate bond/angle types.
    #
    # Fallback route (older OpenMM):
    #   build the temporary system with flexible water so the harmonic
    #   bond/angle terms are present explicitly.
    try:
        system = ff.createSystem(
            top,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            rigidWater=True,
            flexibleConstraints=True,
        )
    except TypeError:
        # Older OpenMM versions may not support flexibleConstraints.
        # For the ParmEd conversion only, use a flexible water system so
        # explicit O-H bond and H-O-H angle parameters are present.
        system = ff.createSystem(
            top,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            rigidWater=False,
        )

    # Convert to ParmEd Structure with parameters
    water = parmed.openmm.load_topology(top, system, positions)

    # Small sanity check: fail early if parameters still did not come through
    bad_bonds = [b for b in water.bonds if getattr(b, "type", None) is None]
    if bad_bonds:
        raise RuntimeError(
            "TIP3P water conversion failed: one or more water bonds still have no BondType."
        )

    return water
