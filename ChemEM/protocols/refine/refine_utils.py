# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

import numpy as np
from scipy.optimize import minimize



def get_residue_positions(residue):
    pos = []
    for atom in residue.atoms:
        if atom.element <= 1:
            continue
        pos.append([atom.xx, atom.xy, atom.xz])
    return np.array(pos)


def get_residue_subset_from_points(points, complex_structure, distance_cutoff = 12.0):
    selected_residues = []
    selected_keys = set()
    cutoff2 = distance_cutoff * distance_cutoff
    
    for res in complex_structure.residues:
        atoms = list(res.atoms)
        if not atoms:
            continue
        
        res_xyz = np.array([[atom.xx, atom.xy, atom.xz] for atom in atoms], dtype=float)
        diffs = res_xyz[:, None, :] - points[None, :, :]
        d2 = np.sum(diffs * diffs, axis=-1)
        
        if np.any(d2 <= cutoff2):
            key = (str(res.chain), int(res.number), res.name)
            if key not in selected_keys:
                selected_keys.add(key)
                selected_residues.append(res)
        
    return create_structure_subset(
        complex_structure,
        selected_residues,
    )

def create_structure_subset(full_structure, residue_list):
    """Creates a new parmed structure from a residue subset."""
    all_atom_indices = set()
    for res in residue_list:
        all_atom_indices.update(a.idx for a in res.atoms)
    
    selection = f"@{','.join(str(i + 1) for i in sorted(all_atom_indices))}"
    return full_structure[selection]


#-----below is for ions refactor this shit 



_VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "MG": 1.73,
    "CA": 2.31,
    "ZN": 1.39,
    "MN": 1.61,
    "FE": 1.56,
    "CU": 1.40,
    "CO": 1.53,
    "NI": 1.63,
    "NA": 2.27,
    "K": 2.75,
}


def _safe_norm(v, eps=1e-12):
    n = np.linalg.norm(v)
    return max(n, eps)


def _unit_vector(v):
    v = np.asarray(v, dtype=float)
    return v / _safe_norm(v)


def _fibonacci_sphere(n=256):
    pts = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (2.0 * i) / max(n - 1, 1)
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        pts.append([x, y, z])
    return np.asarray(pts, dtype=float)


def _get_vdw_radius(symbol):
    return _VDW_RADII.get(str(symbol).upper(), 1.70)


def _default_target_distances(ion_name, n_spec):
    ion_name = str(ion_name).upper()
    fallback = {
        "MG": 2.10,
        "CA": 2.40,
        "ZN": 2.10,
        "MN": 2.20,
        "FE": 2.10,
        "CU": 2.05,
        "CO": 2.10,
        "NI": 2.05,
        "NA": 2.40,
        "K": 2.80,
    }
    d = fallback.get(ion_name, 2.20)
    return np.full(n_spec, d, dtype=float)


def _exclude_points_matching_spec(obstacle_xyz, obstacle_radii, spec_xyz, tol=1e-3):
    if len(obstacle_xyz) == 0 or len(spec_xyz) == 0:
        return obstacle_xyz, obstacle_radii

    keep = np.ones(len(obstacle_xyz), dtype=bool)
    for i, p in enumerate(obstacle_xyz):
        if np.any(np.linalg.norm(spec_xyz - p[None, :], axis=1) < tol):
            keep[i] = False

    return obstacle_xyz[keep], obstacle_radii[keep]


def _placement_objective(
    x,
    spec_xyz,
    target_dists,
    obstacle_xyz,
    obstacle_radii,
    ion_radius,
    centroid,
    clash_buffer=0.35,
    w_dist=1.0,
    w_center=0.02,
    w_clash=25.0,
):
    x = np.asarray(x, dtype=float)

    d_spec = np.linalg.norm(spec_xyz - x[None, :], axis=1)
    dist_term = np.mean((d_spec - target_dists) ** 2)

    center_term = np.sum((x - centroid) ** 2)

    clash_term = 0.0
    if len(obstacle_xyz):
        d_obs = np.linalg.norm(obstacle_xyz - x[None, :], axis=1)
        min_allowed = ion_radius + obstacle_radii - clash_buffer
        overlap = np.maximum(0.0, min_allowed - d_obs)
        clash_term = np.sum(overlap ** 2)

    return (w_dist * dist_term) + (w_center * center_term) + (w_clash * clash_term)


def _choose_shell_position_single_anchor(
    anchor,
    target_dist,
    obstacle_xyz,
    obstacle_radii,
    ion_radius,
    preferred_direction=None,
    n_samples=256,
    clash_buffer=0.35,
):
    sphere_dirs = _fibonacci_sphere(n_samples)

    if preferred_direction is not None:
        preferred_direction = _unit_vector(preferred_direction)

    best_pos = None
    best_score = np.inf

    for direction in sphere_dirs:
        p = anchor + target_dist * direction

        if len(obstacle_xyz):
            d_obs = np.linalg.norm(obstacle_xyz - p[None, :], axis=1)
            min_allowed = ion_radius + obstacle_radii - clash_buffer
            overlap = np.maximum(0.0, min_allowed - d_obs)
            clash_pen = np.sum(overlap ** 2)
            clearance = np.min(d_obs - min_allowed)
        else:
            clash_pen = 0.0
            clearance = 10.0

        dir_pen = 0.0
        if preferred_direction is not None:
            dir_pen = 0.1 * (1.0 - np.dot(direction, preferred_direction))

        score = (50.0 * clash_pen) - clearance + dir_pen

        if score < best_score:
            best_score = score
            best_pos = p

    return best_pos


def generate_initial_ion_position(
    spec_xyz,
    obstacle_xyz,
    obstacle_elements,
    ion_name,
    target_dists=None,
    exclude_spec_from_obstacles=True,
    spec_weights=None,
):
    """
    Generate an initial ion position in Å.

    Parameters
    ----------
    spec_xyz : array-like, shape (n_spec, 3)
        Coordinates of the coordinating atoms in Å.
    obstacle_xyz : array-like, shape (n_obs, 3)
        Coordinates of nearby atoms to avoid in Å.
    obstacle_elements : sequence of str, length n_obs
        Element symbols for obstacle atoms, e.g. ["C", "N", "O"].
        Hydrogens should already be excluded.
    ion_name : str
        Ion name, e.g. "MG", "ZN", "CA".
    target_dists : None or float or array-like
        Desired ion-to-spec distances in Å.
        - None: use simple ion-based fallback
        - float: same target distance for every spec atom
        - array-like: one per spec atom
    exclude_spec_from_obstacles : bool
        If True, remove obstacle atoms that coincide with spec atoms.

    Returns
    -------
    np.ndarray, shape (3,)
        Initial ion position in Å.
    """
    spec_xyz = np.asarray(spec_xyz, dtype=float)
    obstacle_xyz = np.asarray(obstacle_xyz, dtype=float)

    if spec_xyz.ndim != 2 or spec_xyz.shape[1] != 3:
        raise ValueError(f"spec_xyz must have shape (n, 3), got {spec_xyz.shape}")

    if len(spec_xyz) == 0:
        raise ValueError("spec_xyz is empty")

    if obstacle_xyz.size == 0:
        obstacle_xyz = np.zeros((0, 3), dtype=float)

    if obstacle_xyz.ndim != 2 or obstacle_xyz.shape[1] != 3:
        raise ValueError(f"obstacle_xyz must have shape (m, 3), got {obstacle_xyz.shape}")

    if len(obstacle_elements) != len(obstacle_xyz):
        raise ValueError(
            f"obstacle_elements length ({len(obstacle_elements)}) does not match "
            f"number of obstacle coordinates ({len(obstacle_xyz)})"
        )

    if target_dists is None:
        target_dists = _default_target_distances(ion_name, len(spec_xyz))
    elif np.isscalar(target_dists):
        target_dists = np.full(len(spec_xyz), float(target_dists), dtype=float)
    else:
        target_dists = np.asarray(target_dists, dtype=float)
        if target_dists.shape != (len(spec_xyz),):
            raise ValueError(
                f"target_dists must have shape ({len(spec_xyz)},), got {target_dists.shape}"
            )

    obstacle_radii = np.array([_get_vdw_radius(e) for e in obstacle_elements], dtype=float)
    ion_radius = _get_vdw_radius(ion_name)

    if exclude_spec_from_obstacles:
        obstacle_xyz, obstacle_radii = _exclude_points_matching_spec(
            obstacle_xyz, obstacle_radii, spec_xyz
        )

    if spec_weights is not None:
        w = np.asarray(spec_weights, dtype=float)
        w = w / w.sum()
        centroid = np.sum(spec_xyz * w[:, None], axis=0)
    else:
        centroid = np.mean(spec_xyz, axis=0)

    if len(spec_xyz) == 1:
        anchor = spec_xyz[0]
        target_dist = float(target_dists[0])

        if len(obstacle_xyz):
            vecs = anchor[None, :] - obstacle_xyz
            weights = 1.0 / np.maximum(np.linalg.norm(vecs, axis=1), 1e-3) ** 2
            preferred_direction = np.sum(vecs * weights[:, None], axis=0)
        else:
            preferred_direction = np.array([1.0, 0.0, 0.0], dtype=float)

        return _choose_shell_position_single_anchor(
            anchor=anchor,
            target_dist=target_dist,
            obstacle_xyz=obstacle_xyz,
            obstacle_radii=obstacle_radii,
            ion_radius=ion_radius,
            preferred_direction=preferred_direction,
        )

    x0 = centroid.copy()
    trial_points = [x0]

    # Shell guesses from each spec atom toward centroid
    for si in range(len(spec_xyz)):
        direction = centroid - spec_xyz[si]
        if np.linalg.norm(direction) < 1e-8:
            direction = np.array([1.0, 0.0, 0.0], dtype=float)
        shell_guess = _choose_shell_position_single_anchor(
            anchor=spec_xyz[si],
            target_dist=float(target_dists[si]),
            obstacle_xyz=obstacle_xyz,
            obstacle_radii=obstacle_radii,
            ion_radius=ion_radius,
            preferred_direction=direction,
            n_samples=128,
        )
        trial_points.append(shell_guess)

    # Fibonacci sphere samples around centroid at mean target distance
    mean_target_dist = float(np.mean(target_dists))
    fib_dirs = _fibonacci_sphere(64)
    fib_candidates = centroid[None, :] + mean_target_dist * fib_dirs

    # Score candidates by clash penalty and pick top 5
    fib_scores = []
    for fc in fib_candidates:
        score = _placement_objective(
            fc, spec_xyz, target_dists, obstacle_xyz,
            obstacle_radii, ion_radius, centroid,
        )
        fib_scores.append(score)
    fib_scores = np.array(fib_scores)
    top_fib_idx = np.argsort(fib_scores)[:5]
    for idx in top_fib_idx:
        trial_points.append(fib_candidates[idx])

    # Small random perturbations of centroid
    rng = np.random.default_rng(42)
    for _ in range(5):
        perturb = rng.normal(0.0, 0.5, size=3)
        trial_points.append(centroid + perturb)

    best_x = None
    best_fun = np.inf

    for start in trial_points:
        res = minimize(
            _placement_objective,
            x0=np.asarray(start, dtype=float),
            args=(
                spec_xyz,
                target_dists,
                obstacle_xyz,
                obstacle_radii,
                ion_radius,
                centroid,
            ),
            method="L-BFGS-B",
        )

        x = res.x if res.success else np.asarray(start, dtype=float)
        f = _placement_objective(
            x,
            spec_xyz,
            target_dists,
            obstacle_xyz,
            obstacle_radii,
            ion_radius,
            centroid,
        )

        if f < best_fun:
            best_fun = f
            best_x = x

    return np.asarray(best_x, dtype=float)



def find_atom_from_spec_by_coord_and_element(atom_spec, complex_structure, tol=1e-3):
    """
    Find the matching atom in a ParmEd complex_structure by comparing
    x, y, z coordinates and element.
    """
    target_xyz = np.asarray(atom_spec.get_point(), dtype=float)
    target_elem = atom_spec.get_element()

    for atom in complex_structure.atoms:
        atom_elem = atom.element_name.upper()
        atom_xyz = np.array([atom.xx, atom.xy, atom.xz], dtype=float)

        if atom_elem != target_elem:
            continue

        if np.allclose(atom_xyz, target_xyz, atol=tol, rtol=0.0):
            return atom

    raise RuntimeError(
        f"[ERROR] Could not find matching atom for element={target_elem} "
        f"at xyz={target_xyz.tolist()} within tol={tol}"
    )

def _default_k_ang_for_cn(cn):
    """Return a default k_ang scaled by coordination number."""
    defaults = {2: 600.0, 3: 600.0, 4: 600.0, 5: 500.0, 6: 400.0, 7: 400.0}
    return defaults.get(int(cn), 500.0)