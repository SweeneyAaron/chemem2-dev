"""
SR2 

-- Move function 1 new thing need fit in map rot vs trans steepest decent hybrid
--Move function 2 branch walker rebuilder 
--need to do avererage Q score improvements 
--also it looks like its just trying to fix a small branch of atoms not like the whole branch. 
--also need to add optional openmm refinement at the end of each thing 



for each ligand pose we first need avalible info 
-- atom positions 
-- torsions 
-- branches
-- best positions + scores
-- data for scoring (Mol copy?)



need to apply a scorer to this 

Global scorer -> CCC/MI/SCI 
Global Scorer -> Qscore
per atom -> Qscore 



once we have a starting thing, we do the 

Move function 1 with barriers to try and fit the ligand. 

Evaluate -> update Q scores, 
         -> find average Q scores 
         -> find branches with %below average Q scores 

Branch walker -> walk the branch in 15 degree moves save top n distinct + +- 15 degree local optimiser to optimise end branch atom. 
save top n 

Optional openMM minimistaion

Move function 1 with barriers to try and fit the ligand. 

Evaluate -> update Q scores, 
         -> fiad average Q scores 
         -> find branches with %below average Q scores 
Accept? 
Accept sub optimal by tempreture? 
iterate 
patiance 

 

"""
from dataclasses import dataclass
import os

import numpy as np
from scipy.spatial import cKDTree
from .smart_utils import (
    build_directional_torsion_walks,
    build_semantic_anchor_blocks,
    print_semantic_blocks,
    draw_semantic_blocks_grid,
)
from .optimisers import (
    fit_in_map,
    FitInMapConfig,
    FitInMapResult,
    protein_ligand_clash,
)
from .scorers import get_scorer, ScoreResult
from ChemEM.protocols.mapQ_score.mapq_utils import compute_qscores_from_emmap


@dataclass
class ProteinCoordinateIndex:
    coords_A: np.ndarray
    atom_indices: np.ndarray
    elements: np.ndarray
    tree: cKDTree

    def probe(self, ligand_coords_A, cutoff_A=9.0):
        ligand_coords_A = np.asarray(ligand_coords_A, dtype=np.float64)
        if ligand_coords_A.ndim != 2 or ligand_coords_A.shape[1] != 3:
            raise ValueError(
                f"ligand_coords_A must have shape (N, 3), got {ligand_coords_A.shape}"
            )
        if ligand_coords_A.shape[0] == 0 or self.coords_A.shape[0] == 0:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros(0, dtype=int),
            )

        hits = self.tree.query_ball_point(ligand_coords_A, r=float(cutoff_A))
        rows = sorted({int(row) for ligand_hits in hits for row in ligand_hits})
        if not rows:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros(0, dtype=int),
            )

        rows = np.asarray(rows, dtype=int)
        return self.coords_A[rows], self.atom_indices[rows]

    def elements_for_atom_indices(self, atom_indices):
        atom_indices = np.asarray(atom_indices, dtype=int).reshape(-1)
        if atom_indices.size == 0:
            return np.zeros(0, dtype=object)
        row_by_atom_index = {
            int(atom_index): row for row, atom_index in enumerate(self.atom_indices)
        }
        return np.asarray(
            [
                (
                    self.elements[row_by_atom_index[int(atom_index)]]
                    if int(atom_index) in row_by_atom_index
                    else "C"
                )
                for atom_index in atom_indices
            ],
            dtype=object,
        )


def build_protein_coordinate_index(protein_coords_A, atom_indices=None, elements=None):
    coords_A = np.asarray(protein_coords_A, dtype=np.float64)
    if coords_A.ndim != 2 or coords_A.shape[1] != 3:
        raise ValueError(f"protein_coords_A must have shape (N, 3), got {coords_A.shape}")

    if atom_indices is None:
        atom_indices = np.arange(coords_A.shape[0], dtype=int)
    else:
        atom_indices = np.asarray(atom_indices, dtype=int).reshape(-1)
        if atom_indices.shape[0] != coords_A.shape[0]:
            raise ValueError(
                "atom_indices must have one entry for each protein coordinate"
            )

    if elements is None:
        elements = np.full(coords_A.shape[0], "C", dtype=object)
    else:
        elements = np.asarray(elements, dtype=object).reshape(-1)
        if elements.shape[0] != coords_A.shape[0]:
            raise ValueError("elements must have one entry for each protein coordinate")

    return ProteinCoordinateIndex(
        coords_A=coords_A,
        atom_indices=atom_indices,
        elements=elements,
        tree=cKDTree(coords_A),
    )


def _atom_atomic_number(atom):
    value = getattr(atom, "atomic_number", None)
    if value is None:
        value = getattr(atom, "element", None)
    if hasattr(value, "atomic_number"):
        value = getattr(value, "atomic_number")
    try:
        return int(value)
    except Exception:
        text = str(getattr(atom, "element_name", "") or "").upper().strip()
        by_symbol = {
            "H": 1,
            "C": 6,
            "N": 7,
            "O": 8,
            "F": 9,
            "P": 15,
            "S": 16,
            "CL": 17,
            "BR": 35,
            "I": 53,
        }
        return by_symbol.get(text, 6)


def _residue_name_from_atom(atom):
    residue = getattr(atom, "residue", None)
    return str(getattr(residue, "name", "") or "").upper().strip()


def _ligand_residue_names(ligands):
    names = set()
    if ligands is None:
        return names
    if not isinstance(ligands, (list, tuple, set)):
        ligands = [ligands]
    for ligand in ligands or []:
        structure = getattr(ligand, "complex_structure", None)
        for residue in getattr(structure, "residues", []) or []:
            name = str(getattr(residue, "name", "") or "").upper().strip()
            if name:
                names.add(name)
    return names


def _is_protein_context_atom(atom, ligand_residue_names=None):
    if _atom_atomic_number(atom) <= 1:
        return False
    residue_name = _residue_name_from_atom(atom)
    if residue_name.startswith("LIG"):
        return False
    if ligand_residue_names and residue_name in ligand_residue_names:
        return False
    return True


class RefineLigand:
    def __init__(self, 
                ligand,
                protein_index=None,
                map_reference=None,
                cutoff_A=9.0):
        self._ligand = ligand 
        
        self._atom_positions = None 
        self._atom_elements = None
        self._atom_indices = None
        self._atom_row_by_mol_index = None
        self._map_reference = map_reference
        self._map_referece = map_reference
        self._protein_index = protein_index
        self.cutoff_A = float(cutoff_A)
        self._excluded_root_blocks = set()
        #init
        self._init_atoms()
        self._init_local_protein(protein_index)
        self._init_torsion_trees()
        self.update_atom_qscores()
        self.get_best_block_by_qscore()
    
    def _init_atoms(self):
        atom_positions = []
        atom_elements = []
        atom_indices = []
        pos = self._ligand.mol.GetConformer().GetPositions()
        for atom, pos in zip(self._ligand.mol.GetAtoms(), pos):
            
            elm = atom.GetSymbol()
            if elm != 'H':

                atom_indices.append(int(atom.GetIdx()))
                atom_elements.append(elm)
                atom_positions.append(pos)
        self._atom_positions = np.array(atom_positions)
        self._atom_elements = np.array(atom_elements)
        self._atom_indices = np.array(atom_indices, dtype=int)
        self._atom_row_by_mol_index = {
            int(atom_idx): row for row, atom_idx in enumerate(self._atom_indices)
        }


    def _init_local_protein(self, protein_index):
        if protein_index is None:
            self.local_coords_A = np.zeros((0, 3), dtype=np.float64)
            self.local_rows = np.zeros(0, dtype=int)
            self.local_elements = np.zeros(0, dtype=object)
            return
        self.local_coords_A, self.local_rows = protein_index.probe(self._atom_positions, cutoff_A=self.cutoff_A)
        self.local_elements = protein_index.elements_for_atom_indices(self.local_rows)
    
    def _init_torsion_trees(self):
        self._rotor_tree = build_semantic_anchor_blocks(self._ligand.mol)
        #debug
        #print_semantic_blocks(self._ligand.mol, self._rotor_tree)
        #draw_semantic_blocks_grid(self._ligand.mol, self._rotor_tree, filename=f"/Users/aaron.sweeney/Documents/chemem2_build/ChemEM2_feb26/chemem2-dev/ligand_semantic_blocks.png", mols_per_row=2)
        
    
    def get_best_block_by_qscore(self):
        excluded = getattr(self, "_excluded_root_blocks", set())
        block_q = []
        for block in self._rotor_tree:
            rows = [
                self._atom_row_by_mol_index[i]
                for i in block.atom_indices
                if i in self._atom_row_by_mol_index
            ]
            atom_q = [self._per_atom_qscores[i] for i in rows]
            block_q.append(float(np.mean(atom_q)) if atom_q else 0.0)

        self._block_qscores = block_q

        candidate_ids = [
            idx for idx in range(len(block_q))
            if int(self._rotor_tree[idx].block_id) not in excluded
        ] or list(range(len(block_q)))
        return max(candidate_ids, key=lambda idx: block_q[idx])
    
    def get_blocks_to_update(self):
        best_idx = self.get_best_block_by_qscore()
        best_q = float(self._block_qscores[best_idx])
        return [
            rot
            for num, rot in enumerate(self._rotor_tree)
            if num != best_idx and float(self._block_qscores[num]) < best_q
        ]

    def get_directional_torsion_walks(self):
        best_idx = self.get_best_block_by_qscore()
        blocks_to_update = self.get_blocks_to_update()
        return build_directional_torsion_walks(
            self._ligand.mol,
            self._rotor_tree,
            root_block_id=best_idx,
            target_block_ids=[block.block_id for block in blocks_to_update],
        )

    def qscore_context_coords_A(self, ligand_coords_A=None):
        if ligand_coords_A is None:
            ligand_coords_A = self._atom_positions
        ligand_coords_A = np.asarray(ligand_coords_A, dtype=np.float64)
        return np.concatenate(
            [ligand_coords_A, self.local_coords_A],
            axis=0,
        )

    def ligand_score_indices(self):
        return np.arange(self._atom_positions.shape[0], dtype=int)

    def update_atom_qscores(self, sigma_ref=0.6):
        if self._map_reference is None:
            self._per_atom_qscores = np.zeros(self._atom_positions.shape[0], dtype=np.float32)
            return
        
        qscores = compute_qscores_from_emmap(
            atoms_xyz=self.qscore_context_coords_A(),
            emmap=self._map_referece,
            sigma_ref=float(sigma_ref),
            radii=None,
            score_indices=self.ligand_score_indices(),
        )

        self._per_atom_qscores = qscores




class SmartRefine2:
    def __init__(self, system): 

        self.system = system 
        self.ligands = []
        self.scorer = "qscore"
        self.fit_config = FitInMapConfig(
            clash_mode="soft",  # "off", "soft", or "hard"
            debug=False,
            clash_diagnostics=False,
        )
        self.patience = 3
        
    def get_protein_complex(self):
        ligand_residue_names = _ligand_residue_names(getattr(self.system, "ligand", []))
        self._protein_atoms = [
            atom
            for atom in self.system.protein.complex_structure.atoms
            if _is_protein_context_atom(atom, ligand_residue_names)
        ]
        self._protein_positions = np.array([[atom.xx,atom.xy,atom.xz] for atom in self._protein_atoms])
        self._protein_elements = np.array(
            [getattr(atom, "element_name", None) or getattr(atom, "atomic_number", None) or atom.element for atom in self._protein_atoms],
            dtype=object,
        )
        self._protein_index = build_protein_coordinate_index(
            self._protein_positions,
            elements=self._protein_elements,
        )

    def get_refine_ligands(self):
        for lig in self.system.ligand:
            self.ligands.append(RefineLigand(lig,
                                            self._protein_index,
                                            self.system.density_map))
   

    def run(self):
        self.get_protein_complex()
        self.get_refine_ligands()
        
        self.fit_results = []
        self.debug_sdf_paths = []
        for ligand_index, ligand in enumerate(list(self.ligands), start=1):
            ligand = refine_ligand(
                ligand,
                scorer=self.scorer,
                fit_config=self.fit_config,
                patience=self.patience,
                debug_dir=getattr(self.system, "output", None),
            )
            result = ligand._last_fit_in_map_result
            self.fit_results.append(result)
            self.ligands[ligand_index - 1] = ligand
            self.debug_sdf_paths.append(
                self.write_refined_ligand_sdf(ligand, result, ligand_index)
            )

        return self.fit_results

    def write_refined_ligand_sdf(self, refine_ligand, result, ligand_index=1, filename=None):
        output_dir = getattr(self.system, "output", ".")
        if filename is None:
            filename = f"smart_refine_2_fit_in_map_ligand_{int(ligand_index):03d}.sdf"
        return write_refined_ligand_sdf(refine_ligand, result, output_dir, filename)

        

def refine_ligand(
    refine_ligand,
    scorer="qscore",
    fit_config=None,
    branch_config=None,
    min_score_improvement=0.0,
    max_clash_penalty_increase=1e-6,
    max_iters=10,
    patience=3,
    selection="branches",
    debug_dir=None,
):
    """SmartRefine2 single-ligand refinement loop.

    Each iteration runs:
        1. fit_in_map  (rigid-body refinement)
        2. accepter    (apply if improved)
        3. branch_walker (torsion search; emits N candidate poses)
        4. fit_in_map on each branch candidate, on a clone of refine_ligand
        5. pick the winner per ``selection``

    The block used as root in step 3 is excluded from being root in the next iteration.

    selection:
        "greedy"   -- winner picked from {base, branches}; never regresses
        "branches" -- winner picked from re-fit branches only
    """
    if selection not in {"greedy", "branches"}:
        raise ValueError(f"selection must be 'greedy' or 'branches', got {selection!r}")
    scorer = get_scorer(scorer)

    refine_ligand = _fit_and_accept(
        refine_ligand,
        scorer,
        fit_config,
        min_score_improvement,
        max_clash_penalty_increase,
    )
    patience_limit = _patience_limit(patience)
    best_raw_score_so_far = _result_best_raw(
        getattr(refine_ligand, "_last_fit_in_map_result", None)
    )
    iterations_completed = 0
    no_improve_iters = 0
    stop_reason = "max_iters"
    _set_refine_loop_diagnostics(
        refine_ligand,
        iterations_completed,
        no_improve_iters,
        stop_reason,
    )

    branch_dir = None
    iter_dir = None
    if debug_dir:
        branch_dir = os.path.join(debug_dir, "branches")
        os.makedirs(branch_dir, exist_ok=True)
        iter_dir = os.path.join(debug_dir, "iters")
        os.makedirs(iter_dir, exist_ok=True)

    for iteration in range(max_iters):
        #refine_ligand = _fit_and_accept(refine_ligand, scorer, fit_config, min_score_improvement)

        rotor_tree = getattr(refine_ligand, "_rotor_tree", None)
        if not rotor_tree or not hasattr(refine_ligand, "get_best_block_by_qscore"):
            stop_reason = "no_rotor_tree"
            break
        root_block_id = int(
            rotor_tree[refine_ligand.get_best_block_by_qscore()].block_id
        )

        branch_results = branch_walker(refine_ligand, scorer=scorer, config=branch_config)
        refine_ligand._last_branch_walker_result = branch_results
        if not branch_results:
            stop_reason = "no_branch_results"
            break

        refine_ligand = _refit_branch_candidates(
            refine_ligand, branch_results, scorer, fit_config,
            min_score_improvement, selection, max_clash_penalty_increase,
        )
        iterations_completed = iteration + 1

        latest_raw_score = _result_best_raw(
            getattr(refine_ligand, "_last_fit_in_map_result", None)
        )
        if latest_raw_score > best_raw_score_so_far + float(min_score_improvement):
            best_raw_score_so_far = latest_raw_score
            no_improve_iters = 0
        else:
            no_improve_iters += 1

        if patience_limit is not None and no_improve_iters >= patience_limit:
            stop_reason = "patience"

        # Rolling exclusion of size 1: the block we just walked away from
        # cannot be picked as root next iteration; it'll get walked then.
        refine_ligand._excluded_root_blocks = {root_block_id}
        _set_refine_loop_diagnostics(
            refine_ligand,
            iterations_completed,
            no_improve_iters,
            stop_reason,
        )

        #debug writer here!!!
        if branch_dir:
            iter_branch_dir = os.path.join(branch_dir, f"iter_{iteration}")
            os.makedirs(iter_branch_dir, exist_ok=True)
            _debug_write_branch_results(
                refine_ligand,
                branch_results,
                iter_branch_dir,
                iteration=iteration,
            )
        #add per iteration debugger here
        _debug_write_iteration_ligand(refine_ligand, iter_dir, iteration=iteration)
        if stop_reason == "patience":
            print(
                "[smart_refine_2] early stopping: "
                f"patience={patience_limit} no_improve_iters={no_improve_iters}"
            )
            break

    _set_refine_loop_diagnostics(
        refine_ligand,
        iterations_completed,
        no_improve_iters,
        stop_reason,
    )
    return refine_ligand


def _patience_limit(patience):
    if patience is None:
        return None
    limit = int(patience)
    return limit if limit > 0 else None


def _result_best_raw(result, default=float("-inf")):
    if result is None:
        return float(default)
    try:
        raw = float(result.best_raw_score)
    except Exception:
        return float(default)
    return raw if np.isfinite(raw) else float(default)


def _set_refine_loop_diagnostics(refine_ligand, iterations_completed, no_improve_iters, stop_reason):
    refine_ligand._sr2_iterations_completed = int(iterations_completed)
    refine_ligand._sr2_no_improve_iters = int(no_improve_iters)
    refine_ligand._sr2_stop_reason = str(stop_reason)


def _fit_and_accept(
    rl,
    scorer,
    config,
    min_score_improvement,
    max_clash_penalty_increase=1e-6,
):
    result = fit_in_map(rl, scorer=scorer, config=config)
    rl._last_fit_in_map_result = result
    _log_fit_in_map(result)
    return accepter(
        rl,
        result,
        min_score_improvement=min_score_improvement,
        max_clash_penalty_increase=max_clash_penalty_increase,
    )


def _refit_branch_candidates(
    base_rl,
    branch_results,
    scorer,
    config,
    min_score_improvement,
    selection,
    max_clash_penalty_increase=1e-6,
):
    """Run fit_in_map on a clone of base_rl for each branch_walker pose; pick the winner."""
    if not branch_results:
        return base_rl

    candidates = []
    for branch_result in branch_results:
        clone = _clone_refine_ligand(base_rl)
        clone = apply_refinement(clone, branch_result)
        fit_result = fit_in_map(clone, scorer=scorer, config=config)
        clone._last_fit_in_map_result = fit_result
        _log_fit_in_map(fit_result)
        clone = accepter(
            clone,
            fit_result,
            min_score_improvement=min_score_improvement,
            max_clash_penalty_increase=max_clash_penalty_increase,
        )
        candidates.append((clone, clone._last_fit_in_map_result))

    if selection == "greedy":
        candidates.append((base_rl, base_rl._last_fit_in_map_result))

    return _pick_best_ligand(candidates)


def _pick_best_ligand(candidates):
    winner, _ = min(
        candidates,
        key=lambda pair: (
            int(pair[1].best_clash_count),
            -float(pair[1].best_raw_score),
        ),
    )
    return winner


def _clone_refine_ligand(rl):
    """Shallow copy with its own RDKit mol and _atom_positions so mutations don't leak."""
    import copy
    from types import SimpleNamespace
    from rdkit import Chem

    clone = copy.copy(rl)
    clone._ligand = SimpleNamespace(mol=Chem.Mol(rl._ligand.mol))
    clone._atom_positions = np.asarray(rl._atom_positions, dtype=np.float64).copy()
    clone._excluded_root_blocks = set(getattr(rl, "_excluded_root_blocks", set()))
    return clone


def _log_fit_in_map(result):
    mode = dict(getattr(result, "score_terms", {}) or {}).get("fit_clash_mode", "?")
    translation = getattr(result, "best_translation_A", None)
    translation_norm = (
        float(np.linalg.norm(np.asarray(translation, dtype=np.float64)))
        if translation is not None
        else 0.0
    )
    print(
        "[smart_refine_2] fit-in-map "
        f"mode={mode} "
        f"raw {result.initial_raw_score:+.5f}->{result.best_raw_score:+.5f} "
        f"accepted_score {result.initial_objective:+.5f}->{result.best_objective:+.5f} "
        f"clashes {result.initial_clash_count}->{result.best_clash_count} "
        f"clash_penalty {result.initial_clash_penalty:.5f}->{result.best_clash_penalty:.5f} "
        f"translation={translation_norm:.3f}A"
    )
    worst_pairs = dict(getattr(result, "clash_terms", {}) or {}).get("worst_pairs") or []
    if worst_pairs:
        print(f"[smart_refine_2] worst clash pairs: {worst_pairs}")


def _debug_write_iteration_ligand(refine_ligand, output_dir, iteration=0):
    if not output_dir:
        return
    result = getattr(refine_ligand, "_last_fit_in_map_result", None)
    if result is None:
        return
    fname = f"refine_ligand_iter{int(iteration):03d}.sdf"
    try:
        write_refined_ligand_sdf(refine_ligand, result, output_dir, fname)
    except Exception as exc:
        print(f"[smart_refine_2] per-iteration debug write failed for {fname}: {exc}")


def _debug_write_branch_results(refine_ligand, results, output_dir, iteration=0):
    if not output_dir or not results:
        return
    for idx, r in enumerate(results):
        fname = f"branch_walker_iter{int(iteration):03d}_cand{int(idx):02d}.sdf"
        try:
            write_refined_ligand_sdf(refine_ligand, r, output_dir, fname)
        except Exception as exc:
            print(f"[smart_refine_2] branch-walker debug write failed for {fname}: {exc}")


@dataclass
class BranchWalkConfig:
    coarse_step_deg: float = 15.0
    coarse_keep_fraction: float = 0.20
    local_window_deg: float = 15.0
    local_step_deg: float = 3.0
    max_keep_per_step: int = 3
    clash_cutoff_A: float = 5.0
    max_vdw_overlap_A: float = 0.3
    max_hbond_overlap_A: float = 0.8
    similar_score_tol: float = 1e-3
    sigma_ref: float = 0.6


@dataclass
class _BranchCandidate:
    coords_A: np.ndarray
    dihedrals_deg: tuple
    frontier_score: float
    clash_count: int
    clash_penalty: float


def branch_walker(refine_ligand, scorer=None, config=None):
    """Walk torsions outward from the best block; return one FitInMapResult per surviving beam candidate.

    The walker keeps up to ``config.max_keep_per_step`` candidates per torsion step,
    scoring frontier atoms via Qscore against the local protein context with a
    relaxed VDW allowance. Returned results follow the FitInMapResult contract so
    the existing accepter/apply_refinement helpers work unchanged.
    """
    config = config or BranchWalkConfig()
    scorer = get_scorer(scorer)

    walks = (
        refine_ligand.get_directional_torsion_walks()
        if hasattr(refine_ligand, "get_directional_torsion_walks")
        else ()
    )
    refine_ligand._last_directional_torsion_walks = walks
    if not walks:
        print("[smart_refine_2] branch-walker: no walks available")
        return []

    base_coords = np.asarray(refine_ligand._atom_positions, dtype=np.float64).copy()
    base_score_result, base_clash = _full_ligand_evaluation(
        refine_ligand, scorer, base_coords, config
    )
    initial_raw = float(base_score_result.value)
    initial_objective = initial_raw if base_clash.count == 0 else float("-inf")

    total_steps = sum(len(walk.steps) for walk in walks)
    progress = _branch_progress(total_steps)

    results = []
    total_evals = 0
    try:
        for walk in walks:
            survivors, n_evals = _walk_beam_search(
                refine_ligand, walk, base_coords, config, progress=progress
            )
            total_evals += n_evals
            for cand in survivors:
                results.append(
                    _build_walker_result(
                        refine_ligand,
                        scorer,
                        walk,
                        cand,
                        base_score_result,
                        base_clash,
                        initial_raw,
                        initial_objective,
                        total_evals,
                        config,
                    )
                )
    finally:
        if progress is not None:
            progress.close()

    print(
        "[smart_refine_2] branch-walker produced "
        f"{len(results)} candidate results from {len(walks)} walks "
        f"({total_evals} evaluations)"
    )
    #import pdb 
    #pdb.set_trace()
    return results


def _walk_beam_search(refine_ligand, walk, base_coords, config, progress=None):
    initial = _BranchCandidate(
        coords_A=base_coords.copy(),
        dihedrals_deg=(),
        frontier_score=0.0,
        clash_count=0,
        clash_penalty=0.0,
    )
    beam = [initial]
    total_evals = 0

    for step in walk.steps:
        frontier_rows = _heavy_rows(refine_ligand, step.frontier_atom_indices)
        moved_rows, moved_mol_ids = _moved_heavy_rows(refine_ligand, step.moved_atom_indices)
        if frontier_rows.size == 0 or moved_rows.size == 0:
            if progress is not None:
                progress.update(1)
            continue
        frontier_elements = np.asarray(refine_ligand._atom_elements, dtype=object)[frontier_rows]

        next_cands = []
        for parent in beam:
            kept, n = _evaluate_step_for_parent(
                refine_ligand, step, parent,
                moved_rows, moved_mol_ids, frontier_rows, frontier_elements,
                config,
            )
            total_evals += n
            next_cands.extend(kept)

        if progress is not None:
            progress.set_postfix(
                walk=walk.walk_id,
                step=step.order,
                beam=len(next_cands),
                evals=total_evals,
            )
            progress.update(1)

        if not next_cands:
            break
        beam = _select_beam(next_cands, config)

    if beam and beam[0].dihedrals_deg == ():
        return [], total_evals
    return beam, total_evals


def _evaluate_step_for_parent(
    refine_ligand, step, parent,
    moved_rows, moved_mol_ids, frontier_rows, frontier_elements,
    config,
):
    from rdkit.Chem import rdMolTransforms

    mol = _clone_with_coords(refine_ligand, parent.coords_A)
    conf = mol.GetConformer(0)
    dihedral = tuple(int(x) for x in step.dihedral)
    current_angle = float(rdMolTransforms.GetDihedralDeg(conf, *dihedral))

    n_evals = 0
    coarse_cands = []
    for delta in np.arange(-180.0, 180.0, config.coarse_step_deg):
        new_angle = current_angle + float(delta)
        rdMolTransforms.SetDihedralDeg(conf, *dihedral, new_angle)
        cand = _candidate_from_conformer(
            refine_ligand, conf, parent, moved_rows, moved_mol_ids,
            frontier_rows, frontier_elements, new_angle, config,
        )
        coarse_cands.append(cand)
        n_evals += 1

    kept_coarse = _filter_coarse(coarse_cands, config)
    if not kept_coarse:
        return [], n_evals

    refined = []
    for coarse in kept_coarse:
        coarse_angle = float(coarse.dihedrals_deg[-1])
        local_cands = []
        for delta in np.arange(
            -config.local_window_deg,
            config.local_window_deg + 1e-9,
            config.local_step_deg,
        ):
            new_angle = coarse_angle + float(delta)
            rdMolTransforms.SetDihedralDeg(conf, *dihedral, new_angle)
            local_cands.append(
                _candidate_from_conformer(
                    refine_ligand, conf, parent, moved_rows, moved_mol_ids,
                    frontier_rows, frontier_elements, new_angle, config,
                )
            )
            n_evals += 1
        refined.append(_best_candidate(local_cands))

    return refined, n_evals


def _candidate_from_conformer(
    refine_ligand, conf, parent,
    moved_rows, moved_mol_ids, frontier_rows, frontier_elements,
    angle_deg, config,
):
    coords = parent.coords_A.copy()
    for row, mol_idx in zip(moved_rows.tolist(), moved_mol_ids.tolist()):
        p = conf.GetAtomPosition(int(mol_idx))
        coords[row, 0] = float(p.x)
        coords[row, 1] = float(p.y)
        coords[row, 2] = float(p.z)

    frontier_coords = coords[frontier_rows]
    frontier_score = _frontier_qscore(refine_ligand, coords, frontier_rows, config)
    clash = protein_ligand_clash(
        frontier_coords,
        frontier_elements,
        refine_ligand.local_coords_A,
        refine_ligand.local_elements,
        cutoff_A=config.clash_cutoff_A,
        max_vdw_overlap_A=config.max_vdw_overlap_A,
        max_hbond_overlap_A=config.max_hbond_overlap_A,
    )
    return _BranchCandidate(
        coords_A=coords,
        dihedrals_deg=parent.dihedrals_deg + (float(angle_deg),),
        frontier_score=float(frontier_score),
        clash_count=int(clash.count),
        clash_penalty=float(clash.penalty),
    )


def _filter_coarse(cands, config):
    if not cands:
        return []
    best_score = max(c.frontier_score for c in cands)
    threshold = best_score - config.coarse_keep_fraction * abs(best_score)
    best_clash_count = min(c.clash_count for c in cands)
    nonzero_penalties = [c.clash_penalty for c in cands if c.clash_count > 0]
    best_clash_penalty = min(nonzero_penalties) if nonzero_penalties else 0.0

    kept = [
        c for c in cands
        if c.frontier_score >= threshold
        and c.clash_count <= best_clash_count + 1
        and (best_clash_penalty == 0.0 or c.clash_penalty <= 2.0 * best_clash_penalty)
    ]
    if kept:
        return kept
    # Fallback: keep the single lowest-clash candidate so the walk has something to propagate.
    return [min(cands, key=lambda c: (c.clash_count, c.clash_penalty, -c.frontier_score))]


def _select_beam(cands, config):
    if not cands:
        return []
    seen = set()
    deduped = []
    for c in sorted(cands, key=lambda c: (c.clash_count, -c.frontier_score, c.clash_penalty)):
        key = tuple(round(float(a), 0) for a in c.dihedrals_deg)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
        if len(deduped) >= config.max_keep_per_step:
            break
    return deduped


def _best_candidate(cands):
    return min(cands, key=lambda c: (c.clash_count, -c.frontier_score, c.clash_penalty))


def _heavy_rows(refine_ligand, mol_indices):
    row_by_mol = refine_ligand._atom_row_by_mol_index
    rows = [int(row_by_mol[int(m)]) for m in mol_indices if int(m) in row_by_mol]
    return np.asarray(rows, dtype=int)


def _moved_heavy_rows(refine_ligand, mol_indices):
    row_by_mol = refine_ligand._atom_row_by_mol_index
    rows, mol_ids = [], []
    for m in mol_indices:
        mi = int(m)
        if mi in row_by_mol:
            rows.append(int(row_by_mol[mi]))
            mol_ids.append(mi)
    return np.asarray(rows, dtype=int), np.asarray(mol_ids, dtype=int)


def _clone_with_coords(refine_ligand, heavy_coords_A):
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    mol = Chem.Mol(refine_ligand._ligand.mol)
    conf = mol.GetConformer(0)
    for row, mol_idx in enumerate(refine_ligand._atom_indices.tolist()):
        x, y, z = heavy_coords_A[row]
        conf.SetAtomPosition(int(mol_idx), Point3D(float(x), float(y), float(z)))
    return mol


def _frontier_qscore(refine_ligand, ligand_coords_A, frontier_rows, config):
    if frontier_rows.size == 0:
        return 0.0
    map_reference = getattr(refine_ligand, "_map_reference", None) or getattr(
        refine_ligand, "_map_referece", None
    )
    if map_reference is None:
        return 0.0
    local = np.asarray(refine_ligand.local_coords_A, dtype=np.float64)
    atoms_xyz = (
        np.concatenate([ligand_coords_A, local], axis=0) if local.size else ligand_coords_A
    )
    q = compute_qscores_from_emmap(
        atoms_xyz=atoms_xyz,
        emmap=map_reference,
        sigma_ref=float(config.sigma_ref),
        radii=None,
        score_indices=np.asarray(frontier_rows, dtype=int),
    )
    q = np.asarray(q, dtype=np.float64)
    finite = q[np.isfinite(q)]
    return float(np.mean(finite)) if finite.size else 0.0


def _branch_progress(total_steps):
    if total_steps <= 0:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        print(f"[smart_refine_2] branch-walker: running {total_steps} torsion steps")
        return None
    return tqdm(total=int(total_steps), desc="branch-walker", unit="step", leave=False)


def _full_ligand_evaluation(refine_ligand, scorer, coords_A, config):
    """Score full ligand + check clashes for the FitInMapResult bridge."""
    score_out = scorer.score(refine_ligand, coords_A)
    score_result = (
        score_out
        if isinstance(score_out, ScoreResult)
        else ScoreResult(value=float(score_out), terms={})
    )
    clash = protein_ligand_clash(
        coords_A,
        refine_ligand._atom_elements,
        refine_ligand.local_coords_A,
        refine_ligand.local_elements,
        cutoff_A=config.clash_cutoff_A,
        max_vdw_overlap_A=config.max_vdw_overlap_A,
        max_hbond_overlap_A=config.max_hbond_overlap_A,
    )
    return score_result, clash


def _build_walker_result(
    refine_ligand, scorer, walk, candidate,
    base_score_result, base_clash,
    initial_raw, initial_objective, total_evals, config,
):
    score_result, clash = _full_ligand_evaluation(
        refine_ligand, scorer, candidate.coords_A, config
    )
    best_raw = float(score_result.value)
    best_objective = best_raw if clash.count == 0 else float("-inf")
    score_terms = {
        "walk_id": int(walk.walk_id),
        "block_route": tuple(int(x) for x in walk.block_route),
        "dihedrals_deg": tuple(float(a) for a in candidate.dihedrals_deg),
        "frontier_score": float(candidate.frontier_score),
        **dict(score_result.terms),
    }
    clash_terms = {
        "penalty": float(clash.penalty),
        "count": int(clash.count),
        "max_overlap_A": float(clash.max_overlap_A),
        "max_allowed_overlap_A": float(clash.max_allowed_overlap_A),
        "max_excess_overlap_A": float(clash.max_excess_overlap_A),
        "pairs_checked": int(clash.pairs_checked),
    }
    return FitInMapResult(
        best_coords_A=candidate.coords_A.copy(),
        initial_raw_score=float(initial_raw),
        best_raw_score=best_raw,
        initial_objective=float(initial_objective),
        best_objective=best_objective,
        initial_clash_penalty=float(base_clash.penalty),
        best_clash_penalty=float(clash.penalty),
        initial_clash_count=int(base_clash.count),
        best_clash_count=int(clash.count),
        best_max_overlap_A=float(clash.max_overlap_A),
        steps=int(len(walk.steps)),
        evaluations=int(total_evals),
        converged=True,
        final_step_size_A=float(config.local_step_deg),
        score_terms=score_terms,
        clash_terms=clash_terms,
        best_rotation_matrix=None,
        best_translation_A=None,
    )


def should_accept_refinement(
    result,
    min_score_improvement=0.0,
    max_clash_penalty_increase=1e-6,
):
    if not np.isfinite(float(result.best_raw_score)):
        return False
    if float(result.best_raw_score) <= (
        float(result.initial_raw_score) + float(min_score_improvement)
    ):
        return False

    initial_clashes = int(result.initial_clash_count)
    best_clashes = int(result.best_clash_count)
    if initial_clashes <= 0:
        if best_clashes > 0:
            return False
        return bool(np.isfinite(float(result.best_objective)))

    allowed_penalty = float(result.initial_clash_penalty) + float(
        max_clash_penalty_increase
    )
    if float(result.best_clash_penalty) > allowed_penalty:
        return False
    return True


def apply_refinement(refine_ligand, result):
    best_coords = np.asarray(result.best_coords_A, dtype=np.float64)
    if best_coords.shape != refine_ligand._atom_positions.shape:
        raise ValueError("best_coords_A must match RefineLigand heavy-atom coordinates")

    if hasattr(refine_ligand, "_ligand") and hasattr(refine_ligand._ligand, "mol"):
        refine_ligand._ligand.mol = refined_ligand_mol(refine_ligand, result)

    refine_ligand._atom_positions = best_coords.copy()
    protein_index = getattr(refine_ligand, "_protein_index", None)
    if protein_index is not None:
        refine_ligand._init_local_protein(protein_index)

    if hasattr(refine_ligand, "update_atom_qscores"):
        refine_ligand.update_atom_qscores()
    if hasattr(refine_ligand, "get_best_block_by_qscore"):
        refine_ligand.get_best_block_by_qscore()
    return refine_ligand


def accepter(
    refine_ligand,
    result,
    min_score_improvement=0.0,
    max_clash_penalty_increase=1e-6,
):
    if not should_accept_refinement(
        result,
        min_score_improvement=min_score_improvement,
        max_clash_penalty_increase=max_clash_penalty_increase,
    ):
        print(
            "[smart_refine_2] rejected fit-in-map "
            f"delta_raw={result.delta_raw_score:+.5f} "
            f"clashes {result.initial_clash_count}->{result.best_clash_count} "
            f"clash_penalty {result.initial_clash_penalty:.5f}->{result.best_clash_penalty:.5f}"
        )
        return refine_ligand

    print(
        "[smart_refine_2] accepted fit-in-map "
        f"delta_raw={result.delta_raw_score:+.5f} "
        f"clashes {result.initial_clash_count}->{result.best_clash_count} "
        f"clash_penalty {result.initial_clash_penalty:.5f}->{result.best_clash_penalty:.5f}"
    )
    return apply_refinement(refine_ligand, result)

def refined_ligand_mol(refine_ligand, result):
    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D
    except Exception as exc:
        raise RuntimeError("RDKit is required to write refined ligand SDF files") from exc

    mol = Chem.Mol(refine_ligand._ligand.mol)
    if mol.GetNumConformers() == 0:
        conf = Chem.Conformer(mol.GetNumAtoms())
        mol.AddConformer(conf, assignId=True)

    source_conf = mol.GetConformer(0)
    all_coords = np.asarray(source_conf.GetPositions(), dtype=np.float64)
    rotation = getattr(result, "best_rotation_matrix", None)
    translation = getattr(result, "best_translation_A", None)
    if rotation is not None and translation is not None:
        centroid = np.mean(np.asarray(refine_ligand._atom_positions, dtype=np.float64), axis=0)
        all_coords = (
            (all_coords - centroid) @ np.asarray(rotation, dtype=np.float64).T
        ) + centroid + np.asarray(translation, dtype=np.float64)

    best_coords = np.asarray(result.best_coords_A, dtype=np.float64)
    atom_indices = np.asarray(refine_ligand._atom_indices, dtype=int)
    if best_coords.shape[0] != atom_indices.shape[0]:
        raise ValueError("best_coords_A must have one row per ligand heavy atom")

    all_coords[atom_indices] = best_coords
    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    conf = Chem.Conformer(out.GetNumAtoms())
    for atom_idx, xyz in enumerate(all_coords):
        conf.SetAtomPosition(
            int(atom_idx),
            Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])),
        )
    out.AddConformer(conf, assignId=True)
    _refresh_hydrogen_positions_from_heavy_geometry(out, Chem)

    props = {
        "sr2_initial_raw_score": result.initial_raw_score,
        "sr2_best_raw_score": result.best_raw_score,
        "sr2_delta_raw_score": result.delta_raw_score,
        "sr2_initial_objective": result.initial_objective,
        "sr2_best_objective": result.best_objective,
        "sr2_delta_objective": result.delta_objective,
        "sr2_best_clash_excess_penalty": result.best_clash_penalty,
        "sr2_best_clash_count": result.best_clash_count,
        "sr2_steps": result.steps,
        "sr2_evaluations": result.evaluations,
    }
    for key, value in props.items():
        out.SetProp(key, str(value))
    return out


def _refresh_hydrogen_positions_from_heavy_geometry(mol, Chem):
    """Regenerate H coordinates while preserving the existing atom order."""
    if mol.GetNumConformers() == 0:
        return mol

    heavy_atom_indices = [
        int(atom.GetIdx()) for atom in mol.GetAtoms() if int(atom.GetAtomicNum()) > 1
    ]
    if not heavy_atom_indices:
        return mol
    if not any(int(atom.GetAtomicNum()) == 1 for atom in mol.GetAtoms()):
        return mol

    try:
        heavy_only = Chem.RemoveHs(Chem.Mol(mol))
        if heavy_only.GetNumAtoms() != len(heavy_atom_indices):
            return mol
        regenerated = Chem.AddHs(heavy_only, addCoords=True)
    except Exception:
        return mol
    if regenerated.GetNumConformers() == 0:
        return mol

    conf = mol.GetConformer(0)
    regenerated_conf = regenerated.GetConformer(0)
    for rebuilt_heavy_idx, original_heavy_idx in enumerate(heavy_atom_indices):
        original_hydrogens = _bonded_hydrogen_indices(mol, original_heavy_idx)
        if not original_hydrogens:
            continue

        regenerated_hydrogens = _bonded_hydrogen_indices(
            regenerated, rebuilt_heavy_idx
        )
        if len(original_hydrogens) != len(regenerated_hydrogens):
            continue

        for original_h_idx, regenerated_h_idx in zip(
            original_hydrogens, regenerated_hydrogens
        ):
            conf.SetAtomPosition(
                int(original_h_idx),
                regenerated_conf.GetAtomPosition(int(regenerated_h_idx)),
            )
    return mol


def _bonded_hydrogen_indices(mol, atom_idx):
    return sorted(
        int(neighbor.GetIdx())
        for neighbor in mol.GetAtomWithIdx(int(atom_idx)).GetNeighbors()
        if int(neighbor.GetAtomicNum()) == 1
    )


def write_refined_ligand_sdf(refine_ligand, result, output_dir, filename):
    try:
        from rdkit import Chem
    except Exception as exc:
        raise RuntimeError("RDKit is required to write refined ligand SDF files") from exc

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    mol = refined_ligand_mol(refine_ligand, result)
    writer = Chem.SDWriter(path)
    try:
        writer.write(mol)
    finally:
        writer.close()
    print(f"[smart_refine_2] wrote refined ligand SDF: {path}")
    return path
