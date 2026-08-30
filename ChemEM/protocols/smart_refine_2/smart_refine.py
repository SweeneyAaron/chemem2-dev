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
from contextlib import contextmanager
from dataclasses import dataclass
import os

import numpy as np
from scipy.spatial import cKDTree
from .smart_utils import (
    build_directional_torsion_walks,
    build_semantic_anchor_blocks,
    print_semantic_blocks,
    draw_semantic_blocks_grid,
    aliphatic_ring_systems,
    generate_ring_conformers,
    extract_exo_torsions,
)
from .optimisers import (
    fit_in_map,
    FitInMapConfig,
    FitInMapResult,
    combine_clash_results,
    ligand_self_clash,
    protein_ligand_clash,
)
from .scorers import get_scorer, ScoreResult
from ChemEM.protocols.mapQ_score.mapq_utils import compute_qscores_from_emmap


ChemEMSimulationSetup = None
MinimiseInPlace = None
submap_from_structure = None
update_global_positions = None
update_ligand_positions = None
get_residue_positions = None
get_residue_subset_from_points = None


@dataclass
class DriftGuardConfig:
    """Opt-in guards that keep SmartRefine2 anchored to the starting pose.

    ``centroid_trust`` down-weights candidates whose ligand centroid moves more
    than ``centroid_radius_A`` from the start (flat-bottom, ``centroid_k`` raw-
    score units per Angstrom beyond). ``envelope_gate`` rejects candidates whose
    fraction of atoms in above-threshold density drops more than
    ``envelope_slack`` below the starting coverage. Both default off, so the
    guard helpers are pure no-ops unless explicitly enabled.
    """
    centroid_trust: bool = False
    centroid_radius_A: float = 2.0
    centroid_k: float = 0.05
    envelope_gate: bool = False
    envelope_threshold_sigma: float = 1.0
    envelope_slack: float = 0.1

    @property
    def enabled(self) -> bool:
        return bool(self.centroid_trust or self.envelope_gate)


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
    def __init__(
        self,
        ligand,
        protein_index=None,
        map_reference=None,
        cutoff_A=9.0,
        qscore_candidate_dirs=128,
        freeze_block_qscore=0.0,
        exo_torsions_enabled=True,
        exo_step_deg=20.0,
        exo_min_downstream=3,
    ):
        self._ligand = ligand
        self._ligand_object = ligand

        self._atom_positions = None
        self._atom_elements = None
        self._atom_indices = None
        self._atom_row_by_mol_index = None
        self._map_reference = map_reference
        self._map_referece = map_reference
        self._protein_index = protein_index
        self.cutoff_A = float(cutoff_A)
        self._qscore_candidate_dirs = int(qscore_candidate_dirs)
        # Blocks whose mean per-atom Qscore is >= this threshold are treated as
        # already well-fit and frozen out of the torsion search (never become a
        # walk target), so the search only moves poorly-fit atoms. 0 disables.
        self._freeze_block_qscore = float(freeze_block_qscore)
        # Exocyclic ring-branch torsions: seeded as the first step of the branch
        # walk so the beam search explores (ring-bond swing -> downstream torsions)
        # jointly. _exo_step_deg sets the swing's coarse angular step.
        self._exo_torsions_enabled = bool(exo_torsions_enabled)
        self._exo_step_deg = float(exo_step_deg)
        self._exo_min_downstream = int(exo_min_downstream)
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
        # Exocyclic ring-branch torsions: the extra DOFs that let a substituent
        # attached to a ring (e.g. a phosphate tail off a ribose) swing about a
        # ring bond. Computed once; the exo walker reads it via getattr so older
        # RefineLigand instances degrade to a no-op.
        self._exo_torsions = extract_exo_torsions(
            self._ligand.mol, min_downstream=getattr(self, "_exo_min_downstream", 3)
        )
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
        # Freeze blocks that are already well-fit (>= threshold): they stay part
        # of the rigid frame and are never torsion-searched, so the walker only
        # perturbs the poorly-fit regions (e.g. an ambiguous phosphate tail).
        freeze = float(getattr(self, "_freeze_block_qscore", 0.0) or 0.0)
        return [
            rot
            for num, rot in enumerate(self._rotor_tree)
            if num != best_idx
            and float(self._block_qscores[num]) < best_q
            and (freeze <= 0.0 or float(self._block_qscores[num]) < freeze)
        ]

    def _warhead_block_id(self, warhead_row):
        """block_id of the rotor-tree block that owns the covalent warhead atom, or
        None. `build_directional_torsion_walks` protects the root block's atoms from
        moving, so rooting there keeps the bonded warhead fixed under all torsions."""
        try:
            warhead_mol_idx = int(self._atom_indices[int(warhead_row)])
        except Exception:
            return None
        for block in self._rotor_tree:
            if warhead_mol_idx in set(block.atom_indices):
                return int(block.block_id)
        return None

    def _drop_walks_moving_other_anchors(self, walks, root_warhead_row):
        """Remove torsion walks that would displace a covalently bonded atom other
        than the one the tree is rooted at.

        Rooting protects only the root block's atoms. With a crosslinker a torsion
        between the two bonded atoms would satisfy one bond while stretching the
        other, so those walks are dropped outright. No-op for a single bond, where
        rooting already protects the only warhead.
        """
        anchors = getattr(self, "_covalent_anchors", None) or []
        if len(anchors) < 2:
            return walks

        other_mol_idxs = set()
        for row, _xyz, _r0 in anchors:
            if int(row) == int(root_warhead_row):
                continue
            try:
                other_mol_idxs.add(int(self._atom_indices[int(row)]))
            except Exception:
                continue
        if not other_mol_idxs:
            return walks

        kept = []
        for walk in walks:
            moves_anchor = any(
                other_mol_idxs & set(int(i) for i in step.frontier_atom_indices)
                for step in walk.steps
            )
            if not moves_anchor:
                kept.append(walk)

        dropped = len(walks) - len(kept)
        if dropped:
            print(f"[covalent] smart-refine: dropped {dropped} torsion walk(s) that "
                  "would move a second covalently bonded atom.")
        return tuple(kept)

    def get_directional_torsion_walks(self):
        exo = (
            getattr(self, "_exo_torsions", None)
            if getattr(self, "_exo_torsions_enabled", False)
            else None
        )
        # Covalent ligand: root the torsion tree at the warhead block so the bonded
        # warhead is never displaced by a torsion; flex every other block
        # (target_block_ids=None -> all non-root blocks).
        warhead_row = getattr(self, "_covalent_warhead_row", None)
        if warhead_row is not None:
            root_block_id = self._warhead_block_id(warhead_row)
            if root_block_id is not None:
                walks = build_directional_torsion_walks(
                    self._ligand.mol,
                    self._rotor_tree,
                    root_block_id=root_block_id,
                    target_block_ids=None,
                    exo_torsions=exo,
                )
                return self._drop_walks_moving_other_anchors(walks, warhead_row)

        best_idx = self.get_best_block_by_qscore()
        blocks_to_update = self.get_blocks_to_update()
        return build_directional_torsion_walks(
            self._ligand.mol,
            self._rotor_tree,
            root_block_id=best_idx,
            target_block_ids=[block.block_id for block in blocks_to_update],
            exo_torsions=exo,
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
            candidate_dirs=int(getattr(self, "_qscore_candidate_dirs", 128)),
        )

        self._per_atom_qscores = qscores




_TAIL_AWARE_BUNDLE = {
    # option_name -> tail_aware_value
    # When --sr2-tail-aware is passed, the bundle sets each option to its
    # tail_aware_value ONLY IF the user did not pass that option explicitly.
    # The corresponding argparse defaults are `argparse.SUPPRESS`, so a
    # missing user value leaves the attribute absent from `options` and we
    # detect that via `hasattr`. An explicit `--sr2-foo 0` from the user
    # sets `options.sr2_foo = 0` and `hasattr(options, "sr2_foo")` is True,
    # so the user's 0 wins over the bundle default even when 0 happens to
    # match the argparse default.
    #
    # NOT in the bundle (intentional):
    # - metropolis_temp: stochastic exploration made trajectories
    #   seed-dependent (round 5 finding).
    # - coarse_keep_fraction and angular_diversity_sectors: resolved per
    #   ligand in SmartRefine2._adaptive_branch_config (round 10) based on
    #   the longest semantic-walk depth -- looser settings on long-tail
    #   ligands, tight defaults on compact ones.
    # Users who want them as static values can pass the individual
    # --sr2-branch-* flags explicitly.
    "sr2_branch_downstream_lookahead_weight": 0.0,
    "sr2_branch_max_keep_per_step":           8,
    "sr2_kick_tries":                         3,
    "sr2_patience":                           6,
    "sr2_root_tabu_size":                     4,
    # Composite clash/raw trade-off picker + accepter. Without this the
    # lexicographic (clash_count, -raw_score) picker sacrifices large raw
    # improvements to save a single clash, which manifests as wrong-rotamer
    # final poses on compact ligands.
    "sr2_clash_tradeoff_lambda":              0.05,
}


def _apply_tail_aware_bundle(options):
    """Mutate ``options`` in place to apply the tail-aware bundle defaults
    when ``--sr2-tail-aware`` was passed. Only fills in each field if the
    user did NOT pass it explicitly (detected via ``hasattr`` thanks to
    argparse SUPPRESS on the bundle flags)."""
    if options is None:
        return
    if not bool(getattr(options, "sr2_tail_aware", False)):
        return
    applied = []
    for name, tail_value in _TAIL_AWARE_BUNDLE.items():
        if not hasattr(options, name):
            setattr(options, name, tail_value)
            applied.append(f"{name}={tail_value!r}")
    if applied:
        print(
            "[smart_refine_2] --sr2-tail-aware bundle applied: "
            + ", ".join(applied)
            + ", selection=greedy"
        )


_ROBUST_BUNDLE = {
    # --sr2-robust canonical "robust pipeline" profile. Applied unconditionally
    # (robust means robust), so it overrides individual guard flags. Most values
    # already match the post-benchmark defaults; the distinct effect is enabling
    # pre-minimisation. Keep in sync with the argparse defaults in protocol_spec.
    "sr2_pre_minimise":          True,
    "sr2_selection":             "greedy",
    "sr2_centroid_trust":        True,
    "sr2_centroid_trust_radius": 5.0,
    "sr2_centroid_trust_k":      0.4,
    "sr2_envelope_gate":         True,
    "sr2_envelope_slack":        0.15,
    "sr2_freeze_block_qscore":   0.7,
    "sr2_exo_torsions":          True,
}


def _apply_robust_bundle(options):
    """Mutate ``options`` in place to apply the --sr2-robust profile. Unlike the
    tail-aware bundle this overrides explicit per-field flags, because robust is
    an all-or-nothing 'safe pipeline' switch."""
    if options is None:
        return
    if not bool(getattr(options, "sr2_robust", False)):
        return
    applied = []
    for name, value in _ROBUST_BUNDLE.items():
        setattr(options, name, value)
        applied.append(f"{name}={value!r}")
    print("[smart_refine_2] --sr2-robust bundle applied: " + ", ".join(applied))


class SmartRefine2:
    def __init__(self, system):

        self.system = system
        self.ligands = []
        self.scorer = "qscore"
        options = getattr(system, "options", None)
        # Apply --sr2-robust then --sr2-tail-aware overrides before reading any
        # of the bundled options so the cascaded reads below see the new values.
        _apply_robust_bundle(options)
        _apply_tail_aware_bundle(options)
        self.optimisation_score = getattr(
            options,
            "sr2_optimisation_score",
            getattr(options, "sr2_optimization_score", None),
        )
        self.optimisation_weights = getattr(
            options,
            "sr2_optimisation_weights",
            getattr(options, "sr2_optimization_weights", None),
        )
        self.acceptance_score = getattr(options, "sr2_acceptance_score", None)
        self.acceptance_weights = getattr(options, "sr2_acceptance_weights", None)
        self.qscore_candidate_dirs = int(
            getattr(options, "sr2_qscore_candidate_dirs", 128) or 128
        )
        # Freeze well-fit blocks (mean Qscore >= threshold) out of the torsion
        # search. 0 disables. Default 0.7. Explicit 0 from the user must win, so
        # don't use `or` (it would treat 0 as missing).
        _freeze = getattr(options, "sr2_freeze_block_qscore", 0.7)
        self.freeze_block_qscore = float(_freeze) if _freeze is not None else 0.7
        self.fit_config = FitInMapConfig(
            clash_mode="soft",  # "off", "soft", or "hard"
            debug=False,
            clash_diagnostics=False,
            max_steps=int(getattr(options, "sr2_fit_in_map_max_steps", 64) or 64),
            early_stop_tol=getattr(options, "sr2_fit_in_map_early_stop_tol", None),
        )
        # Opt-in tuning: all defaults below reproduce previous behaviour.
        coarse_keep_fraction = getattr(options, "sr2_branch_coarse_keep_fraction", None)
        angular_diversity = getattr(options, "sr2_branch_angular_diversity_sectors", None)
        downstream_lookahead = getattr(options, "sr2_branch_downstream_lookahead_weight", None)
        metropolis_temp = getattr(options, "sr2_branch_metropolis_temp", None)
        metropolis_decay = getattr(options, "sr2_branch_metropolis_decay", None)
        self.branch_config = BranchWalkConfig(
            coarse_step_deg=float(
                getattr(options, "sr2_branch_coarse_step_deg", 15.0) or 15.0
            ),
            max_keep_per_step=int(
                getattr(options, "sr2_branch_max_keep_per_step", 3) or 3
            ),
            coarse_keep_fraction=(
                float(coarse_keep_fraction) if coarse_keep_fraction is not None else 0.20
            ),
            angular_diversity_sectors=(
                int(angular_diversity) if angular_diversity is not None else 0
            ),
            downstream_lookahead_weight=(
                float(downstream_lookahead) if downstream_lookahead is not None else 0.0
            ),
            metropolis_temp=(
                float(metropolis_temp) if metropolis_temp is not None else None
            ),
            metropolis_decay=(
                float(metropolis_decay) if metropolis_decay is not None else 0.7
            ),
            clash_tradeoff_lambda=(
                float(getattr(options, "sr2_branch_clash_tradeoff_lambda", None))
                if getattr(options, "sr2_branch_clash_tradeoff_lambda", None) is not None
                else None
            ),
        )
        outer_lambda = getattr(options, "sr2_clash_tradeoff_lambda", None)
        self.clash_tradeoff_lambda = (
            float(outer_lambda) if outer_lambda is not None else None
        )
        self.kick_config = KickConfig(
            tries=int(getattr(options, "sr2_kick_tries", 0) or 0),
            qscore_threshold=float(
                getattr(options, "sr2_kick_qscore_threshold", 0.5) or 0.5
            ),
            stagnation_tol=float(
                getattr(options, "sr2_kick_stagnation_tol", 1e-3) or 1e-3
            ),
            jitter_deg=float(
                getattr(options, "sr2_kick_jitter_deg", 30.0) or 30.0
            ),
            seed=getattr(options, "sr2_kick_seed", None),
        )
        self.patience = int(getattr(options, "sr2_patience", 3) or 3)
        self.root_tabu_size = int(getattr(options, "sr2_root_tabu_size", 1) or 1)
        # selection mode for branch-candidate picking. "branches" picks the
        # best of the re-fit branches only; "greedy" includes the current
        # pose in the pool so a worse-than-base set of branches cannot force
        # a per-iteration regression. Greedy is now the DEFAULT (benchmark
        # showed "branches" regressions outweigh its exploration benefit at
        # the <=2A metric); it pairs with the end-of-loop no-regression gate.
        # Opt back into the old behaviour with --sr2-selection branches.
        selection_opt = getattr(options, "sr2_selection", None)
        if selection_opt in {"greedy", "branches"}:
            self.selection = selection_opt
        else:
            self.selection = "greedy"
        # Round-5 polish placement: opt-in only. The on-stall polish needs
        # --sr2-polish-on-stall; the post-loop final polish needs
        # --sr2-final-minimise. Either can be globally killed by
        # --sr2-no-polish. See plan history for the round-3/4 experiments
        # that motivated demoting polish back to opt-in.
        no_polish = bool(getattr(options, "sr2_no_polish", False))
        self.polish_on_stall_enabled = (
            bool(getattr(options, "sr2_polish_on_stall", False)) and not no_polish
        )
        # Opt-in pre-minimisation: run one OpenMM local-refine minimisation on
        # each ligand before the search loop (see run()). Default OFF.
        self.pre_minimise_enabled = bool(
            getattr(options, "sr2_pre_minimise", False)
        )
        # OpenMM polish strategy: "standard" (single minimisation) or "fragment"
        # (iterative fragment-pinned minimiser). Opt-in via --sr2-minimiser.
        self.sr2_minimiser = str(getattr(options, "sr2_minimiser", "standard") or "standard")
        self.fragmin_block_tol = float(
            getattr(options, "sr2_fragmin_block_tol", 0.1) or 0.1
        )
        self.fragmin_max_iters = int(
            getattr(options, "sr2_fragmin_max_iters", 4) or 4
        )
        # Aliphatic-ring conformer sampling. Default ON, but only ever fires for
        # ligands that actually have a poorly-fit aliphatic ring system.
        self.ring_flex_enabled = bool(getattr(options, "sr2_ring_flex", True))
        self.ring_flex_confs = int(getattr(options, "sr2_ring_flex_confs", 8) or 8)
        self.exo_torsions_enabled = bool(getattr(options, "sr2_exo_torsions", True))
        self.exo_step_deg = float(getattr(options, "sr2_exo_step_deg", 20.0) or 20.0)
        self.exo_min_downstream = int(
            getattr(options, "sr2_exo_min_downstream", 3) or 3
        )
        self.ring_flex_rmsd = float(
            getattr(options, "sr2_ring_flex_rmsd", 0.3) or 0.3
        )
        # Anti-drift guards (centroid trust + density-envelope gate), anchored
        # to the starting pose. Default ON (tight radius / strong k) to stop the
        # torsion search drifting into neighbouring density on flexible ligands.
        # The getattr fallbacks mirror the argparse defaults so options built
        # without these attrs (e.g. the orchestrator) get the same behaviour.
        self.drift_guard = DriftGuardConfig(
            centroid_trust=bool(getattr(options, "sr2_centroid_trust", True)),
            centroid_radius_A=float(
                getattr(options, "sr2_centroid_trust_radius", 5.0) or 5.0
            ),
            centroid_k=float(getattr(options, "sr2_centroid_trust_k", 0.4) or 0.4),
            envelope_gate=bool(getattr(options, "sr2_envelope_gate", True)),
            envelope_threshold_sigma=float(
                getattr(options, "sr2_envelope_threshold_sigma", 1.0) or 1.0
            ),
            envelope_slack=float(
                getattr(options, "sr2_envelope_slack", 0.15) or 0.15
            ),
        )

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
            rl = RefineLigand(
                lig,
                self._protein_index,
                self.system.density_map,
                qscore_candidate_dirs=self.qscore_candidate_dirs,
                freeze_block_qscore=self.freeze_block_qscore,
                exo_torsions_enabled=self.exo_torsions_enabled,
                exo_step_deg=self.exo_step_deg,
                exo_min_downstream=self.exo_min_downstream,
            )
            # Covalent ligands: keep the warhead tethered to its residue during the
            # Q-score search (no-op for non-covalent ligands).
            _install_covalent_anchor_on_rl(rl, self.system.protein.complex_structure)
            self.ligands.append(rl)

    def _final_minimise_enabled(self):
        options = getattr(self.system, "options", None)
        if bool(getattr(options, "sr2_no_polish", False)):
            return False
        return bool(getattr(options, "sr2_final_minimise", False))

    def _refresh_refine_ligand_contexts(self):
        for ligand in self.ligands:
            if hasattr(ligand, "_init_local_protein"):
                ligand._protein_index = self._protein_index
                ligand._init_local_protein(self._protein_index)

    def _acceptance_scorer(self):
        """The same acceptance scorer the refine loop uses, for gating polishes."""
        _, acceptance = _resolve_refine_scorers(
            self.scorer,
            optimisation_scorer=self.optimisation_score,
            optimisation_weights=self.optimisation_weights,
            acceptance_scorer=self.acceptance_score,
            acceptance_weights=self.acceptance_weights,
            scorer_options=getattr(self.system, "options", None),
        )
        return acceptance

    def _pose_objective(self, refine_ligand, scorer, coords):
        """Acceptance objective of a pose, via the shared _objective_from_eval so
        it is in the same units as every other gate (soft clash penalty by
        default, not a -inf sentinel -- a clashing pose must stay rankable or the
        polish and no-regression gates degenerate into tautologies). Returns None
        when the pose cannot be scored (no usable map / scorer error) so the caller
        skips gating rather than crashing or reverting on a meaningless score."""
        try:
            score_result, clash = _full_ligand_evaluation(
                refine_ligand, scorer, np.asarray(coords, dtype=np.float64), self.fit_config
            )
            value = float(score_result.value)
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return _objective_from_eval(
            value, clash, self.fit_config,
            int(np.asarray(refine_ligand._atom_positions).shape[0]),
        )

    def _polish(self, refine_ligand, result, *, context_label):
        """Dispatch to the configured OpenMM polish (standard or fragment-pinned).
        Both share the same (refine_ligand, result) contract and re-score result."""
        scorer = self._acceptance_scorer()
        common = dict(
            system=self.system,
            protein_index_refresh_cb=self._refresh_protein_context,
            options=getattr(self.system, "options", None),
            context_label=context_label,
            acceptance_scorer=scorer,
            eval_config=self.fit_config,
        )
        if self.sr2_minimiser == "fragment":
            return fragment_pinned_polish_ligand(
                refine_ligand, result,
                pose_objective_fn=self._pose_objective,
                block_tol=self.fragmin_block_tol,
                max_iters=self.fragmin_max_iters,
                **common,
            )
        return local_refine_polish_ligand(refine_ligand, result, **common)

    def _final_map_minimise_ligand(self, refine_ligand, result):
        if not self._final_minimise_enabled():
            return refine_ligand, result
        # The polish mutates refine_ligand AND result in place, so snapshot the
        # pre-polish pose/objective before the call to support a revert. An
        # unconstrained OpenMM minimisation can walk the pose off the density
        # basin; gate it on the acceptance objective so a worsening polish is
        # discarded. (The fragment minimiser also self-reverts; the gate is then
        # a harmless no-op.)
        scorer = self._acceptance_scorer()
        pre_objective = self._pose_objective(
            refine_ligand, scorer, refine_ligand._atom_positions
        )
        pre_result = _snapshot_fit_result(result)
        polished_rl, polished_result = self._polish(
            refine_ligand, result, context_label="final polish"
        )
        post_objective = self._pose_objective(
            polished_rl, scorer, polished_rl._atom_positions
        )
        # Skip gating when either pose couldn't be scored (no map / scorer error).
        gate_ok = (
            pre_result is not None
            and pre_objective is not None
            and post_objective is not None
        )
        if gate_ok and post_objective < pre_objective - _NO_REGRESSION_EPS:
            apply_refinement(polished_rl, pre_result)
            polished_rl._last_fit_in_map_result = pre_result
            print(
                "[smart_refine_2] final polish reverted: objective "
                f"{post_objective:.5f} < pre-polish {pre_objective:.5f}"
            )
            return polished_rl, pre_result
        return polished_rl, polished_result

    def _refresh_protein_context(self):
        self.get_protein_complex()
        self._refresh_refine_ligand_contexts()

    def _build_polish_cb(self):
        if not self.polish_on_stall_enabled:
            return None

        def _polish(rl, result):
            return self._polish(rl, result, context_label="stall polish")

        return _polish

    def get_output(self):
        self.output = os.path.join(self.system.output, 'smart_refine')
        os.makedirs(self.output, exist_ok=True)

    def _adaptive_branch_config(self, refine_ligand):
        """Return a possibly-modified copy of ``self.branch_config`` for this
        ligand. When the ligand's longest directional walk exceeds
        ``--sr2-tail-aware-rotor-threshold`` (default 12) dihedrals, bump
        ``coarse_keep_fraction`` to 0.40 and ``angular_diversity_sectors`` to
        6 so the walker has the beam diversity needed to find deep low-clash
        basins on long-tail ligands. Compact ligands stay with the tighter
        defaults that work better for them. User-explicit
        --sr2-branch-coarse-keep-fraction or
        --sr2-branch-angular-diversity-sectors values win (detected via
        ``hasattr`` on options).
        """
        import copy as _copy

        bc = _copy.copy(self.branch_config)
        options = getattr(self.system, "options", None)
        if options is None:
            return bc

        rotor_tree = getattr(refine_ligand, "_rotor_tree", None)
        mol = getattr(getattr(refine_ligand, "_ligand", None), "mol", None)
        if not rotor_tree or mol is None:
            return bc

        threshold = int(
            getattr(options, "sr2_tail_aware_rotor_threshold", 12) or 12
        )

        try:
            root_idx = refine_ligand.get_best_block_by_qscore()
            root_block_id = int(rotor_tree[root_idx].block_id)
            target_ids = [
                int(b.block_id) for b in rotor_tree
                if int(b.block_id) != root_block_id
            ]
            walks = build_directional_torsion_walks(
                mol,
                rotor_tree,
                root_block_id=root_block_id,
                target_block_ids=target_ids,
            )
            max_depth = max((len(w.steps) for w in walks), default=0)
        except Exception as exc:
            print(
                f"[smart_refine_2] adaptive branch_config: depth probe failed "
                f"({exc}); using static config"
            )
            return bc

        if max_depth <= threshold:
            print(
                f"[smart_refine_2] adaptive branch_config: max walk depth "
                f"{max_depth} <= threshold {threshold} -> tight defaults "
                f"(coarse_keep={bc.coarse_keep_fraction}, "
                f"angular_diversity={bc.angular_diversity_sectors})"
            )
            return bc

        applied = []
        if not hasattr(options, "sr2_branch_coarse_keep_fraction"):
            bc.coarse_keep_fraction = 0.40
            applied.append("coarse_keep_fraction=0.40")
        if not hasattr(options, "sr2_branch_angular_diversity_sectors"):
            bc.angular_diversity_sectors = 6
            applied.append("angular_diversity_sectors=6")

        suffix = ", ".join(applied) if applied else "no override (user-explicit)"
        print(
            f"[smart_refine_2] adaptive branch_config: max walk depth "
            f"{max_depth} > threshold {threshold} -> {suffix}"
        )
        return bc


    def run(self):
        self.get_protein_complex()
        self.get_refine_ligands()
        self.get_output()
        self.fit_results = []
        self.debug_sdf_paths = []

        for ligand_index, ligand in enumerate(list(self.ligands), start=1):
            if self.pre_minimise_enabled:
                # Gate the pre-minimisation too: it sets the no-regression anchor,
                # so a polish that drags the pose off-density must not silently
                # become the baseline. Snapshot the docked-input pose/objective,
                # run the (re-scoring) polish, and revert if it made things worse.
                scorer = self._acceptance_scorer()
                pre_coords = np.asarray(
                    ligand._atom_positions, dtype=np.float64
                ).copy()
                pre_objective = self._pose_objective(ligand, scorer, pre_coords)
                ligand, _ = self._polish(
                    ligand, None, context_label="pre-search minimisation"
                )
                post_objective = self._pose_objective(
                    ligand, scorer, ligand._atom_positions
                )
                if (
                    pre_objective is not None
                    and post_objective is not None
                    and post_objective < pre_objective - _NO_REGRESSION_EPS
                ):
                    ligand._atom_positions = pre_coords
                    if getattr(ligand, "_protein_index", None) is not None:
                        ligand._init_local_protein(ligand._protein_index)
                    ligand.update_atom_qscores()
                    ligand.get_best_block_by_qscore()
                    print(
                        "[smart_refine_2] pre-search minimisation reverted: "
                        f"objective {post_objective:.5f} < input {pre_objective:.5f}"
                    )
            per_ligand_branch_config = self._adaptive_branch_config(ligand)
            ligand = refine_ligand(
                ligand,
                scorer=self.scorer,
                optimisation_scorer=self.optimisation_score,
                optimisation_weights=self.optimisation_weights,
                acceptance_scorer=self.acceptance_score,
                acceptance_weights=self.acceptance_weights,
                scorer_options=getattr(self.system, "options", None),
                fit_config=self.fit_config,
                branch_config=per_ligand_branch_config,
                kick_config=self.kick_config,
                patience=self.patience,
                root_tabu_size=self.root_tabu_size,
                clash_tradeoff_lambda=self.clash_tradeoff_lambda,
                polish_cb=self._build_polish_cb(),
                selection=self.selection,
                debug_dir=getattr(self.system, "output", None),
                ring_flex_enabled=self.ring_flex_enabled,
                ring_flex_confs=self.ring_flex_confs,
                ring_flex_rmsd=self.ring_flex_rmsd,
                drift_guard=self.drift_guard,
            )
            result = ligand._last_fit_in_map_result
            ligand, result = self._final_map_minimise_ligand(ligand, result)
            self.fit_results.append(result)
            self.ligands[ligand_index - 1] = ligand
            self.debug_sdf_paths.append(
                self.write_refined_ligand_sdf(ligand, result, ligand_index)
            )
            self._maybe_dump_parity_snapshot(ligand, result, ligand_index)

        return self.fit_results

    def _maybe_dump_parity_snapshot(self, refine_ligand, result, ligand_index):
        """If CHEMEM_SR2_PARITY_DIR is set, save a per-ligand snapshot for
        the parity-test scaffold (per_atom_qscores, final_coords, clash info)."""
        parity_dir = os.environ.get("CHEMEM_SR2_PARITY_DIR")
        if not parity_dir:
            return
        try:
            from ChemEM.tests.test_smart_refine_parity import save_baseline
            out_dir = os.path.join(parity_dir, f"ligand_{int(ligand_index):03d}")
            save_baseline(
                out_dir,
                per_atom_qscores=refine_ligand._per_atom_qscores,
                final_coords_A=refine_ligand._atom_positions,
                clash_penalty=float(getattr(result, "best_clash_penalty", 0.0)),
                clash_count=int(getattr(result, "best_clash_count", 0)),
                meta={
                    "ligand_index": int(ligand_index),
                    "qscore_candidate_dirs": int(
                        getattr(refine_ligand, "_qscore_candidate_dirs", 128)
                    ),
                    "fit_max_steps": int(self.fit_config.max_steps),
                    "fit_early_stop_tol": self.fit_config.early_stop_tol,
                    "branch_coarse_step_deg": float(self.branch_config.coarse_step_deg),
                    "branch_max_keep_per_step": int(self.branch_config.max_keep_per_step),
                },
            )
            print(f"[smart_refine_2] parity snapshot saved -> {out_dir}")
        except Exception as exc:
            print(f"[smart_refine_2] parity snapshot failed: {exc}")

    def write_refined_ligand_sdf(self, refine_ligand, result, ligand_index=1, filename=None):
        output_dir = getattr(self, "output", ".")
        if filename is None:
            filename = f"Ligand_{int(ligand_index):03d}.sdf"
        # Full-ligand SDF (leaving group retained) — same path for covalent and
        # non-covalent; refine_ligand._ligand.mol is already at the refined pose.
        sdf_path = write_refined_ligand_sdf(refine_ligand, result, output_dir, filename)
        # Covalent ligand: additionally write the covalently-bonded complex PDB (the
        # covalent bond is represented only there, not in the ligand SDF). The real
        # Ligand (with covalent_link) lives on _ligand_object — the refinement loop
        # replaces _ligand with a bare SimpleNamespace(mol=…) via _clone_refine_ligand.
        lig_obj = getattr(refine_ligand, "_ligand_object", None) or getattr(refine_ligand, "_ligand", None)
        if lig_obj is not None and getattr(lig_obj, "covalent_link", None) is not None:
            self._write_covalent_complex_pdb(refine_ligand, lig_obj, ligand_index, output_dir)
        return sdf_path

    def _write_covalent_complex_pdb(self, refine_ligand, lig, ligand_index, output_dir):
        """Write the covalently-bonded complex PDB (protein + F-removed bonded ligand)
        for a smart-refined covalent ligand. The refined coordinates live on the
        refinement's working mol (`refine_ligand._ligand.mol`), not necessarily on
        `lig.mol` (which may be the un-refined original)."""
        from ChemEM.parsers.covalent_output import (
            write_covalent_complex_pdb,
            build_covalent_complex_structure,
        )
        # Sync lig (the real Ligand) to the refined mol pose so the covalent merge uses
        # the refined coordinates. The working mol has the same atom order as lig.mol.
        try:
            src = getattr(getattr(refine_ligand, "_ligand", None), "mol", None) or lig.mol
            refined = np.asarray(src.GetConformer(0).GetPositions(), dtype=float)
            lig.set_positions(refined)
        except Exception as exc:
            print(f"[covalent] smart-refine: could not sync ligand positions: {exc}")

        # build_covalent_complex_structure copies the protein internally (parmed .copy),
        # so the live system.protein is not mutated.
        try:
            complex_struct = build_covalent_complex_structure(
                self.system.protein.complex_structure, lig
            )
            pdb_out = os.path.join(
                output_dir, f"covalent_complex_Ligand_{int(ligand_index):03d}.pdb"
            )
            write_covalent_complex_pdb(complex_struct, pdb_out)
            print(f"[covalent] smart-refine: wrote covalent complex PDB for "
                  f"ligand {ligand_index}")
        except Exception as exc:
            print(f"[covalent] smart-refine: covalent complex PDB failed: {exc}")


def _final_minimisation_dependencies():
    global ChemEMSimulationSetup
    global MinimiseInPlace
    global submap_from_structure
    global update_global_positions
    global update_ligand_positions
    global get_residue_positions
    global get_residue_subset_from_points

    if ChemEMSimulationSetup is None or MinimiseInPlace is None:
        from ChemEM.protocols.refine.pose_minimiser import (
            ChemEMSimulationSetup as _ChemEMSimulationSetup,
            MinimiseInPlace as _MinimiseInPlace,
        )

        ChemEMSimulationSetup = _ChemEMSimulationSetup
        MinimiseInPlace = _MinimiseInPlace
    if submap_from_structure is None:
        from ChemEM.protocols.core.density import (
            submap_from_structure as _submap_from_structure,
        )

        submap_from_structure = _submap_from_structure
    if update_global_positions is None or update_ligand_positions is None:
        from ChemEM.protocols.core.simulation import (
            update_global_positions as _update_global_positions,
            update_ligand_positions as _update_ligand_positions,
        )

        update_global_positions = _update_global_positions
        update_ligand_positions = _update_ligand_positions
    if get_residue_positions is None or get_residue_subset_from_points is None:
        from ChemEM.protocols.refine.refine_utils import (
            get_residue_positions as _get_residue_positions,
            get_residue_subset_from_points as _get_residue_subset_from_points,
        )

        get_residue_positions = _get_residue_positions
        get_residue_subset_from_points = _get_residue_subset_from_points

    return (
        ChemEMSimulationSetup,
        MinimiseInPlace,
        submap_from_structure,
        update_global_positions,
        update_ligand_positions,
        get_residue_positions,
        get_residue_subset_from_points,
    )


@contextmanager
def _suppress_polish_atom_warnings():
    """Filter ``Warning: Atom ('', '213', 'CX') in local structure not found
    in full structure.`` lines emitted by ``copy_global_positions`` during
    the polish step. The ligand atoms are absent from
    ``system.protein.complex_structure`` (protein only), so every ligand
    atom triggers the warning. The ligand is updated via
    ``copy_ligand_positions`` separately; these warnings carry no
    information and historically generated 80+ noise lines per polish.
    """
    import io
    import sys as _sys

    buf = io.StringIO()
    orig_stdout = _sys.stdout
    _sys.stdout = buf
    try:
        yield
    finally:
        _sys.stdout = orig_stdout
        for line in buf.getvalue().splitlines():
            if "in local structure not found in full structure" in line:
                continue
            if line.strip():
                print(line)


def _fragment_min_decision(
    block_before, block_after, overall_before, overall_after,
    block_tol=0.1, overall_tol=0.0,
):
    """Decide whether a fragment-pinned minimisation step is acceptable.

    ``block_before``/``block_after`` are per-block mean Q-scores aligned to the
    rotor-tree order (i.e. ``refine_ligand._block_qscores``). Returns
    ``("accept", [])`` when the overall objective held (improved or within
    ``overall_tol``) AND no block dropped by more than ``block_tol``; otherwise
    ``("pin", [block_row, ...])`` listing the rotor-tree rows to pin and re-run.
    If the overall regressed but every block is within ``block_tol`` (no single
    culprit), pin the single most-dropped block so the loop still makes progress.

    A non-finite objective on either side means the two poses are not comparable,
    so the objective is treated as NOT having held. Comparing sentinels directly
    made this a tautology (``-inf >= -inf`` is True), which accepted the very
    first minimisation unconditionally and meant no fragment was ever pinned.
    """
    n = min(len(block_before), len(block_after))
    deltas = [float(block_after[i]) - float(block_before[i]) for i in range(n)]
    comparable = np.isfinite(float(overall_before)) and np.isfinite(float(overall_after))
    overall_ok = comparable and (
        float(overall_after) >= float(overall_before) - float(overall_tol)
    )
    dropped = [i for i, d in enumerate(deltas) if d < -float(block_tol)]

    if overall_ok and not dropped:
        return "accept", []
    if dropped:
        return "pin", dropped
    # Overall regressed but no block beyond tol: pin the worst-dropping block.
    if deltas:
        return "pin", [int(min(range(n), key=lambda i: deltas[i]))]
    return "accept", []


def _build_local_refine_env(refine_ligand, *, system, options, context_label="polish"):
    """Build the OpenMM local-refine environment (ligand + nearby residues +
    density submap) used by the polish minimisers. Returns
    ``(env, ligand, protein_structure, deps)`` or ``None`` when no density map is
    available. Shared by the standard polish and the fragment-pinned minimiser so
    both use byte-identical settings (pin_k=5000, localise=False, and the map weight
    and solvent model resolved from --global-k / --implicit-solvent)."""
    density_map = getattr(system, "density_map", None)
    if bool(getattr(options, "no_map", False)) or density_map is None:
        print(f"[smart_refine_2] {context_label} skipped: no density map available")
        return None

    # Local imports: this module keeps the OpenMM stack out of import time (see
    # _final_minimisation_dependencies), and these two pull openmm.app in with them.
    from openmm import app
    from ChemEM.protocols.core.simulation import resolve_global_k, resolve_implicit_solvent

    deps = _final_minimisation_dependencies()
    (
        setup_cls,
        minimiser_cls,
        make_submap,
        copy_global_positions,
        copy_ligand_positions,
        residue_positions,
        residue_subset,
    ) = deps

    ligand = _sync_ligand_object_from_refine_ligand(refine_ligand)
    protein_structure = system.protein.complex_structure
    ligand_points = residue_positions(ligand.complex_structure.residues[0])
    local_structure = residue_subset(
        ligand_points,
        protein_structure,
        distance_cutoff=float(getattr(options, "local_radius", 12.0)),
    )
    local_map = make_submap(local_structure, density_map)

    env = setup_cls(
        protein_structure=local_structure,
        ligand_structure=[ligand],
        density_map=local_map,
        platform_name=getattr(system, "platform", "CPU"),
        protein_restraint="protein",
        pin_k=5000.0,
        localise=False,
        global_k=resolve_global_k(options, 150.0),
        solvent=resolve_implicit_solvent(options, app.GBn2),
        pin_specs=getattr(options, "pin_specs", []),
        distance_specs=getattr(options, "distance_specs", []),
        resource_owner=system,
    )
    return env, ligand, protein_structure, deps


def _apply_polished_env_to_ligand(
    refine_ligand, result, *, env, ligand, protein_structure, deps, final_energy,
    protein_index_refresh_cb, acceptance_scorer, eval_config, context_label,
):
    """Copy minimised positions back onto the RefineLigand, refresh its caches
    (per-atom + per-block Q-scores), and re-score ``result`` so the before/after
    acceptance gates compare REAL post-minimisation scores."""
    copy_global_positions = deps[3]
    copy_ligand_positions = deps[4]
    with _suppress_polish_atom_warnings():
        copy_global_positions(
            full_structure=protein_structure,
            local_structure=env.complex_structure,
        )
    copy_ligand_positions(
        local_structure=env.complex_structure,
        ligand_objects=[ligand],
    )
    if protein_index_refresh_cb is not None:
        protein_index_refresh_cb()
    protein_index = getattr(refine_ligand, "_protein_index", None)
    _refresh_refine_ligand_from_ligand_object(
        refine_ligand,
        protein_index=protein_index,
        result=result,
        final_energy=final_energy,
    )
    # _refresh_refine_ligand_from_ligand_object only updates best_coords_A; without
    # this the result's best_raw_score/best_objective stay at their pre-min values
    # and every before/after gate is a no-op.
    if result is not None and acceptance_scorer is not None and eval_config is not None:
        try:
            scorer = get_scorer(acceptance_scorer)
            score_result, clash = _full_ligand_evaluation(
                refine_ligand, scorer,
                np.asarray(refine_ligand._atom_positions, dtype=np.float64),
                eval_config,
            )
            result.best_raw_score = float(score_result.value)
            result.best_objective = _objective_from_eval(
                score_result.value, clash, eval_config,
                int(np.asarray(refine_ligand._atom_positions).shape[0]),
            )
            result.best_clash_penalty = float(clash.penalty)
            result.best_clash_count = int(clash.count)
        except Exception as exc:
            print(f"[smart_refine_2] {context_label} re-score skipped: {exc}")


def local_refine_polish_ligand(
    refine_ligand,
    result,
    *,
    system,
    protein_index_refresh_cb,
    options=None,
    context_label="polish",
    acceptance_scorer=None,
    eval_config=None,
):
    """Run an OpenMM local-refine polish on ``refine_ligand`` in place.

    Mirrors the body of ``SmartRefine2._final_map_minimise_ligand`` (which is
    itself the same code path as ``--refine --local-refine``): set up a
    ChemEMSimulationSetup environment around the ligand + nearby residues,
    minimise with the density bias, then copy positions back and refresh the
    RefineLigand's caches. Returns ``(refine_ligand, result)``; if the polish
    cannot run (no map, OpenMM unavailable, minimiser returned None) the
    ligand and result are returned untouched.

    Required arguments:
      - ``system`` — the SmartRefine2 system (needs ``.protein.complex_structure``
        and optionally ``.density_map``, ``.platform``).
      - ``protein_index_refresh_cb`` — callable invoked after the polish to
        rebuild the protein coordinate index and refresh any RefineLigand
        contexts (the SmartRefine2 caller passes
        ``lambda: (self.get_protein_complex(), self._refresh_refine_ligand_contexts())``).
      - ``options`` — typically the argparse Namespace on the system; read for
        ``no_map``, ``local_radius``, ``do_biased_md``, ``pin_specs``,
        ``distance_specs``. Falls back to defaults when None.
      - ``context_label`` — short string ("polish", "final polish",
        "per-iter polish") used only in the log line.
    """
    try:
        built = _build_local_refine_env(
            refine_ligand, system=system, options=options, context_label=context_label
        )
        if built is None:
            return refine_ligand, result
        env, ligand, protein_structure, deps = built
        minimiser_cls = deps[1]

        final_energy = minimiser_cls(env).run(
            do_biased_md=getattr(options, "do_biased_md", False),
            md_ps=5.0,
            max_iters=200,
        )
        if final_energy is None:
            print(
                f"[smart_refine_2] {context_label} skipped: minimiser returned no energy"
            )
            return refine_ligand, result

        _apply_polished_env_to_ligand(
            refine_ligand, result,
            env=env, ligand=ligand, protein_structure=protein_structure, deps=deps,
            final_energy=final_energy,
            protein_index_refresh_cb=protein_index_refresh_cb,
            acceptance_scorer=acceptance_scorer, eval_config=eval_config,
            context_label=context_label,
        )
        initial_raw = float(getattr(result, "initial_raw_score", 0.0)) if result is not None else 0.0
        best_raw = float(getattr(result, "best_raw_score", initial_raw)) if result is not None else initial_raw
        print(
            f"[smart_refine_2] {context_label} finished "
            f"E={float(final_energy):.2f} kcal/mol "
            f"raw {initial_raw:+.5f}->{best_raw:+.5f}"
        )
    except Exception as exc:
        print(
            f"[smart_refine_2] {context_label} failed; "
            f"keeping pre-polish pose: {exc}"
        )
    return refine_ligand, result


def fragment_pinned_polish_ligand(
    refine_ligand,
    result,
    *,
    system,
    protein_index_refresh_cb,
    options=None,
    context_label="fragment polish",
    acceptance_scorer=None,
    eval_config=None,
    pose_objective_fn=None,
    block_tol=0.1,
    max_iters=4,
):
    """Iterative fragment-pinned OpenMM minimiser (opt-in alternative to the
    standard polish).

    Minimise, score each semantic block's Q vs the INPUT pose, and accept when the
    overall objective held (improved or within tolerance) AND no block dropped by
    more than ``block_tol``. Otherwise pin the dropped blocks at their input
    positions, restart from the input pose, and re-minimise — the pinned set grows
    monotonically. After ``max_iters`` with no acceptance, revert to the input
    pose. Same ``(refine_ligand, result)`` contract as ``local_refine_polish_ligand``.
    """
    if pose_objective_fn is None:
        # Without an overall objective we can't gate per-fragment; fall back.
        return local_refine_polish_ligand(
            refine_ligand, result, system=system,
            protein_index_refresh_cb=protein_index_refresh_cb, options=options,
            context_label=context_label, acceptance_scorer=acceptance_scorer,
            eval_config=eval_config,
        )

    scorer = get_scorer(acceptance_scorer) if acceptance_scorer is not None else None
    input_coords = np.asarray(refine_ligand._atom_positions, dtype=np.float64).copy()
    overall_input = pose_objective_fn(refine_ligand, scorer, input_coords)
    block_input = list(getattr(refine_ligand, "_block_qscores", []) or [])

    accepted = False
    try:
        from openmm import unit
        from ChemEM.protocols.core.forces import ForceBuilder

        built = _build_local_refine_env(
            refine_ligand, system=system, options=options, context_label=context_label
        )
        if built is None:
            return refine_ligand, result
        env, ligand, protein_structure, deps = built
        minimiser_cls = deps[1]

        heavy = list(getattr(env, "ligand_heavy_indices", []) or [])
        n_heavy = int(refine_ligand._atom_positions.shape[0])
        if len(heavy) != n_heavy:
            print(
                f"[smart_refine_2] {context_label}: ligand heavy-atom count mismatch "
                f"(env {len(heavy)} vs refine {n_heavy}); using standard polish"
            )
            return local_refine_polish_ligand(
                refine_ligand, result, system=system,
                protein_index_refresh_cb=protein_index_refresh_cb, options=options,
                context_label=context_label, acceptance_scorer=acceptance_scorer,
                eval_config=eval_config,
            )

        ctx = env.simulation.context
        input_full_nm = np.asarray(
            ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64,
        )

        # One weighted pin force over all ligand heavy atoms, toggled per iteration.
        pin_force, slot_by_atom = ForceBuilder.create_weighted_positional_pin(
            heavy, input_full_nm, k_name="k_frag_pin"
        )
        env.complex_system.addForce(pin_force)
        ctx.reinitialize(preserveState=True)

        pinned_openmm = set()
        for it in range(int(max_iters)):
            ctx.setPositions(input_full_nm * unit.nanometer)  # restart from input
            for atom_idx, slot in slot_by_atom.items():
                w = 1.0 if atom_idx in pinned_openmm else 0.0
                x0, y0, z0 = (float(v) for v in input_full_nm[atom_idx])
                pin_force.setParticleParameters(slot, atom_idx, [x0, y0, z0, w])
            pin_force.updateParametersInContext(ctx)
            ctx.setParameter("k_frag_pin", 5000.0 if pinned_openmm else 0.0)

            final_energy = minimiser_cls(env).run(
                do_biased_md=getattr(options, "do_biased_md", False),
                md_ps=5.0, max_iters=200,
            )
            if final_energy is None:
                print(f"[smart_refine_2] {context_label} skipped: minimiser returned no energy")
                break

            _apply_polished_env_to_ligand(
                refine_ligand, result, env=env, ligand=ligand,
                protein_structure=protein_structure, deps=deps, final_energy=final_energy,
                protein_index_refresh_cb=protein_index_refresh_cb,
                acceptance_scorer=acceptance_scorer, eval_config=eval_config,
                context_label=context_label,
            )

            overall_after = pose_objective_fn(refine_ligand, scorer, refine_ligand._atom_positions)
            block_after = list(getattr(refine_ligand, "_block_qscores", []) or [])
            if overall_after is None or overall_input is None:
                break

            decision, dropped_rows = _fragment_min_decision(
                block_input, block_after, overall_input, overall_after, block_tol=block_tol
            )
            if decision == "accept":
                print(
                    f"[smart_refine_2] {context_label} accepted at iter {it + 1} "
                    f"(overall {overall_input:+.5f}->{overall_after:+.5f})"
                )
                accepted = True
                break

            newly = set()
            for block_row in dropped_rows:
                if block_row >= len(refine_ligand._rotor_tree):
                    continue
                for mol_idx in refine_ligand._rotor_tree[block_row].atom_indices:
                    row = refine_ligand._atom_row_by_mol_index.get(int(mol_idx))
                    if row is not None:
                        newly.add(int(heavy[int(row)]))
            if newly <= pinned_openmm:
                break  # no new atoms to pin -> no further progress
            pinned_openmm |= newly
            print(
                f"[smart_refine_2] {context_label} iter {it + 1}: pinning "
                f"{len(dropped_rows)} block(s) ({len(pinned_openmm)} atoms), re-running"
            )

        if not accepted:
            # Revert to the input pose (restore coords + caches + result).
            refine_ligand._atom_positions = input_coords
            if getattr(refine_ligand, "_protein_index", None) is not None:
                refine_ligand._init_local_protein(refine_ligand._protein_index)
            if hasattr(refine_ligand, "update_atom_qscores"):
                refine_ligand.update_atom_qscores()
            if hasattr(refine_ligand, "get_best_block_by_qscore"):
                refine_ligand.get_best_block_by_qscore()
            if result is not None and scorer is not None and eval_config is not None:
                try:
                    sr_res, clash = _full_ligand_evaluation(
                        refine_ligand, scorer, input_coords, eval_config
                    )
                    result.best_coords_A = input_coords.copy()
                    result.best_raw_score = float(sr_res.value)
                    result.best_objective = _objective_from_eval(
                        sr_res.value, clash, eval_config, int(input_coords.shape[0])
                    )
                    result.best_clash_penalty = float(clash.penalty)
                    result.best_clash_count = int(clash.count)
                except Exception:
                    pass
            print(
                f"[smart_refine_2] {context_label} reverted to input after "
                f"{int(max_iters)} iteration(s)"
            )
    except Exception as exc:
        print(
            f"[smart_refine_2] {context_label} failed; keeping pre-polish pose: {exc}"
        )
    return refine_ligand, result


def _sync_ligand_object_from_refine_ligand(refine_ligand):
    working_ligand = getattr(refine_ligand, "_ligand", None)
    ligand_object = getattr(refine_ligand, "_ligand_object", working_ligand)
    if (
        ligand_object is None
        or not hasattr(ligand_object, "set_positions")
        or working_ligand is None
        or not hasattr(working_ligand, "mol")
    ):
        raise RuntimeError("RefineLigand does not expose a syncable ligand object")

    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D
    except Exception as exc:
        raise RuntimeError("RDKit is required to sync ligand coordinates") from exc

    mol = Chem.Mol(working_ligand.mol)
    if mol.GetNumConformers() == 0:
        mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()), assignId=True)

    conf = mol.GetConformer(0)
    coords = np.asarray(conf.GetPositions(), dtype=np.float64)
    atom_indices = np.asarray(refine_ligand._atom_indices, dtype=int)
    heavy_coords = np.asarray(refine_ligand._atom_positions, dtype=np.float64)
    if heavy_coords.shape[0] != atom_indices.shape[0]:
        raise ValueError("RefineLigand heavy atom coordinates do not match atom indices")

    coords[atom_indices] = heavy_coords
    for atom_idx, xyz in enumerate(coords):
        conf.SetAtomPosition(
            int(atom_idx),
            Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])),
        )
    _refresh_hydrogen_positions_from_heavy_geometry(mol, Chem)

    ligand_object.mol = mol
    ligand_object.set_positions(
        np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)
    )
    return ligand_object


def _refresh_refine_ligand_from_ligand_object(
    refine_ligand,
    protein_index=None,
    result=None,
    final_energy=None,
):
    ligand_object = getattr(refine_ligand, "_ligand_object", None)
    working_ligand = getattr(refine_ligand, "_ligand", None)
    if ligand_object is not None and hasattr(ligand_object, "mol"):
        if working_ligand is None:
            refine_ligand._ligand = ligand_object
        elif hasattr(working_ligand, "mol"):
            working_ligand.mol = ligand_object.mol

    refine_ligand._init_atoms()
    if protein_index is not None:
        refine_ligand._protein_index = protein_index
        refine_ligand._init_local_protein(protein_index)
    if hasattr(refine_ligand, "update_atom_qscores"):
        refine_ligand.update_atom_qscores()
    if hasattr(refine_ligand, "get_best_block_by_qscore"):
        refine_ligand.get_best_block_by_qscore()

    if result is not None:
        result.best_coords_A = np.asarray(refine_ligand._atom_positions, dtype=np.float64).copy()
        result.best_rotation_matrix = None
        result.best_translation_A = None
        if isinstance(getattr(result, "score_terms", None), dict) and final_energy is not None:
            result.score_terms["final_minimise_energy_kcal"] = float(final_energy)
    return refine_ligand



def refine_ligand(
    refine_ligand,
    scorer="qscore",
    optimisation_scorer=None,
    optimisation_weights=None,
    acceptance_scorer=None,
    acceptance_weights=None,
    scorer_options=None,
    fit_config=None,
    branch_config=None,
    kick_config=None,
    min_score_improvement=0.0,
    max_clash_penalty_increase=1e-6,
    max_iters=25,
    patience=3,
    root_tabu_size=1,
    selection="branches",
    debug_dir=None,
    rng=None,
    clash_tradeoff_lambda=None,
    polish_cb=None,
    ring_flex_enabled=False,
    ring_flex_confs=8,
    ring_flex_rmsd=0.3,
    drift_guard=None,
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
        "greedy"   -- winner picked from {base, branches} with a raw-score
                      floor: only candidates whose post-fit raw is >= the
                      base's raw are eligible. The (clash_count, -raw) min
                      then picks the lowest-clash improver. Guarantees the
                      iteration cannot regress the raw score even when a
                      branch has fewer clashes than the base.
        "branches" -- winner picked from re-fit branches only (no floor);
                      can regress to a worse raw if every branch is worse
                      than the base.
    """
    if selection not in {"greedy", "branches"}:
        raise ValueError(f"selection must be 'greedy' or 'branches', got {selection!r}")
    optimisation_scorer, acceptance_scorer = _resolve_refine_scorers(
        scorer,
        optimisation_scorer=optimisation_scorer,
        optimisation_weights=optimisation_weights,
        acceptance_scorer=acceptance_scorer,
        acceptance_weights=acceptance_weights,
        scorer_options=scorer_options,
    )

    # Anchor the anti-drift guards to the STARTING pose (before any fit). The
    # snapshot lives on the refine_ligand object so it rides shallow-copy clones
    # made by _clone_refine_ligand, and is read at every accept/select call.
    _install_drift_anchor(refine_ligand, drift_guard)

    from collections import deque
    if rng is None:
        kick_seed = getattr(kick_config, "seed", None) if kick_config else None
        rng = np.random.default_rng(kick_seed) if kick_seed is not None else np.random.default_rng()
    tabu_size = max(0, int(root_tabu_size))
    recent_roots: deque[int] = deque(maxlen=tabu_size) if tabu_size > 0 else deque(maxlen=0)
    block_qscore_history: deque[list[float]] = deque(maxlen=2)

    refine_ligand = _fit_and_accept(
        refine_ligand,
        optimisation_scorer,
        acceptance_scorer,
        fit_config,
        min_score_improvement,
        max_clash_penalty_increase,
        clash_tradeoff_lambda=clash_tradeoff_lambda,
    )
    # No-regression anchor: snapshot the post-initial-fit pose (the fair
    # baseline — it already reflects any --sr2-pre-minimise). If the whole
    # search/polish run ends up worse than this by SR2's own acceptance
    # objective, the end-of-loop gate below reverts to it. This is what keeps
    # the aggressive torsion search from ever returning a pose worse than the
    # input, the catastrophic-regression failure mode on flexible ligands.
    anchor_result = _snapshot_fit_result(
        getattr(refine_ligand, "_last_fit_in_map_result", None)
    )
    anchor_objective = _result_best_objective(anchor_result)

    # Best-so-far pose. Without this the end-of-loop gate can only rewind to the
    # anchor, so a run that improved for ten iterations and then slipped on the
    # eleventh shipped the *input* pose — every intermediate gain discarded.
    # Seeded from the anchor so a run that never improves behaves exactly as before.
    best_snapshot = anchor_result
    best_objective_so_far = anchor_objective

    def _record_best(rl):
        """Snapshot ``rl``'s current result if it beats the best objective seen."""
        nonlocal best_snapshot, best_objective_so_far
        result = getattr(rl, "_last_fit_in_map_result", None)
        if result is None:
            return
        objective = _result_best_objective(result)
        if not np.isfinite(objective):
            return
        if not np.isfinite(best_objective_so_far) or objective > (
            best_objective_so_far + _NO_REGRESSION_EPS
        ):
            best_objective_so_far = objective
            best_snapshot = _snapshot_fit_result(result)

    # Snapshot the initial block Qscores so the very first iteration's kick
    # check has a valid "previous" reference.
    initial_block_qscores = getattr(refine_ligand, "_block_qscores", None)
    if initial_block_qscores is not None:
        block_qscore_history.append([float(q) for q in initial_block_qscores])
    refine_ligand._block_qscore_history = block_qscore_history
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

    # Round-11 polish-on-stall guard: tracks whether we've attempted a polish
    # since the last successful branch_walker iteration. Prevents back-to-back
    # polish attempts when the polish itself doesn't unstick the walker.
    polished_since_last_walker_success = False

    for iteration in range(max_iters):
        #refine_ligand = _fit_and_accept(refine_ligand, scorer, fit_config, min_score_improvement)

        rotor_tree = getattr(refine_ligand, "_rotor_tree", None)
        if not rotor_tree or not hasattr(refine_ligand, "get_best_block_by_qscore"):
            stop_reason = "no_rotor_tree"
            break
        root_block_id = int(
            rotor_tree[refine_ligand.get_best_block_by_qscore()].block_id
        )

        branch_results = branch_walker(
            refine_ligand,
            scorer=acceptance_scorer,
            config=branch_config,
            iteration=iteration,
            rng=rng,
        )
        refine_ligand._last_branch_walker_result = branch_results
        if ring_flex_enabled:
            # Aliphatic ring systems are rigid "core" blocks in the torsion
            # walker, so a wrong ring pucker can never be corrected by it.
            # Inject alternative ring puckers as extra candidates; they ride
            # the same refit/selection path as torsion candidates below.
            ring_results = ring_conformer_walker(
                refine_ligand,
                scorer=acceptance_scorer,
                config=branch_config,
                n_confs=ring_flex_confs,
                min_ring_rmsd=ring_flex_rmsd,
            )
            if ring_results:
                branch_results = list(branch_results) + list(ring_results)
        # Exocyclic ring-branch torsions are now seeded as the first step of the
        # relevant branch walk inside branch_walker (get_directional_torsion_walks),
        # so they ride the normal beam search above — no separate walker here.
        if not branch_results:
            # Round-12: walker exhausting its targets is a stall. Polish
            # always continues (no raw gate) because the walker may find
            # new walks after per-atom Qscores recompute, even when mean
            # raw is unchanged. The polished_since_last_walker_success
            # guard prevents back-to-back polishes -- one polish per stall
            # episode, then break if the next walker is still empty.
            if (
                polish_cb is not None
                and not polished_since_last_walker_success
            ):
                # Snapshot: the polish re-scores the result object IN PLACE, so
                # handing it the live one would rewrite the loop's recorded
                # scores even on a polish we then discard.
                last_result = _snapshot_fit_result(
                    getattr(refine_ligand, "_last_fit_in_map_result", None)
                )
                if last_result is not None:
                    polished_rl, polished_result = polish_cb(refine_ligand, last_result)
                    refine_ligand = polished_rl
                    if polished_result is not None:
                        refine_ligand._last_fit_in_map_result = polished_result
                    refine_ligand._block_qscore_history = block_qscore_history
                    polished_raw = (
                        _result_best_raw(polished_result)
                        if polished_result is not None else best_raw_score_so_far
                    )
                    if polished_raw > best_raw_score_so_far:
                        best_raw_score_so_far = polished_raw
                    _record_best(refine_ligand)
                    polished_since_last_walker_success = True
                    current_block_qscores = getattr(refine_ligand, "_block_qscores", None)
                    if current_block_qscores is not None:
                        block_qscore_history.append(
                            [float(q) for q in current_block_qscores]
                        )
                    _set_refine_loop_diagnostics(
                        refine_ligand,
                        iterations_completed,
                        no_improve_iters,
                        "max_iters",
                    )
                    continue
            stop_reason = "no_branch_results"
            break

        # Walker succeeded -- reset the polish-on-stall guard so a future
        # walker-empty exit can attempt a polish again.
        polished_since_last_walker_success = False

        refine_ligand = _refit_branch_candidates(
            refine_ligand,
            branch_results,
            optimisation_scorer,
            acceptance_scorer,
            fit_config,
            min_score_improvement,
            selection,
            max_clash_penalty_increase,
            clash_tradeoff_lambda=clash_tradeoff_lambda,
        )
        iterations_completed = iteration + 1
        _record_best(refine_ligand)

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

        # Rolling root tabu: keep the last N roots in a deque so the loop has
        # to visit other blocks as root before reusing one. tabu_size == 1
        # reproduces the previous size-1 exclusion.
        if tabu_size > 0:
            recent_roots.append(int(root_block_id))
            refine_ligand._excluded_root_blocks = set(recent_roots)
        else:
            refine_ligand._excluded_root_blocks = set()

        # Snapshot per-block Qscores so the kick eligibility check at the
        # next stall has up-to-date "current" and "previous" values.
        current_block_qscores = getattr(refine_ligand, "_block_qscores", None)
        if current_block_qscores is not None:
            block_qscore_history.append([float(q) for q in current_block_qscores])
        refine_ligand._block_qscore_history = block_qscore_history

        _set_refine_loop_diagnostics(
            refine_ligand,
            iterations_completed,
            no_improve_iters,
            stop_reason,
        )

        #debug writer here!!!
        #if branch_dir:
        #    iter_branch_dir = os.path.join(branch_dir, f"iter_{iteration}")
        #    os.makedirs(iter_branch_dir, exist_ok=True)
        #    _debug_write_branch_results(
        #        refine_ligand,
        #        branch_results,
        #        iter_branch_dir,
        #        iteration=iteration,
        #    )
        #add per iteration debugger here
        #_debug_write_iteration_ligand(refine_ligand, iter_dir, iteration=iteration)
        if stop_reason == "patience":
            # First recovery move: OpenMM local-refine polish (deterministic,
            # cheap). Per the round-4 redesign, polish only runs here — NOT
            # after every iteration — to avoid the polish vs. Qscore tug-of-war
            # observed in round 3. If polish unblocks the search the loop
            # continues; otherwise we fall through to the stochastic kick.
            if polish_cb is not None:
                # Snapshot — see the note on the walker-empty polish above; this
                # branch discards the polish outright unless it improves, so it
                # must not let the polish rewrite the live result's scores.
                last_result = _snapshot_fit_result(
                    getattr(refine_ligand, "_last_fit_in_map_result", None)
                )
                if last_result is not None:
                    polished_rl, polished_result = polish_cb(refine_ligand, last_result)
                    polished_raw = _result_best_raw(polished_result) if polished_result is not None else float("-inf")
                    if polished_raw > best_raw_score_so_far + float(min_score_improvement):
                        refine_ligand = polished_rl
                        if polished_result is not None:
                            refine_ligand._last_fit_in_map_result = polished_result
                        refine_ligand._block_qscore_history = block_qscore_history
                        best_raw_score_so_far = polished_raw
                        _record_best(refine_ligand)
                        no_improve_iters = 0
                        stop_reason = "max_iters"
                        # Round-11: mark polish-attempted so the next walker
                        # iteration, if it returns empty, won't fire another
                        # polish immediately. Resets when the walker succeeds.
                        polished_since_last_walker_success = True
                        current_block_qscores = getattr(refine_ligand, "_block_qscores", None)
                        if current_block_qscores is not None:
                            block_qscore_history.append(
                                [float(q) for q in current_block_qscores]
                            )
                        _set_refine_loop_diagnostics(
                            refine_ligand,
                            iterations_completed,
                            no_improve_iters,
                            stop_reason,
                        )
                        continue

            kicked_rl = refine_ligand
            kick_improved = False
            if kick_config is not None and int(getattr(kick_config, "tries", 0)) > 0:
                kicked_rl, kick_improved = _score_driven_kick_round(
                    refine_ligand,
                    optimisation_scorer,
                    acceptance_scorer,
                    fit_config,
                    branch_config,
                    kick_config,
                    min_score_improvement,
                    max_clash_penalty_increase,
                    selection,
                    rng,
                    clash_tradeoff_lambda=clash_tradeoff_lambda,
                )
            if kick_improved:
                refine_ligand = kicked_rl
                refine_ligand._block_qscore_history = block_qscore_history
                latest_raw_score = _result_best_raw(
                    getattr(refine_ligand, "_last_fit_in_map_result", None)
                )
                best_raw_score_so_far = max(best_raw_score_so_far, latest_raw_score)
                _record_best(refine_ligand)
                no_improve_iters = 0
                stop_reason = "max_iters"
                current_block_qscores = getattr(refine_ligand, "_block_qscores", None)
                if current_block_qscores is not None:
                    block_qscore_history.append(
                        [float(q) for q in current_block_qscores]
                    )
                _set_refine_loop_diagnostics(
                    refine_ligand,
                    iterations_completed,
                    no_improve_iters,
                    stop_reason,
                )
                continue
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

    # No-regression gate: if the final pose is worse than the best pose the run
    # actually produced, revert to that. Falling back to the anchor only when the
    # search never beat it means SR2 returns its best work, not merely something
    # no worse than its input.
    revert_target = best_snapshot if best_snapshot is not None else anchor_result
    revert_objective = max(best_objective_so_far, anchor_objective)
    refine_ligand = _apply_no_regression_gate(
        refine_ligand, revert_target, revert_objective
    )

    return refine_ligand


@dataclass
class KickConfig:
    """Opt-in score-driven basin-hopping after the patience break.

    The kick perturbs torsions on the walks from the current root to each
    block that is *both* poorly-fit (Qscore below ``qscore_threshold``) *and*
    stagnating (Qscore improvement since last iteration <= ``stagnation_tol``).
    All other torsions are left alone — well-fit regions are not disturbed.
    With tries == 0 the kick is a no-op and the loop breaks on patience as
    before."""

    tries: int = 0
    qscore_threshold: float = 0.5
    stagnation_tol: float = 1e-3
    jitter_deg: float = 30.0
    seed: int | None = None


def _kick_eligible_block_ids(refine_ligand, kick_config):
    """Return the list of block IDs that pass both the poor-fit and the
    stagnation criteria. Empty list => no kick this round."""
    block_qscores = getattr(refine_ligand, "_block_qscores", None)
    if block_qscores is None or not len(block_qscores):
        return []
    history = list(getattr(refine_ligand, "_block_qscore_history", []) or [])
    prev = history[-1] if history else None
    rotor_tree = getattr(refine_ligand, "_rotor_tree", None) or ()
    qscore_threshold = float(kick_config.qscore_threshold)
    stagnation_tol = float(kick_config.stagnation_tol)
    eligible = []
    for idx, q_now in enumerate(block_qscores):
        if idx >= len(rotor_tree):
            continue
        q_now = float(q_now)
        if q_now >= qscore_threshold:
            continue
        if prev is not None and idx < len(prev):
            q_prev = float(prev[idx])
            delta = q_now - q_prev
            if delta > stagnation_tol:
                # The block is poor but still improving — leave it to the
                # normal walker.
                continue
        eligible.append(int(rotor_tree[idx].block_id))
    return eligible


def _score_driven_kick_round(
    refine_ligand,
    optimisation_scorer,
    acceptance_scorer,
    fit_config,
    branch_config,
    kick_config,
    min_score_improvement,
    max_clash_penalty_increase,
    selection,
    rng,
    clash_tradeoff_lambda=None,
):
    """Score-driven basin-hopping kick. Returns (refine_ligand, improved_bool).

    Jitter is applied only to torsions on walks from the current root to
    each kick-eligible block. Well-fit regions are not touched. We then
    re-run ``fit_in_map`` to relax the kicked geometry and keep the best of
    {pre-kick, all kick attempts}.
    """
    from rdkit.Chem import rdMolTransforms

    tries = int(kick_config.tries)
    if tries <= 0:
        return refine_ligand, False
    rotor_tree = getattr(refine_ligand, "_rotor_tree", None)
    if not rotor_tree:
        return refine_ligand, False
    eligible_block_ids = _kick_eligible_block_ids(refine_ligand, kick_config)
    if not eligible_block_ids:
        return refine_ligand, False

    current_root_idx = refine_ligand.get_best_block_by_qscore()
    current_root_block_id = int(rotor_tree[current_root_idx].block_id)

    walks = build_directional_torsion_walks(
        refine_ligand._ligand.mol,
        rotor_tree,
        root_block_id=current_root_block_id,
        target_block_ids=eligible_block_ids,
    )
    if not walks:
        return refine_ligand, False

    # Collect the unique set of (i, j, k, l) dihedrals to jitter — across all
    # eligible walks. A single dihedral may appear in multiple walks; we only
    # jitter it once per attempt.
    jitter_dihedrals = []
    seen_axes = set()
    for walk in walks:
        for step in walk.steps:
            axis = tuple(int(x) for x in step.torsion.axis)
            if axis in seen_axes:
                continue
            seen_axes.add(axis)
            jitter_dihedrals.append(tuple(int(x) for x in step.dihedral))

    if not jitter_dihedrals:
        return refine_ligand, False

    baseline_score = _result_best_raw(
        getattr(refine_ligand, "_last_fit_in_map_result", None)
    )
    print(
        f"[smart_refine_2] score-driven kick: tries={tries} "
        f"eligible_blocks={eligible_block_ids} jitter={kick_config.jitter_deg:.1f}deg "
        f"jitter_dihedrals={len(jitter_dihedrals)} baseline_raw={baseline_score:+.5f}"
    )

    best_rl = refine_ligand
    best_score = baseline_score
    improved = False

    for attempt in range(tries):
        clone = _clone_refine_ligand(refine_ligand)
        mol = clone._ligand.mol
        conf = mol.GetConformer(0)
        for dihedral in jitter_dihedrals:
            try:
                current_angle = float(rdMolTransforms.GetDihedralDeg(conf, *dihedral))
            except Exception:
                continue
            delta = float(rng.uniform(-kick_config.jitter_deg, kick_config.jitter_deg))
            rdMolTransforms.SetDihedralDeg(conf, *dihedral, current_angle + delta)
        # Write the perturbed coords back to the heavy-atom array.
        for row, mol_idx in enumerate(clone._atom_indices.tolist()):
            p = conf.GetAtomPosition(int(mol_idx))
            clone._atom_positions[row, 0] = float(p.x)
            clone._atom_positions[row, 1] = float(p.y)
            clone._atom_positions[row, 2] = float(p.z)
        if hasattr(clone, "update_atom_qscores"):
            clone.update_atom_qscores()

        # Relax with fit_in_map; the existing _fit_and_accept handles the
        # rescore-for-acceptance + accepter logic.
        clone = _fit_and_accept(
            clone,
            optimisation_scorer,
            acceptance_scorer,
            fit_config,
            min_score_improvement,
            max_clash_penalty_increase,
            clash_tradeoff_lambda=clash_tradeoff_lambda,
        )
        attempt_score = _result_best_raw(
            getattr(clone, "_last_fit_in_map_result", None)
        )
        print(
            f"[smart_refine_2] kick attempt {attempt + 1}/{tries} raw={attempt_score:+.5f}"
        )
        if attempt_score > best_score + float(min_score_improvement):
            best_score = attempt_score
            best_rl = clone
            improved = True

    if improved:
        print(
            f"[smart_refine_2] kick improved raw {baseline_score:+.5f} -> {best_score:+.5f}"
        )
    else:
        print("[smart_refine_2] kick did not improve; retaining pre-kick ligand")
    return best_rl, improved


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


# Tolerance below which a pose is treated as "not worse" by the no-regression
# gate. Pure numerical jitter (re-scoring the same coords) must not trip it.
_NO_REGRESSION_EPS = 1e-9


def _result_best_objective(result, default=float("-inf")):
    """Acceptance objective of a fit result (raw score minus clash penalty).

    This is the quantity SR2 actually optimises (see
    ``_objective_from_raw_and_clash`` and ``_rescore_fit_result_for_acceptance``),
    so the no-regression gate compares poses on it. Falls back to the raw score
    when ``best_objective`` is absent/non-finite (e.g. a hard-clash ``-inf``)."""
    if result is None:
        return float(default)
    try:
        obj = float(result.best_objective)
    except Exception:
        return _result_best_raw(result, default)
    if np.isfinite(obj):
        return obj
    return float(obj)


def _regression_detected(final_result, final_objective, target_result, target_objective):
    """True when ``final_result`` is worse than the revert target.

    Compares on the acceptance objective, but falls back to the raw score when
    either side is non-finite. Without that fallback a single ``-inf`` leaking in
    from anywhere silently rewinds the whole run: a ``-inf`` always loses to a
    finite target no matter how good the pose actually is. All objectives now come
    from ``_objective_from_eval`` so this should not trigger, but the gate is the
    last thing standing between the search and the output — it must not be the
    place where a sentinel value costs the user their result.
    """
    if np.isfinite(final_objective) and np.isfinite(target_objective):
        return final_objective < target_objective - _NO_REGRESSION_EPS
    final_raw = _result_best_raw(final_result)
    target_raw = _result_best_raw(target_result)
    if not np.isfinite(final_raw) or not np.isfinite(target_raw):
        # Nothing is comparable — keep what the search produced.
        return False
    return final_raw < target_raw - _NO_REGRESSION_EPS


def _apply_no_regression_gate(refine_ligand, anchor_result, anchor_objective):
    """Revert ``refine_ligand`` to ``anchor_result`` when its final acceptance
    objective has regressed below ``anchor_objective``, and record the decision on
    ``._sr2_reverted_to_anchor``. This is the global guarantee that SR2 never
    returns a pose worse (by its own objective) than the best it has seen —
    the catastrophic-regression failure mode.

    ``anchor_result`` is the best-so-far snapshot when the loop has one, else the
    post-initial-fit pose; the caller decides."""
    final_result = getattr(refine_ligand, "_last_fit_in_map_result", None)
    final_objective = _result_best_objective(final_result)
    if anchor_result is not None and _regression_detected(
        final_result, final_objective, anchor_result, anchor_objective
    ):
        apply_refinement(refine_ligand, anchor_result)
        refine_ligand._last_fit_in_map_result = anchor_result
        refine_ligand._sr2_reverted_to_anchor = True
        print(
            "[smart_refine_2] no-regression gate: reverted to best-so-far "
            f"(final_objective={final_objective:+.5f} < "
            f"target_objective={anchor_objective:+.5f}, "
            f"raw {_result_best_raw(final_result):+.5f}->"
            f"{_result_best_raw(anchor_result):+.5f})"
        )
    else:
        refine_ligand._sr2_reverted_to_anchor = False
    return refine_ligand


def _snapshot_fit_result(result):
    """Return an aliasing-free shallow copy of a fit result with its coordinate
    array copied, safe to stash as a revert anchor while the loop keeps
    mutating the live result objects."""
    if result is None:
        return None
    import copy
    snapshot = copy.copy(result)
    try:
        snapshot.best_coords_A = np.asarray(result.best_coords_A, dtype=np.float64).copy()
    except Exception:
        pass
    return snapshot


def _set_refine_loop_diagnostics(refine_ligand, iterations_completed, no_improve_iters, stop_reason):
    refine_ligand._sr2_iterations_completed = int(iterations_completed)
    refine_ligand._sr2_no_improve_iters = int(no_improve_iters)
    refine_ligand._sr2_stop_reason = str(stop_reason)


def _resolve_refine_scorers(
    scorer,
    *,
    optimisation_scorer=None,
    optimisation_weights=None,
    acceptance_scorer=None,
    acceptance_weights=None,
    scorer_options=None,
):
    role_specific = any(
        value is not None
        for value in (
            optimisation_scorer,
            optimisation_weights,
            acceptance_scorer,
            acceptance_weights,
        )
    )
    if not role_specific:
        resolved = _get_scorer_for_role(scorer, options=scorer_options)
        return resolved, resolved

    fit_source = optimisation_scorer if optimisation_scorer is not None else scorer
    accept_source = acceptance_scorer if acceptance_scorer is not None else "qscore"
    fit_scorer = _get_scorer_for_role(
        fit_source,
        weights=optimisation_weights,
        options=scorer_options,
    )
    accept_scorer = _get_scorer_for_role(
        accept_source,
        weights=acceptance_weights,
        options=scorer_options,
    )
    if getattr(fit_scorer, "spec_key", None) == getattr(accept_scorer, "spec_key", None):
        accept_scorer = fit_scorer
    return fit_scorer, accept_scorer


def _get_scorer_for_role(scorer, weights=None, options=None):
    if weights is None and options is None:
        return get_scorer(scorer)
    return get_scorer(scorer, weights=weights, options=options)


def _fit_and_accept(
    rl,
    optimisation_scorer,
    acceptance_scorer,
    config,
    min_score_improvement,
    max_clash_penalty_increase=1e-6,
    clash_tradeoff_lambda=None,
):
    result = fit_in_map(rl, scorer=optimisation_scorer, config=config)
    _rescore_fit_result_for_acceptance(
        rl,
        result,
        acceptance_scorer=acceptance_scorer,
        optimisation_scorer=optimisation_scorer,
        config=config,
    )
    rl._last_fit_in_map_result = result
    _log_fit_in_map(result)
    return accepter(
        rl,
        result,
        min_score_improvement=min_score_improvement,
        max_clash_penalty_increase=max_clash_penalty_increase,
        clash_tradeoff_lambda=clash_tradeoff_lambda,
    )


def _refit_branch_candidates(
    base_rl,
    branch_results,
    optimisation_scorer,
    acceptance_scorer,
    config,
    min_score_improvement,
    selection,
    max_clash_penalty_increase=1e-6,
    clash_tradeoff_lambda=None,
):
    """Run fit_in_map on a clone of base_rl for each branch_walker pose; pick the winner."""
    if not branch_results:
        return base_rl

    candidates = []
    for branch_result in branch_results:
        clone = _clone_refine_ligand(base_rl)
        clone = apply_refinement(clone, branch_result)
        fit_result = fit_in_map(clone, scorer=optimisation_scorer, config=config)
        _rescore_fit_result_for_acceptance(
            clone,
            fit_result,
            acceptance_scorer=acceptance_scorer,
            optimisation_scorer=optimisation_scorer,
            config=config,
        )
        clone._last_fit_in_map_result = fit_result
        _log_fit_in_map(fit_result)
        clone = accepter(
            clone,
            fit_result,
            min_score_improvement=min_score_improvement,
            max_clash_penalty_increase=max_clash_penalty_increase,
            clash_tradeoff_lambda=clash_tradeoff_lambda,
        )
        candidates.append((clone, clone._last_fit_in_map_result))

    raw_floor = None
    if selection == "greedy":
        candidates.append((base_rl, base_rl._last_fit_in_map_result))
        base_result = base_rl._last_fit_in_map_result
        if base_result is not None:
            try:
                raw_floor = float(base_result.best_raw_score)
                if not np.isfinite(raw_floor):
                    raw_floor = None
            except Exception:
                raw_floor = None

    return _pick_best_ligand(
        candidates,
        clash_tradeoff_lambda=clash_tradeoff_lambda,
        raw_floor=raw_floor,
    )


def _score_as_result(scorer, refine_ligand, coords_A):
    score_out = scorer.score(refine_ligand, coords_A)
    score_result = (
        score_out
        if isinstance(score_out, ScoreResult)
        else ScoreResult(value=float(score_out), terms={})
    )
    raw = float(score_result.value)
    if not np.isfinite(raw):
        raw = float("-inf")
    return ScoreResult(value=raw, terms=dict(score_result.terms))


def _objective_from_raw_and_clash(raw, clash_count, clash_penalty, mode, weight, n_atoms):
    raw = float(raw)
    if not np.isfinite(raw):
        return float("-inf")
    mode = str(mode).lower().strip()
    if mode == "off":
        return raw
    if mode == "hard":
        return raw if int(clash_count) == 0 else float("-inf")
    normalized_penalty = float(clash_penalty) / max(1, int(n_atoms))
    return float(raw) - float(weight) * normalized_penalty


def _objective_from_eval(raw, clash, config, n_atoms):
    """Single source of truth for "the acceptance objective" of a scored pose.

    Every gate in SR2 must derive the objective through here so they all speak the
    same units. Previously the polish and branch-walker paths hard-coded the *hard*
    convention (``-inf`` on a single clash) while fit_in_map used the configured
    ``clash_mode`` (soft by default). Any ligand that cannot reach zero clashes --
    i.e. any large flexible ligand -- then scored ``-inf`` on one side of the
    end-of-loop no-regression comparison and a finite soft objective on the other,
    so the gate reverted to the anchor and threw the whole search away.
    """
    return _objective_from_raw_and_clash(
        raw,
        clash.count,
        clash.penalty,
        getattr(config, "clash_mode", "soft"),
        getattr(config, "clash_weight", 1.0),
        n_atoms,
    )


def _rescore_fit_result_for_acceptance(
    refine_ligand,
    result,
    *,
    acceptance_scorer,
    optimisation_scorer,
    config,
):
    if acceptance_scorer is optimisation_scorer:
        return result

    initial_coords = np.asarray(refine_ligand._atom_positions, dtype=np.float64)
    best_coords = np.asarray(result.best_coords_A, dtype=np.float64)
    initial_score = _score_as_result(acceptance_scorer, refine_ligand, initial_coords)
    best_score = _score_as_result(acceptance_scorer, refine_ligand, best_coords)

    score_terms = dict(getattr(result, "score_terms", {}) or {})
    clash_mode = score_terms.get(
        "fit_clash_mode",
        getattr(config, "clash_mode", "off") if config is not None else "off",
    )
    clash_weight = float(
        score_terms.get(
            "fit_clash_weight",
            getattr(config, "clash_weight", 1.0) if config is not None else 1.0,
        )
    )
    n_atoms = initial_coords.shape[0]

    optimisation_terms = dict(score_terms)
    result.score_terms = score_terms
    result.score_terms.update(
        {
            "optimisation_initial_raw_score": float(result.initial_raw_score),
            "optimisation_best_raw_score": float(result.best_raw_score),
            "optimisation_initial_objective": float(result.initial_objective),
            "optimisation_best_objective": float(result.best_objective),
            "optimisation_score_terms": optimisation_terms,
            "acceptance_initial_raw_score": float(initial_score.value),
            "acceptance_best_raw_score": float(best_score.value),
            "acceptance_initial_score_terms": dict(initial_score.terms),
            "acceptance_best_score_terms": dict(best_score.terms),
        }
    )
    result.initial_raw_score = float(initial_score.value)
    result.best_raw_score = float(best_score.value)
    result.initial_objective = _objective_from_raw_and_clash(
        result.initial_raw_score,
        result.initial_clash_count,
        result.initial_clash_penalty,
        clash_mode,
        clash_weight,
        n_atoms,
    )
    result.best_objective = _objective_from_raw_and_clash(
        result.best_raw_score,
        result.best_clash_count,
        result.best_clash_penalty,
        clash_mode,
        clash_weight,
        n_atoms,
    )
    return result


def _pick_best_ligand(candidates, clash_tradeoff_lambda=None, raw_floor=None):
    # Anti-drift guard, read off each candidate clone (no-op -> (0.0, True), so
    # the ranking below is bit-for-bit unchanged when no guard is active). The
    # envelope gate drops candidates that leave the starting blob; the centroid
    # penalty down-ranks far-moving candidates within the surviving pool.
    penalty_by_id = {}
    passing = []
    for pair in candidates:
        pen, ok = _drift_assess(pair[0], pair[1].best_coords_A)
        penalty_by_id[id(pair)] = pen
        if ok:
            passing.append(pair)
    gated = passing if passing else candidates

    def _pen(pair):
        return penalty_by_id.get(id(pair), 0.0)

    if clash_tradeoff_lambda is not None:
        lam = float(clash_tradeoff_lambda)
        winner, _ = max(
            gated,
            key=lambda pair: (
                float(pair[1].best_raw_score)
                - lam * float(pair[1].best_clash_penalty)
                - _pen(pair)
            ),
        )
        return winner

    pool = gated
    if raw_floor is not None:
        # Under selection="greedy" we promised "never regresses". Filter out
        # candidates whose post-fit raw is below the base's raw so the
        # lexicographic (clash_count, -raw_score) picker can't sacrifice raw
        # to save a clash. The base itself trivially satisfies raw >= raw_floor,
        # so the pool is always non-empty under greedy.
        improvers = [
            c for c in gated
            if float(c[1].best_raw_score) >= float(raw_floor)
        ]
        if improvers:
            pool = improvers
        print(
            "[smart_refine_2] picker(greedy) "
            f"raw_floor={float(raw_floor):+.5f} "
            f"N_total={len(candidates)} N_improvers={len(pool) if improvers else 0}"
        )

    winner, _ = min(
        pool,
        key=lambda pair: (
            int(pair[1].best_clash_count),
            -(float(pair[1].best_raw_score) - _pen(pair)),
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
    clone._ligand_object = getattr(rl, "_ligand_object", rl._ligand)
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
    ligand_clash_topological_exclusion_bonds: int = 3
    similar_score_tol: float = 1e-3
    sigma_ref: float = 0.6
    # Clash convention for the objective attached to walker candidates. Mirrors
    # FitInMapConfig's defaults so _objective_from_eval yields the same units on
    # both paths -- see the note there.
    clash_mode: str = "soft"
    clash_weight: float = 1.0
    # opt-in tuning: 0.0 = current behaviour (frontier-only scoring).
    # Positive values add a per-candidate term equal to alpha * mean(Qscore of
    # moved-but-not-frontier atoms) so that early-walk decisions are biased
    # toward dihedrals that swing the downstream chain toward density.
    downstream_lookahead_weight: float = 0.0
    # opt-in beam-diversity: when > 0, _select_beam buckets candidates into
    # N angular sectors (by the *last* dihedral) and keeps at least one
    # survivor per non-empty sector before filling remaining slots by score.
    angular_diversity_sectors: int = 0
    # opt-in Metropolis acceptance during beam selection. None = strict greedy
    # (current behaviour). A positive temperature lets _select_beam swap in
    # worse candidates with prob exp(-(best - cand) / T_iter), where
    # T_iter = metropolis_temp * metropolis_decay ** outer_iteration.
    metropolis_temp: float | None = None
    metropolis_decay: float = 0.7
    # opt-in composite ranking inside the branch walker. None = current
    # behaviour (rank by frontier_score, tie-break by clash_count then
    # penalty). A positive value uses (frontier_score - lambda * clash_penalty)
    # as the ranking and threshold key so candidates with slightly lower Qscore
    # but much lower clash penalty are kept in the beam.
    clash_tradeoff_lambda: float | None = None


@dataclass
class _BranchCandidate:
    coords_A: np.ndarray
    dihedrals_deg: tuple
    frontier_score: float
    clash_count: int
    clash_penalty: float


def branch_walker(refine_ligand, scorer=None, config=None, iteration=0, rng=None):
    """Walk torsions outward from the best block; return one FitInMapResult per surviving beam candidate.

    The walker keeps up to ``config.max_keep_per_step`` candidates per torsion step,
    scoring frontier atoms via Qscore against the local protein context with a
    relaxed VDW allowance. Returned results follow the FitInMapResult contract so
    the existing accepter/apply_refinement helpers work unchanged.

    ``iteration`` is the outer-loop iteration index (used only for the optional
    Metropolis temperature schedule). ``rng`` lets callers pass a seeded
    generator so benchmarks reproduce; if None a fresh one is created.
    """
    config = config or BranchWalkConfig()
    scorer = get_scorer(scorer)
    if rng is None:
        rng = np.random.default_rng()

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
    initial_objective = _objective_from_eval(
        initial_raw, base_clash, config, int(base_coords.shape[0])
    )

    total_steps = sum(len(walk.steps) for walk in walks)
    progress = _branch_progress(total_steps)

    results = []
    total_evals = 0
    try:
        for walk in walks:
            survivors, n_evals = _walk_beam_search(
                refine_ligand, walk, base_coords, config,
                progress=progress, iteration=iteration, rng=rng,
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


def _walk_beam_search(refine_ligand, walk, base_coords, config, progress=None, iteration=0, rng=None):
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
        frontier_rows, frontier_mol_ids = _heavy_rows_and_mol_ids(
            refine_ligand,
            step.frontier_atom_indices,
        )
        moved_rows, moved_mol_ids = _moved_heavy_rows(refine_ligand, step.moved_atom_indices)
        fixed_mol_ids = _fixed_ligand_atom_indices(
            refine_ligand,
            step.moved_atom_indices,
        )
        if frontier_rows.size == 0 or moved_rows.size == 0:
            if progress is not None:
                progress.update(1)
            continue
        frontier_elements = np.asarray(refine_ligand._atom_elements, dtype=object)[frontier_rows]
        # Atoms moved by this dihedral but not locked in here — they will be
        # repositioned by subsequent dihedrals further out on the walk. Their
        # current Qscore is used as a soft downstream look-ahead when
        # config.downstream_lookahead_weight > 0.
        frontier_row_set = set(int(r) for r in frontier_rows.tolist())
        downstream_rows = np.asarray(
            [int(r) for r in moved_rows.tolist() if int(r) not in frontier_row_set],
            dtype=int,
        )

        next_cands = []
        for parent in beam:
            kept, n = _evaluate_step_for_parent(
                refine_ligand, step, parent,
                moved_rows, moved_mol_ids,
                frontier_rows, frontier_mol_ids, frontier_elements,
                downstream_rows,
                fixed_mol_ids,
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
        beam = _select_beam(next_cands, config, iteration=iteration, rng=rng)

    if beam and beam[0].dihedrals_deg == ():
        return [], total_evals
    return beam, total_evals


def _evaluate_exo_step_for_parent(
    refine_ligand, step, parent,
    moved_rows, moved_mol_ids,
    frontier_rows, frontier_mol_ids, frontier_elements,
    downstream_rows, fixed_mol_ids, config,
):
    """Apply an exocyclic ring-branch swing: rotate the moved atoms about the
    ring-bond axis (j,k) over a coarse+local angle sweep, scoring each pose like
    a normal walk step. Mirrors _evaluate_step_for_parent's structure but uses an
    explicit rotation (the math from the standalone exo walker) instead of
    rdMolTransforms.SetDihedralDeg, which cannot rotate a ring bond."""
    from rdkit.Geometry import Point3D
    from .optimisers import _rotation_matrix

    row_by_mol = refine_ligand._atom_row_by_mol_index
    j, k = int(step.axis[0]), int(step.axis[1])
    if j not in row_by_mol or k not in row_by_mol:
        return [], 0

    base_coords = parent.coords_A
    pJ = base_coords[int(row_by_mol[j])]
    pK = base_coords[int(row_by_mol[k])]
    axis = pK - pJ
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return [], 0
    axis = axis / norm

    step_deg = float(getattr(refine_ligand, "_exo_step_deg", 0.0) or config.coarse_step_deg)
    mol = _clone_with_coords(refine_ligand, base_coords)
    conf = mol.GetConformer(0)
    moved = list(zip(moved_rows.tolist(), moved_mol_ids.tolist()))

    def _apply(angle_deg):
        R = _rotation_matrix(axis * np.deg2rad(float(angle_deg)))
        for row, mol_idx in moved:
            v = base_coords[int(row)]
            vr = (v - pJ) @ R.T + pJ
            conf.SetAtomPosition(int(mol_idx), Point3D(float(vr[0]), float(vr[1]), float(vr[2])))

    n_evals = 0
    coarse_cands = []
    for angle in np.arange(-180.0, 180.0, step_deg):
        _apply(angle)
        coarse_cands.append(
            _candidate_from_conformer(
                refine_ligand, conf, parent, moved_rows, moved_mol_ids,
                frontier_rows, frontier_mol_ids, frontier_elements,
                downstream_rows, fixed_mol_ids, float(angle), config,
            )
        )
        n_evals += 1

    kept_coarse = _filter_coarse(coarse_cands, config)
    if not kept_coarse:
        return [], n_evals

    refined = []
    for coarse in kept_coarse:
        coarse_angle = float(coarse.dihedrals_deg[-1])
        local_cands = []
        for delta in np.arange(
            -config.local_window_deg, config.local_window_deg + 1e-9, config.local_step_deg
        ):
            angle = coarse_angle + float(delta)
            _apply(angle)
            local_cands.append(
                _candidate_from_conformer(
                    refine_ligand, conf, parent, moved_rows, moved_mol_ids,
                    frontier_rows, frontier_mol_ids, frontier_elements,
                    downstream_rows, fixed_mol_ids, float(angle), config,
                )
            )
            n_evals += 1
        refined.append(_best_candidate(local_cands, config))

    return refined, n_evals


def _evaluate_step_for_parent(
    refine_ligand, step, parent,
    moved_rows, moved_mol_ids,
    frontier_rows, frontier_mol_ids, frontier_elements,
    downstream_rows,
    fixed_mol_ids,
    config,
):
    if getattr(step, "is_exo", False):
        # Ring-bond axis: SetDihedralDeg is undefined, so rotate the moved atoms
        # directly about the j-k axis (the exo-swing degree of freedom).
        return _evaluate_exo_step_for_parent(
            refine_ligand, step, parent,
            moved_rows, moved_mol_ids,
            frontier_rows, frontier_mol_ids, frontier_elements,
            downstream_rows, fixed_mol_ids, config,
        )

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
            frontier_rows, frontier_mol_ids, frontier_elements,
            downstream_rows,
            fixed_mol_ids, new_angle, config,
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
                    frontier_rows, frontier_mol_ids, frontier_elements,
                    downstream_rows,
                    fixed_mol_ids, new_angle, config,
                )
            )
            n_evals += 1
        refined.append(_best_candidate(local_cands, config))

    return refined, n_evals


def _candidate_from_conformer(
    refine_ligand, conf, parent,
    moved_rows, moved_mol_ids,
    frontier_rows, frontier_mol_ids, frontier_elements,
    downstream_rows,
    fixed_mol_ids, angle_deg, config,
):
    coords = parent.coords_A.copy()
    for row, mol_idx in zip(moved_rows.tolist(), moved_mol_ids.tolist()):
        p = conf.GetAtomPosition(int(mol_idx))
        coords[row, 0] = float(p.x)
        coords[row, 1] = float(p.y)
        coords[row, 2] = float(p.z)

    frontier_coords = coords[frontier_rows]
    frontier_score = _frontier_qscore(refine_ligand, coords, frontier_rows, config)
    lookahead_weight = float(getattr(config, "downstream_lookahead_weight", 0.0) or 0.0)
    if lookahead_weight != 0.0 and downstream_rows.size > 0:
        downstream_score = _frontier_qscore(
            refine_ligand, coords, downstream_rows, config
        )
        frontier_score = float(frontier_score) + lookahead_weight * float(downstream_score)
    protein_clash = protein_ligand_clash(
        frontier_coords,
        frontier_elements,
        refine_ligand.local_coords_A,
        refine_ligand.local_elements,
        cutoff_A=config.clash_cutoff_A,
        max_vdw_overlap_A=config.max_vdw_overlap_A,
        max_hbond_overlap_A=config.max_hbond_overlap_A,
    )
    self_clash = ligand_self_clash(
        coords,
        refine_ligand._atom_elements,
        getattr(getattr(refine_ligand, "_ligand", None), "mol", None),
        ligand_atom_indices=getattr(refine_ligand, "_atom_indices", None),
        query_atom_indices=frontier_mol_ids,
        reference_atom_indices=fixed_mol_ids,
        cutoff_A=config.clash_cutoff_A,
        max_vdw_overlap_A=config.max_vdw_overlap_A,
        max_hbond_overlap_A=config.max_hbond_overlap_A,
        topological_exclusion_bonds=int(
            getattr(config, "ligand_clash_topological_exclusion_bonds", 3)
        ),
    )
    clash = combine_clash_results(protein_clash, self_clash)
    return _BranchCandidate(
        coords_A=coords,
        dihedrals_deg=parent.dihedrals_deg + (float(angle_deg),),
        frontier_score=float(frontier_score),
        clash_count=int(clash.count),
        clash_penalty=float(clash.penalty),
    )


def _walker_composite(cand, config):
    """Return the candidate's ranking key for the branch walker.

    Without a configured lambda this is the existing frontier-only score; with
    lambda > 0 it folds a penalty term in so candidates with slightly lower
    Qscore but much lower clash penalty stay in the beam.
    """
    lam = getattr(config, "clash_tradeoff_lambda", None)
    if lam is None:
        return float(cand.frontier_score)
    return float(cand.frontier_score) - float(lam) * float(cand.clash_penalty)


def _filter_coarse(cands, config):
    if not cands:
        return []
    lam = getattr(config, "clash_tradeoff_lambda", None)
    scored = [(_walker_composite(c, config), c) for c in cands]
    best_score = max(s for s, _ in scored)
    threshold = best_score - config.coarse_keep_fraction * abs(best_score)
    best_clash_count = min(c.clash_count for c in cands)
    nonzero_penalties = [c.clash_penalty for c in cands if c.clash_count > 0]
    best_clash_penalty = min(nonzero_penalties) if nonzero_penalties else 0.0

    if lam is None:
        kept = [
            c for s, c in scored
            if s >= threshold
            and c.clash_count <= best_clash_count + 1
            and (best_clash_penalty == 0.0 or c.clash_penalty <= 2.0 * best_clash_penalty)
        ]
    else:
        # Under composite ranking the score itself already penalises clashes,
        # so the count and penalty side-conditions are slacker: keep anything
        # whose composite is within the threshold band.
        kept = [c for s, c in scored if s >= threshold]
    if kept:
        return kept
    # Fallback: keep the single best composite/lowest-clash candidate so the
    # walk has something to propagate.
    if lam is None:
        return [min(cands, key=lambda c: (c.clash_count, c.clash_penalty, -c.frontier_score))]
    return [max(scored, key=lambda sc: sc[0])[1]]


def _select_beam(cands, config, iteration=0, rng=None):
    if not cands:
        return []
    lam = getattr(config, "clash_tradeoff_lambda", None)
    # Dedup by rounded dihedral chain so different angles in the same bucket
    # don't all survive. Sort key uses the composite when lambda is set so the
    # beam's ranking agrees with the outer accepter's.
    if lam is None:
        sort_key = lambda c: (c.clash_count, -c.frontier_score, c.clash_penalty)
    else:
        sort_key = lambda c: (-_walker_composite(c, config), c.clash_count, c.clash_penalty)
    seen = set()
    deduped = []
    for c in sorted(cands, key=sort_key):
        key = tuple(round(float(a), 0) for a in c.dihedrals_deg)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    beam_size = int(config.max_keep_per_step)
    if not deduped:
        return []

    sectors = int(getattr(config, "angular_diversity_sectors", 0) or 0)
    if sectors > 0:
        deduped = _apply_angular_diversity(deduped, beam_size, sectors)
    else:
        deduped = deduped[:beam_size]

    temp = getattr(config, "metropolis_temp", None)
    if temp is not None and float(temp) > 0.0 and len(cands) > len(deduped):
        decay = float(getattr(config, "metropolis_decay", 0.7) or 0.7)
        t_iter = float(temp) * (decay ** max(0, int(iteration)))
        if t_iter > 0.0:
            deduped = _metropolis_admit(
                deduped, cands, beam_size, t_iter, rng=rng,
            )
    return deduped


def _apply_angular_diversity(sorted_cands, beam_size, sectors):
    """Keep at least one survivor per occupied angular sector, then fill the
    remaining beam slots by score order. ``sorted_cands`` is already sorted
    best-first."""
    bucket_width = 360.0 / max(1, int(sectors))
    selected = []
    seen_buckets = set()
    leftover = []
    for cand in sorted_cands:
        if not cand.dihedrals_deg:
            leftover.append(cand)
            continue
        last_angle = float(cand.dihedrals_deg[-1])
        bucket = int(((last_angle + 180.0) % 360.0) // bucket_width)
        if bucket not in seen_buckets and len(selected) < beam_size:
            seen_buckets.add(bucket)
            selected.append(cand)
        else:
            leftover.append(cand)
    if len(selected) < beam_size:
        for cand in leftover:
            if len(selected) >= beam_size:
                break
            selected.append(cand)
    return selected


def _metropolis_admit(deterministic_beam, all_cands, beam_size, temperature, rng=None):
    """Optionally swap the weakest seats in the beam for worse candidates
    drawn via Metropolis acceptance, so the search can cross a barrier."""
    if rng is None:
        rng = np.random.default_rng()
    in_beam_ids = {id(c) for c in deterministic_beam}
    outsiders = [c for c in all_cands if id(c) not in in_beam_ids]
    if not outsiders:
        return deterministic_beam
    rng_obj = rng
    beam = list(deterministic_beam)
    best_score = max(c.frontier_score for c in beam)
    # Worst-first so we try to displace the weakest seat.
    weakest_first = sorted(
        range(len(beam)),
        key=lambda i: (beam[i].clash_count, -beam[i].frontier_score, beam[i].clash_penalty),
        reverse=True,
    )
    for slot in weakest_first:
        seat = beam[slot]
        # Random worse candidate.
        idx = int(rng_obj.integers(0, len(outsiders)))
        challenger = outsiders[idx]
        if challenger.clash_count > seat.clash_count:
            continue
        delta = float(best_score) - float(challenger.frontier_score)
        if delta <= 0.0:
            # Challenger is actually better or equal — admit unconditionally.
            beam[slot] = challenger
        else:
            prob = float(np.exp(-delta / float(temperature)))
            if rng_obj.random() < prob:
                beam[slot] = challenger
    return beam


def _best_candidate(cands, config=None):
    lam = getattr(config, "clash_tradeoff_lambda", None) if config is not None else None
    if lam is None:
        return min(cands, key=lambda c: (c.clash_count, -c.frontier_score, c.clash_penalty))
    return max(cands, key=lambda c: _walker_composite(c, config))


def _heavy_rows(refine_ligand, mol_indices):
    rows, _mol_ids = _heavy_rows_and_mol_ids(refine_ligand, mol_indices)
    return rows


def _heavy_rows_and_mol_ids(refine_ligand, mol_indices):
    row_by_mol = refine_ligand._atom_row_by_mol_index
    rows, mol_ids = [], []
    for m in mol_indices:
        mi = int(m)
        if mi in row_by_mol:
            rows.append(int(row_by_mol[mi]))
            mol_ids.append(mi)
    return np.asarray(rows, dtype=int), np.asarray(mol_ids, dtype=int)


def _moved_heavy_rows(refine_ligand, mol_indices):
    return _heavy_rows_and_mol_ids(refine_ligand, mol_indices)


def _fixed_ligand_atom_indices(refine_ligand, moved_atom_indices):
    moved = {int(atom_idx) for atom_idx in moved_atom_indices}
    atom_indices = np.asarray(
        getattr(refine_ligand, "_atom_indices", np.zeros(0, dtype=int)),
        dtype=int,
    ).reshape(-1)
    return np.asarray(
        [int(atom_idx) for atom_idx in atom_indices if int(atom_idx) not in moved],
        dtype=int,
    )


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
        candidate_dirs=int(getattr(refine_ligand, "_qscore_candidate_dirs", 128)),
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
    protein_clash = protein_ligand_clash(
        coords_A,
        refine_ligand._atom_elements,
        refine_ligand.local_coords_A,
        refine_ligand.local_elements,
        cutoff_A=config.clash_cutoff_A,
        max_vdw_overlap_A=config.max_vdw_overlap_A,
        max_hbond_overlap_A=config.max_hbond_overlap_A,
    )
    self_clash = ligand_self_clash(
        coords_A,
        refine_ligand._atom_elements,
        getattr(getattr(refine_ligand, "_ligand", None), "mol", None),
        ligand_atom_indices=getattr(refine_ligand, "_atom_indices", None),
        cutoff_A=config.clash_cutoff_A,
        max_vdw_overlap_A=config.max_vdw_overlap_A,
        max_hbond_overlap_A=config.max_hbond_overlap_A,
        topological_exclusion_bonds=int(
            getattr(config, "ligand_clash_topological_exclusion_bonds", 3)
        ),
    )
    clash = combine_clash_results(protein_clash, self_clash)
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
    best_objective = _objective_from_eval(
        best_raw, clash, config, int(np.asarray(candidate.coords_A).shape[0])
    )
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


def _ring_block_index(rotor_tree, ring_atom_indices):
    """Return the rotor-tree block index whose atoms overlap ``ring_atom_indices``
    most, or None if nothing overlaps."""
    ring_set = set(int(i) for i in ring_atom_indices)
    best_idx, best_overlap = None, 0
    for idx, block in enumerate(rotor_tree):
        overlap = len(ring_set & set(int(a) for a in block.atom_indices))
        if overlap > best_overlap:
            best_idx, best_overlap = idx, overlap
    return best_idx


def _ring_clash(refine_ligand, coords, config):
    """Full protein+self clash for a grafted ring pose (mirrors
    ``_full_ligand_evaluation`` clash handling without rescoring the whole
    ligand)."""
    mol = getattr(getattr(refine_ligand, "_ligand", None), "mol", None)
    protein_clash = protein_ligand_clash(
        coords,
        refine_ligand._atom_elements,
        refine_ligand.local_coords_A,
        refine_ligand.local_elements,
        cutoff_A=config.clash_cutoff_A,
        max_vdw_overlap_A=config.max_vdw_overlap_A,
        max_hbond_overlap_A=config.max_hbond_overlap_A,
    )
    self_clash = ligand_self_clash(
        coords,
        refine_ligand._atom_elements,
        mol,
        ligand_atom_indices=getattr(refine_ligand, "_atom_indices", None),
        cutoff_A=config.clash_cutoff_A,
        max_vdw_overlap_A=config.max_vdw_overlap_A,
        max_hbond_overlap_A=config.max_hbond_overlap_A,
        topological_exclusion_bonds=int(
            getattr(config, "ligand_clash_topological_exclusion_bonds", 3)
        ),
    )
    return combine_clash_results(protein_clash, self_clash)


def ring_conformer_walker(
    refine_ligand, scorer=None, config=None, n_confs=8, min_ring_rmsd=0.3
):
    """Emit FitInMapResult candidates that re-pucker poorly-fit aliphatic rings.

    For each flexible aliphatic ring system whose owning rotor-tree block fits
    the density worse than the best block, generate a small library of
    alternative ring puckers (scaffold/linker geometry preserved), graft each
    onto the current pose, score the ring atoms by Q-score, and keep the top
    ``config.max_keep_per_step`` per ring. Returns the same FitInMapResult type
    as ``branch_walker`` so the existing refit/selection path consumes them
    unchanged. Returns ``[]`` when there is no flexible/poor ring (the common
    case), so it is a cheap no-op for rigid or aromatic-only ligands.
    """
    config = config or BranchWalkConfig()
    scorer = get_scorer(scorer)
    mol = getattr(getattr(refine_ligand, "_ligand", None), "mol", None)
    rotor_tree = getattr(refine_ligand, "_rotor_tree", None)
    row_by_mol = getattr(refine_ligand, "_atom_row_by_mol_index", None)
    if mol is None or rotor_tree is None or row_by_mol is None:
        return []

    ring_systems = aliphatic_ring_systems(mol)
    if not ring_systems:
        return []

    # Per-block Q-scores + best block (gate: only sample rings with room to gain).
    try:
        best_idx = refine_ligand.get_best_block_by_qscore()
    except Exception:
        best_idx = None
    block_q = getattr(refine_ligand, "_block_qscores", None)
    best_q = (
        float(block_q[best_idx])
        if (block_q is not None and best_idx is not None)
        else 1.0
    )

    base_coords = np.asarray(refine_ligand._atom_positions, dtype=np.float64).copy()
    base_score_result, base_clash = _full_ligand_evaluation(
        refine_ligand, scorer, base_coords, config
    )
    initial_raw = float(base_score_result.value)
    initial_objective = _objective_from_eval(
        initial_raw, base_clash, config, int(base_coords.shape[0])
    )

    results = []
    n_evals = 0
    for ring_ids in ring_systems:
        block_idx = _ring_block_index(rotor_tree, ring_ids)
        block_id = int(rotor_tree[block_idx].block_id) if block_idx is not None else -1
        # Skip rings that already fit at least as well as the best block, UNLESS
        # an exocyclic branch hanging off the ring is itself poorly fit — in that
        # case a re-pucker may swing the branch toward its density even though the
        # ring scores fine (e.g. a ribose whose phosphate tail is the problem).
        if (
            block_q is not None
            and block_idx is not None
            and float(block_q[block_idx]) >= best_q
            and not _ring_has_poor_branch(refine_ligand, ring_ids)
        ):
            continue

        ring_rows = np.asarray(
            [int(row_by_mol[int(i)]) for i in ring_ids if int(i) in row_by_mol],
            dtype=int,
        )
        if ring_rows.size < 4:
            continue

        cur_mol = _clone_with_coords(refine_ligand, base_coords)
        confs = generate_ring_conformers(
            cur_mol, ring_ids, n_confs=int(n_confs), min_ring_rmsd=float(min_ring_rmsd)
        )
        cands = []
        for conf in confs:
            coords = base_coords.copy()
            for mol_idx, xyz in conf.items():
                mi = int(mol_idx)
                if mi in row_by_mol:
                    coords[int(row_by_mol[mi])] = xyz
            frontier = _frontier_qscore(refine_ligand, coords, ring_rows, config)
            clash = _ring_clash(refine_ligand, coords, config)
            cands.append(
                _BranchCandidate(
                    coords_A=coords,
                    dihedrals_deg=(),
                    frontier_score=float(frontier),
                    clash_count=int(clash.count),
                    clash_penalty=float(clash.penalty),
                )
            )
            n_evals += 1

        if not cands:
            continue
        cands.sort(
            key=lambda c: (c.clash_count, -c.frontier_score, c.clash_penalty)
        )
        for cand in cands[: int(config.max_keep_per_step)]:
            results.append(
                _build_ring_result(
                    refine_ligand,
                    scorer,
                    cand,
                    base_score_result,
                    base_clash,
                    initial_raw,
                    initial_objective,
                    n_evals,
                    config,
                    block_id,
                )
            )

    if results:
        print(
            "[smart_refine_2] ring-flex produced "
            f"{len(results)} candidate result(s) from {len(ring_systems)} "
            f"aliphatic ring system(s) ({n_evals} evaluations)"
        )
    return results


def _build_ring_result(
    refine_ligand, scorer, candidate,
    base_score_result, base_clash,
    initial_raw, initial_objective, n_evals, config, ring_block_id,
):
    score_result, clash = _full_ligand_evaluation(
        refine_ligand, scorer, candidate.coords_A, config
    )
    best_raw = float(score_result.value)
    best_objective = _objective_from_eval(
        best_raw, clash, config, int(np.asarray(candidate.coords_A).shape[0])
    )
    score_terms = {
        "ring_move": True,
        "ring_block_id": int(ring_block_id),
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
        steps=1,
        evaluations=int(n_evals),
        converged=True,
        final_step_size_A=0.0,
        score_terms=score_terms,
        clash_terms=clash_terms,
        best_rotation_matrix=None,
        best_translation_A=None,
    )


def _downstream_poorly_fit(refine_ligand, down_rows):
    """True when the mean per-atom Q-score over ``down_rows`` is below the best
    block's Q-score — i.e. this branch fits the density worse than the best-fit
    region and is worth moving. Used to gate the exo walker (per torsion) and
    ring-flex (per attached branch)."""
    q = getattr(refine_ligand, "_per_atom_qscores", None)
    down_rows = np.asarray(down_rows, dtype=int)
    if q is None or down_rows.size == 0:
        return False
    q = np.asarray(q, dtype=np.float64)
    try:
        best_idx = refine_ligand.get_best_block_by_qscore()
        best_q = float(refine_ligand._block_qscores[best_idx])
    except Exception:
        best_q = 1.0
    return float(np.mean(q[down_rows])) < best_q


def _ring_has_poor_branch(refine_ligand, ring_ids):
    """True when any exocyclic substituent attached to this ring (via a cached
    exo torsion whose ``k`` ring atom is in ``ring_ids``) is poorly fit. Lets
    ring-flex fire when the ring scores fine but an attached branch does not."""
    exo = getattr(refine_ligand, "_exo_torsions", None)
    row_by_mol = getattr(refine_ligand, "_atom_row_by_mol_index", None)
    if not exo or row_by_mol is None:
        return False
    ring_set = set(int(a) for a in ring_ids)
    for t in exo:
        if int(t.torsion[2]) not in ring_set:  # torsion[2] == k, the ring atom
            continue
        rows = np.asarray(
            [int(row_by_mol[int(a)]) for a in t.downstream_atoms if int(a) in row_by_mol],
            dtype=int,
        )
        if _downstream_poorly_fit(refine_ligand, rows):
            return True
    return False


def _install_drift_anchor(rl, drift_guard):
    """Snapshot the starting-pose anchor for the anti-drift guards onto ``rl``.

    Stores ``_drift_guard`` (or None when disabled) plus the centroid, a cached
    MapGrid, the density threshold and the starting coverage. A no-op (clears the
    guard) when ``drift_guard`` is None/disabled, so every downstream call site
    short-circuits cleanly.
    """
    if drift_guard is None or not getattr(drift_guard, "enabled", False):
        rl._drift_guard = None
        return

    coords = np.asarray(rl._atom_positions, dtype=np.float64)
    rl._drift_guard = drift_guard
    rl._drift_anchor_centroid = coords.mean(axis=0)
    rl._drift_mapgrid = None
    rl._drift_threshold = None
    rl._drift_anchor_coverage = None

    if drift_guard.envelope_gate:
        map_ref = getattr(rl, "_map_reference", None) or getattr(
            rl, "_map_referece", None
        )
        if map_ref is not None:
            try:
                try:
                    from ChemEM.protocols.mapQ_score.mapq_utils import MapGrid
                except ImportError:
                    from protocols.mapQ_score.mapq_utils import MapGrid
                mapgrid = MapGrid.from_emmap(map_ref)
                mean, std = mapgrid.stats()
                threshold = float(mean) + float(
                    drift_guard.envelope_threshold_sigma
                ) * float(std)
                rl._drift_mapgrid = mapgrid
                rl._drift_threshold = threshold
                rl._drift_anchor_coverage = _density_coverage(
                    coords, mapgrid, threshold
                )
            except Exception as exc:
                print(
                    "[smart_refine_2] drift-guard envelope gate disabled "
                    f"(could not build map grid): {exc}"
                )
                rl._drift_mapgrid = None


def _density_coverage(coords_A, mapgrid, threshold):
    """Fraction of heavy atoms sitting in above-``threshold`` density."""
    vals = np.asarray(
        mapgrid.sample_trilinear(np.asarray(coords_A, dtype=np.float64)),
        dtype=np.float64,
    )
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite >= float(threshold)))


def _centroid_penalty(coords_A, anchor_centroid, cfg):
    """Flat-bottom penalty (raw-score units) on centroid displacement."""
    if not getattr(cfg, "centroid_trust", False) or anchor_centroid is None:
        return 0.0
    centroid = np.asarray(coords_A, dtype=np.float64).mean(axis=0)
    d = float(np.linalg.norm(centroid - np.asarray(anchor_centroid, dtype=np.float64)))
    r0 = float(cfg.centroid_radius_A)
    if d <= r0:
        return 0.0
    return float(cfg.centroid_k) * (d - r0)


def _install_covalent_anchor_on_rl(rl, protein_structure, k=0.5, tol_A=0.5):
    """Stash covalent warhead -> fixed-protein-anchor tethers onto ``rl`` so the
    Q-score search never accepts a pose that pulls a warhead off its covalent
    residue. One tether per covalent bond, so a crosslinker is held at both ends.
    No-op unless ``rl._ligand`` carries covalent links, so non-covalent refinement
    is completely unaffected. The protein is pinned during refine, so the anchors
    are fixed points read from the input protein structure.
    """
    rl._covalent_anchors = []          # list of (row, anchor_xyz, r0)
    rl._covalent_warhead_row = None    # first warhead: rigid-body pivot / torsion root
    rl._covalent_anchor_xyz = None
    specs = getattr(getattr(rl, "_ligand", None), "covalent_links", None) or []
    if not specs:
        return
    try:
        from ChemEM.tools.precomputed_data import _covalent_r0_angstrom
    except ImportError:
        from tools.precomputed_data import _covalent_r0_angstrom
    try:
        from ChemEM.parsers.covalent_fragment import _find_complex_atom
    except ImportError:
        from parsers.covalent_fragment import _find_complex_atom

    for spec in specs:
        lig_atom_name = spec.resolved_ligand_atom_name or spec.ligand_atom_spec.split(":")[-1]
        rdkit_idx = rl._ligand.get_atom_idx_from_name(lig_atom_name)
        row = (rl._atom_row_by_mol_index.get(int(rdkit_idx))
               if rdkit_idx is not None else None)
        if row is None:
            print(f"[covalent] smart-refine: warhead '{lig_atom_name}' not a heavy atom; "
                  "drift tether off for this bond.")
            continue

        atom = _find_complex_atom(
            protein_structure,
            spec.resolved_protein_resname,
            spec.resolved_protein_atom_name,
            resnum=spec.resolved_protein_resnum,
            chain=spec.resolved_protein_chain,
        )
        if atom is None:
            print("[covalent] smart-refine: anchor atom not found; drift tether off "
                  "for this bond.")
            continue

        rl._covalent_anchors.append((
            int(row),
            np.array([atom.xx, atom.xy, atom.xz], dtype=np.float64),
            float(_covalent_r0_angstrom(spec)),
        ))

    if not rl._covalent_anchors:
        return

    rl._covalent_k = float(k)
    rl._covalent_tol_A = float(tol_A)
    # The first anchor drives the rigid-body pivot and the torsion-tree root.
    rl._covalent_warhead_row, rl._covalent_anchor_xyz, rl._covalent_r0 = rl._covalent_anchors[0]

    # Pre-anchor: rigid-translate the ligand so the warhead sits at exactly r0 from the
    # anchor. Combined with the warhead-pivot rigid search + warhead-rooted torsions, the
    # warhead then stays at r0 for the whole search (bond length respected, not just held).
    # Only meaningful for a single bond — no translation can satisfy two anchors at
    # once, so with a crosslinker the summed penalty positions the pose instead.
    if len(rl._covalent_anchors) == 1:
        try:
            coords = np.asarray(rl._atom_positions, dtype=np.float64)
            warhead = coords[int(rl._covalent_warhead_row)]
            v = warhead - rl._covalent_anchor_xyz
            n = float(np.linalg.norm(v))
            if n > 1e-9:
                target = rl._covalent_anchor_xyz + rl._covalent_r0 * (v / n)
                rl._atom_positions = coords + (target - warhead)
        except Exception as exc:
            print(f"[covalent] smart-refine: warhead pre-anchor skipped: {exc}")

    for row, xyz, r0 in rl._covalent_anchors:
        print(
            f"[covalent] smart-refine tether: warhead row {row} -> anchor "
            f"({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) r0={r0:.3f} A"
        )


def _covalent_tether_penalty(rl, coords_A):
    """Flat-bottom penalty (raw-score units) keeping each covalent warhead within a
    tolerance of its bond distance to its fixed protein anchor. Penalties over all
    of a ligand's bonds are summed. Returns 0.0 unless a covalent anchor was
    installed on ``rl`` (so non-covalent refinement is bit-for-bit unchanged)."""
    anchors = getattr(rl, "_covalent_anchors", None)
    if not anchors:
        return 0.0
    coords = np.asarray(coords_A, dtype=np.float64)
    tol = float(getattr(rl, "_covalent_tol_A", 0.5))
    k = float(getattr(rl, "_covalent_k", 0.5))

    total = 0.0
    for row, anchor, r0 in anchors:
        if row >= coords.shape[0]:
            continue
        d = float(np.linalg.norm(coords[row] - anchor))
        excess = abs(d - r0) - tol
        if excess > 0.0:
            total += k * excess
    return total


def _drift_guard_of(rl):
    """Return the active DriftGuardConfig snapshotted on ``rl``, or None."""
    cfg = getattr(rl, "_drift_guard", None)
    if cfg is None or not getattr(cfg, "enabled", False):
        return None
    return cfg


def _drift_assess(rl, coords_A):
    """Return ``(centroid_penalty, passes_envelope)`` for a candidate pose.

    ``(0.0, True)`` when no guard is active, so callers can always apply the
    result unconditionally.
    """
    # Covalent tether is always active for covalent ligands, independent of the
    # optional anti-drift guards. 0.0 for non-covalent ligands -> unchanged.
    penalty = _covalent_tether_penalty(rl, coords_A)

    cfg = _drift_guard_of(rl)
    if cfg is None:
        return float(penalty), True

    if cfg.centroid_trust:
        penalty += _centroid_penalty(
            coords_A, getattr(rl, "_drift_anchor_centroid", None), cfg
        )

    passes = True
    if cfg.envelope_gate:
        mapgrid = getattr(rl, "_drift_mapgrid", None)
        threshold = getattr(rl, "_drift_threshold", None)
        anchor_cov = getattr(rl, "_drift_anchor_coverage", None)
        if mapgrid is not None and threshold is not None and anchor_cov is not None:
            cov = _density_coverage(coords_A, mapgrid, threshold)
            passes = bool(cov >= float(anchor_cov) - float(cfg.envelope_slack))

    return float(penalty), bool(passes)


def _refinement_composite_delta(result, clash_tradeoff_lambda):
    """Return delta of (raw_score - lambda * clash_penalty) for a FitInMapResult."""
    lam = float(clash_tradeoff_lambda)
    delta_raw = float(result.best_raw_score) - float(result.initial_raw_score)
    delta_penalty = float(result.best_clash_penalty) - float(result.initial_clash_penalty)
    return delta_raw - lam * delta_penalty


def should_accept_refinement(
    result,
    min_score_improvement=0.0,
    max_clash_penalty_increase=1e-6,
    clash_tradeoff_lambda=None,
    refine_ligand=None,
):
    if not np.isfinite(float(result.best_raw_score)):
        return False

    # Anti-drift guards (no-op unless a guard is active on ``refine_ligand``).
    drift_penalty = 0.0
    if refine_ligand is not None:
        drift_penalty, passes_envelope = _drift_assess(
            refine_ligand, result.best_coords_A
        )
        if not passes_envelope:
            return False

    if clash_tradeoff_lambda is not None and float(clash_tradeoff_lambda) >= 0.0:
        # Composite gate: accept iff (raw - lambda * clash_penalty) strictly
        # improves. The two independent ratchets (raw monotonicity + penalty
        # ratchet) are replaced by a single trade-off, so a small raw
        # regression is OK if the clash penalty drops enough, and a small
        # penalty increase is OK if the raw improves enough. The centroid
        # penalty raises the bar a moving pose must clear.
        delta_composite = _refinement_composite_delta(result, clash_tradeoff_lambda)
        return (delta_composite - drift_penalty) > float(min_score_improvement)

    if (float(result.best_raw_score) - drift_penalty) <= (
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
    clash_tradeoff_lambda=None,
):
    composite_suffix = ""
    if clash_tradeoff_lambda is not None and float(clash_tradeoff_lambda) >= 0.0:
        delta_comp = _refinement_composite_delta(result, clash_tradeoff_lambda)
        composite_suffix = f" delta_composite={delta_comp:+.5f} lambda={float(clash_tradeoff_lambda):.4f}"
    if not should_accept_refinement(
        result,
        min_score_improvement=min_score_improvement,
        max_clash_penalty_increase=max_clash_penalty_increase,
        clash_tradeoff_lambda=clash_tradeoff_lambda,
        refine_ligand=refine_ligand,
    ):
        print(
            "[smart_refine_2] rejected fit-in-map "
            f"delta_raw={result.delta_raw_score:+.5f} "
            f"clashes {result.initial_clash_count}->{result.best_clash_count} "
            f"clash_penalty {result.initial_clash_penalty:.5f}->{result.best_clash_penalty:.5f}"
            f"{composite_suffix}"
        )
        return refine_ligand

    print(
        "[smart_refine_2] accepted fit-in-map "
        f"delta_raw={result.delta_raw_score:+.5f} "
        f"clashes {result.initial_clash_count}->{result.best_clash_count} "
        f"clash_penalty {result.initial_clash_penalty:.5f}->{result.best_clash_penalty:.5f}"
        f"{composite_suffix}"
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
