from __future__ import annotations

from collections import defaultdict, deque
import heapq
from dataclasses import dataclass
from typing import Iterable, Literal

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, rdMolTransforms


BlockRole = Literal[
    "core",
    "path",
    "branch",
    "branch_bundle",
    "tail",
    "linker",
    "orphan",
]


@dataclass(frozen=True)
class BlockTorsion:
    """
    A torsion associated with one semantic block.

    The axis is always oriented j -> k.  Apply rotations only to
    moved_atom_indices, which is the full heavy-atom side of the ligand graph
    after cutting j-k.  The block owns this torsion as a semantic waypoint; the
    move may include atoms from multiple semantic blocks.
    """

    order: int
    dihedral: tuple[int, int, int, int]  # i-j-k-l
    axis: tuple[int, int]  # j-k, fixed side -> moved side
    bond_index: int
    moved_atom_indices: tuple[int, ...]
    fixed_atom_indices: tuple[int, ...]
    moved_block_ids: tuple[int, ...] = ()

    def __repr__(self) -> str:
        return (
            "BlockTorsion("
            f"order={self.order}, "
            f"rdkit_torsion={self.dihedral}, "
            f"coord_moves={self.moved_atom_indices}, "
            f"moved_blocks={self.moved_block_ids}"
            ")"
        )


@dataclass(frozen=True)
class SemanticBlock:
    block_id: int
    role: BlockRole
    atom_indices: tuple[int, ...]
    anchor_atoms: tuple[int, ...]
    parent: int | None
    children: tuple[int, ...]
    torsions: tuple[BlockTorsion, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class SemanticTorsionStep:
    order: int
    owner_block_id: int
    torsion: BlockTorsion
    moved_atom_indices: tuple[int, ...]
    frontier_atom_indices: tuple[int, ...]
    moved_block_ids: tuple[int, ...] = ()

    @property
    def axis(self) -> tuple[int, int]:
        return self.torsion.axis

    @property
    def dihedral(self) -> tuple[int, int, int, int]:
        return self.torsion.dihedral

    @property
    def rdkit_torsion(self) -> tuple[int, int, int, int]:
        return self.torsion.dihedral

    @property
    def bond_index(self) -> int:
        return self.torsion.bond_index

    def __repr__(self) -> str:
        return (
            "SemanticTorsionStep("
            f"order={self.order}, "
            f"rdkit_torsion={self.rdkit_torsion}, "
            f"frontier={self.frontier_atom_indices}, "
            f"axis={self.axis}, "
            f"owner_block={self.owner_block_id}, "
            f"moved_blocks={self.moved_block_ids}, "
            f"coord_moves={self.moved_atom_indices}"
            ")"
        )


@dataclass(frozen=True)
class SemanticTorsionWalk:
    walk_id: int
    root_block_id: int
    target_block_ids: tuple[int, ...]
    block_route: tuple[int, ...]
    frontier_atom_indices: tuple[int, ...]
    steps: tuple[SemanticTorsionStep, ...]

    @property
    def axes(self) -> tuple[tuple[int, int], ...]:
        return tuple(step.axis for step in self.steps)

    @property
    def dihedrals(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(step.dihedral for step in self.steps)

    @property
    def rdkit_torsions(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(step.rdkit_torsion for step in self.steps)

    @property
    def frontier_by_torsion(
        self,
    ) -> tuple[tuple[tuple[int, int, int, int], tuple[int, ...]], ...]:
        return tuple(
            (step.rdkit_torsion, step.frontier_atom_indices)
            for step in self.steps
        )

    def __repr__(self) -> str:
        return (
            "SemanticTorsionWalk("
            f"walk_id={self.walk_id}, "
            f"root_block_id={self.root_block_id}, "
            f"target_block_ids={self.target_block_ids}, "
            f"block_route={self.block_route}, "
            f"frontier_by_torsion={self.frontier_by_torsion}, "
            f"steps={self.steps}"
            ")"
        )


def build_semantic_anchor_blocks(
    mol: Chem.Mol,
    *,
    root_atom_idx: int | None = None,
    keep_p_s_tails_intact: bool = True,
    include_torsions: bool = True,
    exclude_amides: bool = True,
    exclude_esters: bool = False,
    exclude_conjugated: bool = False,
) -> tuple[SemanticBlock, ...]:
    """
    Build semantic anchor-inclusive ligand blocks and attach group torsions.

    This is a single workflow:
      1. Build semantic blocks.
      2. Infer parent/child anchors between blocks.
      3. Find directed rotatable torsions from the whole heavy-atom ligand graph.
      4. Attach each torsion to the best semantic owner block.

    Blocks may share anchor/branch-point atoms.  Torsions are owned by one block,
    but moved_atom_indices may include atoms from multiple connected blocks.
    """

    if mol is None:
        raise ValueError("mol is None")

    heavy_atoms, adj = _heavy_adjacency(mol)

    if not heavy_atoms:
        return ()

    ring_systems = _ring_systems(mol)
    specs: list[dict] = []
    assigned: set[int] = set()

    # ---- 1. All ring systems become core blocks first ----
    for ring_atoms in sorted(ring_systems, key=lambda x: (-len(x), min(x))):
        atoms = set(ring_atoms)
        atoms |= _terminal_hetero_substituents(ring_atoms, adj, mol)

        specs.append(
            {
                "role": "core",
                "atoms": atoms,
                "anchor_atoms": set(),
                "reason": "ring system plus terminal hetero substituents",
            }
        )
        assigned |= atoms

    # ---- Fallback for fully acyclic ligands ----
    if not specs:
        root = root_atom_idx if root_atom_idx is not None else _graph_center(heavy_atoms, adj)
        specs.append(
            {
                "role": "core",
                "atoms": {root},
                "anchor_atoms": set(),
                "reason": "acyclic fallback root atom",
            }
        )
        assigned.add(root)

    # ---- 2. Acyclic components outside cores ----
    residual_atoms = heavy_atoms - assigned
    residual_components = _connected_components(residual_atoms, adj)

    for comp in residual_components:
        core_anchors = {
            nbr
            for atom_idx in comp
            for nbr in adj[atom_idx]
            if nbr in assigned
        }

        if not core_anchors:
            specs.append(
                {
                    "role": "orphan",
                    "atoms": set(comp),
                    "anchor_atoms": set(),
                    "reason": "disconnected acyclic component",
                }
            )
            continue

        if keep_p_s_tails_intact and _contains_p_or_s_oxo_tail(mol, comp):
            specs.append(
                {
                    "role": "tail",
                    "atoms": set(comp) | core_anchors,
                    "anchor_atoms": set(core_anchors),
                    "reason": "kept intact because component contains P/S oxo tail chemistry",
                }
            )
            continue

        if len(core_anchors) >= 2:
            specs.append(
                {
                    "role": "linker",
                    "atoms": set(comp) | core_anchors,
                    "anchor_atoms": set(core_anchors),
                    "reason": "acyclic component connects multiple core blocks",
                }
            )
            continue

        anchor = next(iter(core_anchors))
        specs.extend(
            _split_single_anchor_acyclic_component(
                mol=mol,
                adj=adj,
                component=set(comp),
                anchor=anchor,
            )
        )

    # ---- 3. Orient semantic block graph ----
    block_atom_sets = [set(s["atoms"]) for s in specs]
    block_adj = _build_semantic_block_graph(mol, block_atom_sets)
    root_block = _choose_semantic_root_block(specs=specs, root_atom_idx=root_atom_idx)
    parent, children = _orient_semantic_blocks(block_adj, root_block)

    # ---- 4. Infer parent anchors for every block ----
    blocks_without_torsions: list[SemanticBlock] = []
    for block_id, spec in enumerate(specs):
        atoms = set(spec["atoms"])
        anchors = set(spec["anchor_atoms"])
        parent_id = parent.get(block_id)

        if parent_id is not None:
            anchors |= _connection_atoms_in_child(
                mol=mol,
                child_atoms=atoms,
                parent_atoms=block_atom_sets[parent_id],
            )

        blocks_without_torsions.append(
            SemanticBlock(
                block_id=block_id,
                role=spec["role"],
                atom_indices=tuple(sorted(atoms)),
                anchor_atoms=tuple(sorted(anchors)),
                parent=parent_id,
                children=tuple(sorted(children.get(block_id, ()))),
                torsions=(),
                reason=spec["reason"],
            )
        )

    blocks = tuple(blocks_without_torsions)

    if not include_torsions:
        return blocks

    return _attach_block_torsions(
        mol=mol,
        blocks=blocks,
        adj=adj,
        exclude_amides=exclude_amides,
        exclude_esters=exclude_esters,
        exclude_conjugated=exclude_conjugated,
    )


def build_directional_torsion_walks(
    mol: Chem.Mol,
    blocks: tuple[SemanticBlock, ...],
    root_block_id: int,
    target_block_ids: Iterable[int] | None = None,
) -> tuple[SemanticTorsionWalk, ...]:
    """
    Build one-way torsion walks expanding out from a trusted semantic block.

    Raw BlockTorsions intentionally keep both possible cut directions.  This
    helper filters those raw torsions into monotonic walks for branch/path
    rebuilding: no step may move atoms owned by the trusted root, and each later
    step must move a strict subset of the previous step.
    """

    if mol is None:
        raise ValueError("mol is None")

    if not blocks:
        return ()

    block_by_id = {int(block.block_id): block for block in blocks}
    root_block_id = int(root_block_id)
    if root_block_id not in block_by_id:
        raise ValueError(f"root_block_id {root_block_id} is not present in blocks")

    if target_block_ids is None:
        targets = tuple(
            block.block_id for block in blocks if block.block_id != root_block_id
        )
    else:
        targets = tuple(
            sorted(
                {
                    int(block_id)
                    for block_id in target_block_ids
                    if int(block_id) in block_by_id and int(block_id) != root_block_id
                }
            )
        )

    if not targets:
        return ()

    heavy_atoms, adj = _heavy_adjacency(mol)
    block_atom_sets = [set(block.atom_indices) for block in blocks]
    block_branch_points = [
        _block_branch_points(mol=mol, block_id=i, block_atom_sets=block_atom_sets)
        for i in range(len(blocks))
    ]
    block_owned_atoms = [
        _block_owned_atoms(
            block_id=i,
            block_atom_sets=block_atom_sets,
            block_branch_points=block_branch_points,
        )
        for i in range(len(blocks))
    ]

    root_protected_atoms = set(block_by_id[root_block_id].atom_indices)
    if not root_protected_atoms:
        root_protected_atoms = set(block_owned_atoms[root_block_id])

    target_owned_by_block = {
        block_id: set(block_owned_atoms[block_id]) or set(block_by_id[block_id].atom_indices)
        for block_id in targets
    }

    target_owned_atoms: set[int] = set()
    for atoms in target_owned_by_block.values():
        target_owned_atoms |= atoms

    if not target_owned_atoms:
        return ()

    root_distance = _multi_source_distances(
        nodes=heavy_atoms,
        adj=adj,
        roots=root_protected_atoms,
    )
    records = _directional_torsion_records(
        mol=mol,
        blocks=blocks,
        block_atom_sets=block_atom_sets,
        block_branch_points=block_branch_points,
        root_protected_atoms=root_protected_atoms,
        target_owned_atoms=target_owned_atoms,
        root_distance=root_distance,
    )
    if not records:
        return ()

    block_adj = _build_semantic_block_graph(mol, block_atom_sets)
    parent, _children = _orient_semantic_blocks(block_adj, root_block_id)

    raw_chains: list[tuple[dict, ...]] = []
    seen_sequences: set[tuple[tuple[int, int], ...]] = set()
    for target_id in targets:
        target_records = [
            record
            for record in records
            if record["moved"] & target_owned_by_block[target_id]
        ]
        chain = _select_monotonic_torsion_chain(target_records)
        if not chain:
            continue

        sequence = tuple(record["torsion"].axis for record in chain)
        if sequence in seen_sequences:
            continue

        seen_sequences.add(sequence)
        raw_chains.append(tuple(chain))

    walks: list[SemanticTorsionWalk] = []
    for raw_chain in raw_chains:
        steps = _semantic_torsion_steps_from_chain(raw_chain)
        if not steps:
            continue

        frontier_atoms = tuple(
            sorted(
                {
                    atom_idx
                    for step in steps
                    for atom_idx in step.frontier_atom_indices
                }
            )
        )
        covered_targets = tuple(
            block_id
            for block_id in targets
            if set(frontier_atoms) & target_owned_by_block[block_id]
        )
        if not covered_targets:
            continue

        # A shorter chain that is only a prefix of a longer chain is redundant
        # when the longer chain already exposes all of its target atoms as
        # frontier targets.
        if _walk_is_redundant_prefix(
            steps=steps,
            covered_targets=covered_targets,
            candidate_chains=raw_chains,
            target_owned_by_block=target_owned_by_block,
        ):
            continue

        walks.append(
            SemanticTorsionWalk(
                walk_id=-1,
                root_block_id=root_block_id,
                target_block_ids=covered_targets,
                block_route=_combined_block_route(
                    parent=parent,
                    root_block_id=root_block_id,
                    target_block_ids=covered_targets,
                ),
                frontier_atom_indices=frontier_atoms,
                steps=steps,
            )
        )

    walks.sort(
        key=lambda walk: (
            -len(walk.steps[0].moved_atom_indices) if walk.steps else 0,
            walk.block_route,
            walk.target_block_ids,
        )
    )

    return tuple(
        SemanticTorsionWalk(
            walk_id=i,
            root_block_id=walk.root_block_id,
            target_block_ids=walk.target_block_ids,
            block_route=walk.block_route,
            frontier_atom_indices=walk.frontier_atom_indices,
            steps=walk.steps,
        )
        for i, walk in enumerate(walks)
    )


def print_semantic_blocks(mol: Chem.Mol, blocks: tuple[SemanticBlock, ...]) -> None:
    for block in blocks:
        labels = [f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in block.atom_indices]
        print(
            f"B{block.block_id} "
            f"role={block.role} "
            f"atoms={block.atom_indices} "
            f"labels={labels} "
            f"anchors={block.anchor_atoms} "
            f"parent={block.parent} "
            f"children={block.children} "
            f"reason={block.reason}"
        )

        for torsion in block.torsions:
            print(
                f"  T{torsion.order}: "
                f"dihedral={torsion.dihedral} "
                f"axis={torsion.axis} "
                f"moves={torsion.moved_atom_indices} "
                f"moved_blocks={torsion.moved_block_ids} "
                f"fixed={torsion.fixed_atom_indices}"
            )


def draw_semantic_blocks_grid(
    mol: Chem.Mol,
    blocks: tuple[SemanticBlock, ...],
    *,
    filename: str | None = None,
    mols_per_row: int = 2,
    sub_img_size: tuple[int, int] = (500, 380),
    show_atom_indices: bool = True,
):
    """
    Draw one panel per block.

    One panel per block is better than one combined plot because anchor-inclusive
    blocks can share atoms.  Atom labels ending in * are parent anchors.
    """

    base = Chem.Mol(mol)

    if base.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(base)

    mols = []
    legends = []
    highlight_atom_lists = []
    highlight_bond_lists = []

    for block in blocks:
        draw_mol = Chem.Mol(base)
        block_atoms = set(block.atom_indices)
        anchor_atoms = set(block.anchor_atoms)

        if show_atom_indices:
            for atom in draw_mol.GetAtoms():
                idx = atom.GetIdx()
                if idx in block_atoms:
                    label = str(idx)
                    if idx in anchor_atoms:
                        label += "*"
                    atom.SetProp("atomNote", label)

        highlight_bonds = []
        for bond in draw_mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            if a in block_atoms and b in block_atoms:
                highlight_bonds.append(bond.GetIdx())

        mols.append(draw_mol)
        legends.append(f"B{block.block_id} {block.role}: {block.atom_indices}")
        highlight_atom_lists.append(list(block.atom_indices))
        highlight_bond_lists.append(highlight_bonds)

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=legends,
        highlightAtomLists=highlight_atom_lists,
        highlightBondLists=highlight_bond_lists,
        useSVG=False,
    )

    if filename is not None:
        img.save(filename)

    return img


def _attach_block_torsions(
    *,
    mol: Chem.Mol,
    blocks: tuple[SemanticBlock, ...],
    adj: dict[int, set[int]],
    exclude_amides: bool,
    exclude_esters: bool,
    exclude_conjugated: bool,
) -> tuple[SemanticBlock, ...]:
    block_atom_sets = [set(b.atom_indices) for b in blocks]
    block_branch_points = [
        _block_branch_points(mol=mol, block_id=i, block_atom_sets=block_atom_sets)
        for i in range(len(blocks))
    ]
    heavy_atoms = set(adj)
    block_distances = [
        _block_torsion_distances(
            block=block,
            block_branch_points=block_branch_points[block.block_id],
            adj=adj,
        )
        for block in blocks
    ]
    torsions_by_owner: dict[int, list[BlockTorsion]] = defaultdict(list)

    for bond in mol.GetBonds():
        if not _is_rotatable_heavy_bond(
            mol=mol,
            bond=bond,
            adj=adj,
            exclude_amides=exclude_amides,
            exclude_esters=exclude_esters,
            exclude_conjugated=exclude_conjugated,
        ):
            continue

        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        for j, k in ((a, b), (b, a)):
            moved = _component_after_cut(
                adj=adj,
                start=k,
                cut=(j, k),
                allowed=heavy_atoms,
            )
            owner_id = _choose_torsion_owner_block(
                k=k,
                moved=moved,
                blocks=blocks,
                block_atom_sets=block_atom_sets,
            )

            if owner_id is None:
                continue

            axis = (j, k)
            i = _choose_dihedral_anchor_atom(adj=adj, center=j, axis_other=k, moved=moved)
            l = _choose_dihedral_moved_atom(adj=adj, center=k, axis_other=j, moved=moved)

            if i is None or l is None:
                continue

            dihedral = (i, j, k, l)
            _validate_block_torsion(mol=mol, dihedral=dihedral, axis=axis, moved=moved)

            moved_block_ids = _moved_block_ids(
                moved=moved,
                block_atom_sets=block_atom_sets,
                block_branch_points=block_branch_points,
            )

            torsions_by_owner[owner_id].append(
                BlockTorsion(
                    order=-1,
                    dihedral=dihedral,
                    axis=axis,
                    bond_index=bond.GetIdx(),
                    moved_atom_indices=tuple(sorted(moved)),
                    fixed_atom_indices=tuple(sorted(heavy_atoms - moved)),
                    moved_block_ids=moved_block_ids,
                )
            )

    out = []
    for block in blocks:
        torsions = list(torsions_by_owner.get(block.block_id, ()))
        distances = block_distances[block.block_id]
        torsions.sort(
            key=lambda t: (
                _torsion_depth(t, distances),
                -len(t.moved_atom_indices),
                t.axis,
            )
        )
        ordered_torsions = tuple(
            BlockTorsion(
                order=i,
                dihedral=t.dihedral,
                axis=t.axis,
                bond_index=t.bond_index,
                moved_atom_indices=t.moved_atom_indices,
                fixed_atom_indices=t.fixed_atom_indices,
                moved_block_ids=t.moved_block_ids,
            )
            for i, t in enumerate(torsions)
        )

        out.append(
            SemanticBlock(
                block_id=block.block_id,
                role=block.role,
                atom_indices=block.atom_indices,
                anchor_atoms=block.anchor_atoms,
                parent=block.parent,
                children=block.children,
                torsions=ordered_torsions,
                reason=block.reason,
            )
        )

    return tuple(out)


def _block_torsion_distances(
    *,
    block: SemanticBlock,
    block_branch_points: set[int],
    adj: dict[int, set[int]],
) -> dict[int, int]:
    block_atoms = set(block.atom_indices)
    root_atoms = set(block.anchor_atoms) & block_atoms

    if not root_atoms:
        root_atoms = set(block_branch_points) & block_atoms
    if not root_atoms:
        root_atoms = {_graph_center(block_atoms, adj)}

    return _multi_source_distances(nodes=block_atoms, adj=adj, roots=root_atoms)


def _choose_torsion_owner_block(
    *,
    k: int,
    moved: set[int],
    blocks: tuple[SemanticBlock, ...],
    block_atom_sets: list[set[int]],
) -> int | None:
    candidates = []

    for block in blocks:
        atoms = block_atom_sets[block.block_id]

        if k not in atoms:
            continue

        overlap = len(atoms & moved)
        if overlap <= 0:
            continue

        candidates.append(
            (
                -overlap,
                _owner_role_priority(block.role),
                block.block_id,
            )
        )

    if not candidates:
        return None

    return min(candidates)[2]


def _owner_role_priority(role: BlockRole) -> int:
    priorities = {
        "path": 0,
        "core": 1,
        "linker": 2,
        "tail": 3,
        "branch_bundle": 4,
        "branch": 5,
        "orphan": 6,
    }
    return priorities.get(role, 100)


def _moved_block_ids(
    *,
    moved: set[int],
    block_atom_sets: list[set[int]],
    block_branch_points: list[set[int]],
) -> tuple[int, ...]:
    moved_ids = []

    for block_id, atoms in enumerate(block_atom_sets):
        block_owned_atoms = atoms - block_branch_points[block_id]

        if not block_owned_atoms:
            block_owned_atoms = atoms

        if moved & block_owned_atoms:
            moved_ids.append(block_id)

    return tuple(moved_ids)


def _block_owned_atoms(
    *,
    block_id: int,
    block_atom_sets: list[set[int]],
    block_branch_points: list[set[int]],
) -> set[int]:
    atoms = set(block_atom_sets[block_id])
    owned_atoms = atoms - set(block_branch_points[block_id])
    return owned_atoms if owned_atoms else atoms


def _directional_torsion_records(
    *,
    mol: Chem.Mol,
    blocks: tuple[SemanticBlock, ...],
    block_atom_sets: list[set[int]],
    block_branch_points: list[set[int]],
    root_protected_atoms: set[int],
    target_owned_atoms: set[int],
    root_distance: dict[int, int],
) -> list[dict]:
    records = []

    for block in blocks:
        for torsion in block.torsions:
            graph_moved = set(torsion.moved_atom_indices)
            coord_moved = _coordinate_moving_atoms_for_torsion(
                mol=mol,
                torsion=torsion,
                candidate_atoms=graph_moved,
            )

            if not coord_moved:
                continue

            if coord_moved & root_protected_atoms:
                continue

            if not (coord_moved & target_owned_atoms):
                continue

            axis_distance = min(
                root_distance.get(torsion.axis[0], 10**9),
                root_distance.get(torsion.axis[1], 10**9),
            )
            records.append(
                {
                    "owner_block_id": int(block.block_id),
                    "torsion": torsion,
                    "moved": coord_moved,
                    "graph_moved": graph_moved,
                    "moved_block_ids": _moved_block_ids(
                        moved=coord_moved,
                        block_atom_sets=block_atom_sets,
                        block_branch_points=block_branch_points,
                    ),
                    "axis_distance": int(axis_distance),
                }
            )

    records.sort(
        key=lambda record: (
            -len(record["moved"]),
            record["axis_distance"],
            record["owner_block_id"],
            record["torsion"].order,
            record["torsion"].axis,
        )
    )
    return records


def _select_monotonic_torsion_chain(records: list[dict]) -> tuple[dict, ...]:
    chain: list[dict] = []
    previous_moved: set[int] | None = None
    seen_axes: set[tuple[int, int]] = set()

    for record in records:
        axis = record["torsion"].axis
        moved = set(record["moved"])

        if axis in seen_axes:
            continue

        if previous_moved is not None:
            if not moved < previous_moved:
                continue

        chain.append(record)
        seen_axes.add(axis)
        previous_moved = moved

    return tuple(chain)


def _coordinate_moving_atoms_for_torsion(
    *,
    mol: Chem.Mol,
    torsion: BlockTorsion,
    candidate_atoms: set[int],
    delta_deg: float = 20.0,
    tolerance_A: float = 1e-5,
) -> set[int]:
    topology_moved = set(candidate_atoms) - set(torsion.axis)

    if mol is None or mol.GetNumConformers() == 0:
        return topology_moved

    try:
        work = Chem.Mol(mol)
        conf = work.GetConformer()
        before = {
            atom_idx: conf.GetAtomPosition(int(atom_idx))
            for atom_idx in topology_moved
            if 0 <= int(atom_idx) < work.GetNumAtoms()
        }
        current = rdMolTransforms.GetDihedralDeg(conf, *torsion.dihedral)
        rdMolTransforms.SetDihedralDeg(
            conf,
            *torsion.dihedral,
            float(current) + float(delta_deg),
        )
    except Exception:
        return topology_moved

    moving = set()
    n_atoms = work.GetNumAtoms()
    for atom_idx in sorted(topology_moved):
        if atom_idx < 0 or atom_idx >= n_atoms:
            continue

        before_xyz = before[atom_idx]
        after_xyz = conf.GetAtomPosition(int(atom_idx))
        dx = float(after_xyz.x - before_xyz.x)
        dy = float(after_xyz.y - before_xyz.y)
        dz = float(after_xyz.z - before_xyz.z)
        if (dx * dx + dy * dy + dz * dz) ** 0.5 > float(tolerance_A):
            moving.add(int(atom_idx))

    return moving


def _semantic_torsion_steps_from_chain(
    chain: tuple[dict, ...],
) -> tuple[SemanticTorsionStep, ...]:
    steps: list[SemanticTorsionStep] = []

    for order, record in enumerate(chain):
        moved = set(record["moved"])
        next_moved = set(chain[order + 1]["moved"]) if order + 1 < len(chain) else set()
        target_atoms = moved - next_moved

        if not target_atoms:
            continue

        torsion = record["torsion"]
        steps.append(
            SemanticTorsionStep(
                order=len(steps),
                owner_block_id=int(record["owner_block_id"]),
                torsion=torsion,
                moved_atom_indices=tuple(sorted(moved)),
                frontier_atom_indices=tuple(sorted(target_atoms)),
                moved_block_ids=tuple(record["moved_block_ids"]),
            )
        )

    return tuple(steps)


def _walk_is_redundant_prefix(
    *,
    steps: tuple[SemanticTorsionStep, ...],
    covered_targets: tuple[int, ...],
    candidate_chains: list[tuple[dict, ...]],
    target_owned_by_block: dict[int, set[int]],
) -> bool:
    sequence = tuple(step.torsion.axis for step in steps)
    covered_owned_atoms: set[int] = set()
    for block_id in covered_targets:
        covered_owned_atoms |= set(target_owned_by_block[block_id])

    for chain in candidate_chains:
        chain_sequence = tuple(record["torsion"].axis for record in chain)

        if len(chain_sequence) <= len(sequence):
            continue

        if chain_sequence[: len(sequence)] != sequence:
            continue

        longer_steps = _semantic_torsion_steps_from_chain(chain)
        longer_target_atoms = {
            atom_idx
            for step in longer_steps
            for atom_idx in step.frontier_atom_indices
        }
        if covered_owned_atoms <= longer_target_atoms:
            return True

    return False


def _combined_block_route(
    *,
    parent: dict[int, int | None],
    root_block_id: int,
    target_block_ids: tuple[int, ...],
) -> tuple[int, ...]:
    route: list[int] = []

    for target_block_id in target_block_ids:
        target_route = _block_route_from_parent(
            parent=parent,
            root_block_id=root_block_id,
            target_block_id=target_block_id,
        )

        for block_id in target_route:
            if block_id not in route:
                route.append(block_id)

    return tuple(route)


def _block_route_from_parent(
    *,
    parent: dict[int, int | None],
    root_block_id: int,
    target_block_id: int,
) -> tuple[int, ...]:
    route = []
    current: int | None = int(target_block_id)
    seen: set[int] = set()

    while current is not None and current not in seen:
        seen.add(current)
        route.append(current)
        if current == root_block_id:
            break
        current = parent.get(current)

    if not route or route[-1] != root_block_id:
        return (root_block_id, target_block_id)

    return tuple(reversed(route))


def _allowed_parent_boundary_axes(
    *,
    block: SemanticBlock,
    blocks: tuple[SemanticBlock, ...],
    adj: dict[int, set[int]],
) -> set[tuple[int, int]]:
    """
    Return allowed outside->inside axes for boundary torsions into block.

    For non-overlap parent/child blocks, this is the real bond connecting them.
    For overlap child blocks, only one parent-side predecessor is allowed for each
    shared anchor: the one closest to the parent block's own anchor/root.  This
    prevents duplicate axes such as both ring-C* and chain-C* for a terminal OH
    branch attached at C*.
    """

    if block.parent is None:
        return set()

    parent = blocks[block.parent]
    block_atoms = set(block.atom_indices)
    parent_atoms = set(parent.atom_indices)
    parent_roots = set(parent.anchor_atoms) & parent_atoms

    if not parent_roots:
        parent_roots = {_graph_center(parent_atoms, adj)}

    parent_dist = _multi_source_distances(nodes=parent_atoms, adj=adj, roots=parent_roots)
    axes: set[tuple[int, int]] = set()

    for inside in block.anchor_atoms:
        if inside not in block_atoms:
            continue

        # Non-overlap case: parent atom is outside this block.
        outside_candidates = [
            nbr
            for nbr in adj.get(inside, ())
            if nbr in parent_atoms and nbr not in block_atoms
        ]

        # Overlap case: the anchor atom is shared with the parent, so the axis
        # uses the parent-side predecessor attached to the shared anchor.
        if not outside_candidates and inside in parent_atoms:
            outside_candidates = [
                nbr
                for nbr in adj.get(inside, ())
                if nbr in parent_atoms and nbr != inside
            ]

        if not outside_candidates:
            continue

        outside = min(outside_candidates, key=lambda n: (parent_dist.get(n, 10**9), n))
        axes.add((outside, inside))

    return axes


def _torsion_depth(torsion: BlockTorsion, distances: dict[int, int]) -> int:
    j, k = torsion.axis
    return distances.get(k, distances.get(j, 10**9))


def _validate_block_torsion(
    *,
    mol: Chem.Mol,
    dihedral: tuple[int, int, int, int],
    axis: tuple[int, int],
    moved: set[int],
) -> None:
    i, j, k, l = dihedral

    if (j, k) != axis:
        raise AssertionError(f"dihedral {dihedral} does not match axis {axis}")

    if mol.GetBondBetweenAtoms(i, j) is None:
        raise AssertionError(f"invalid torsion: {i}-{j} is not a bond")

    if mol.GetBondBetweenAtoms(j, k) is None:
        raise AssertionError(f"invalid torsion: {j}-{k} is not a bond")

    if mol.GetBondBetweenAtoms(k, l) is None:
        raise AssertionError(f"invalid torsion: {k}-{l} is not a bond")

    if j in moved:
        raise AssertionError(f"axis anchor atom {j} should not be in moved atoms")

    if k not in moved:
        raise AssertionError(f"axis moved atom {k} should be in moved atoms")

    if l not in moved:
        raise AssertionError(f"terminal moved atom {l} should be in moved atoms")


def _choose_dihedral_anchor_atom(
    *,
    adj: dict[int, set[int]],
    center: int,
    axis_other: int,
    moved: set[int],
) -> int | None:
    candidates = [n for n in adj.get(center, ()) if n != axis_other and n not in moved]

    if not candidates:
        return None

    return max(candidates, key=lambda n: (len(adj[n]), -n))


def _choose_dihedral_moved_atom(
    *,
    adj: dict[int, set[int]],
    center: int,
    axis_other: int,
    moved: set[int],
) -> int | None:
    candidates = [n for n in adj.get(center, ()) if n != axis_other and n in moved]

    if not candidates:
        return None

    return max(candidates, key=lambda n: (len(adj[n]), -n))


def _block_branch_points(
    *,
    mol: Chem.Mol,
    block_id: int,
    block_atom_sets: list[set[int]],
) -> set[int]:
    atoms = block_atom_sets[block_id]
    branch_points: set[int] = set()

    for other_id, other_atoms in enumerate(block_atom_sets):
        if other_id == block_id:
            continue

        overlap = atoms & other_atoms
        if overlap:
            # Anchor-inclusive relationship: the shared atom is the branch point.
            # Do not also mark neighbouring atoms as branch points.
            branch_points |= overlap
            continue

        # Non-overlap relationship, e.g. two ring cores connected by a glycosidic
        # bond.  In that case the atom inside this block that touches the other
        # block is the branch point for this block.
        for bond in mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()

            if a in atoms and b in other_atoms:
                branch_points.add(a)
            if b in atoms and a in other_atoms:
                branch_points.add(b)

    return branch_points

def _connection_atoms_in_child(
    *,
    mol: Chem.Mol,
    child_atoms: set[int],
    parent_atoms: set[int],
) -> set[int]:
    # Anchor-inclusive blocks usually overlap with their parent at the real
    # branch point.  If that overlap exists, do not also mark the neighbouring
    # internal atom as an anchor.  That keeps a path like ring-C-C-N-C(branch)
    # rooted at the ring atom, not at both the ring atom and the first chain atom.
    overlap = child_atoms & parent_atoms
    if overlap:
        return set(overlap)

    anchors: set[int] = set()
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        if a in child_atoms and b in parent_atoms:
            anchors.add(a)
        if b in child_atoms and a in parent_atoms:
            anchors.add(b)

    return anchors


def _is_rotatable_heavy_bond(
    *,
    mol: Chem.Mol,
    bond: Chem.Bond,
    adj: dict[int, set[int]],
    exclude_amides: bool,
    exclude_esters: bool,
    exclude_conjugated: bool,
) -> bool:
    a = bond.GetBeginAtomIdx()
    b = bond.GetEndAtomIdx()

    if a not in adj or b not in adj:
        return False

    if bond.GetBondType() != Chem.BondType.SINGLE:
        return False

    if bond.IsInRing():
        return False

    if len(adj[a]) <= 1 or len(adj[b]) <= 1:
        return False

    if exclude_conjugated and bond.GetIsConjugated():
        return False

    if exclude_amides and _is_amide_like_bond(mol, a, b):
        return False

    if exclude_esters and _is_ester_like_bond(mol, a, b):
        return False

    return True


def _is_carbonyl_carbon(mol: Chem.Mol, atom_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetAtomicNum() != 6:
        return False

    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if bond.GetBondType() == Chem.BondType.DOUBLE and other.GetAtomicNum() in {8, 16}:
            return True

    return False


def _is_amide_like_bond(mol: Chem.Mol, a: int, b: int) -> bool:
    za = mol.GetAtomWithIdx(a).GetAtomicNum()
    zb = mol.GetAtomWithIdx(b).GetAtomicNum()
    return (za == 7 and _is_carbonyl_carbon(mol, b)) or (zb == 7 and _is_carbonyl_carbon(mol, a))


def _is_ester_like_bond(mol: Chem.Mol, a: int, b: int) -> bool:
    za = mol.GetAtomWithIdx(a).GetAtomicNum()
    zb = mol.GetAtomWithIdx(b).GetAtomicNum()
    return (za in {8, 16} and _is_carbonyl_carbon(mol, b)) or (
        zb in {8, 16} and _is_carbonyl_carbon(mol, a)
    )


def _multi_source_distances(
    *,
    nodes: set[int],
    adj: dict[int, set[int]],
    roots: set[int],
) -> dict[int, int]:
    roots = set(roots) & nodes
    if not roots:
        return {}

    distances = {root: 0 for root in roots}
    queue = deque(sorted(roots))

    while queue:
        atom_idx = queue.popleft()

        for nbr in adj[atom_idx]:
            if nbr not in nodes or nbr in distances:
                continue

            distances[nbr] = distances[atom_idx] + 1
            queue.append(nbr)

    return distances


def _heavy_adjacency(mol: Chem.Mol) -> tuple[set[int], dict[int, set[int]]]:
    heavy_atoms = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1}
    adj = {i: set() for i in heavy_atoms}

    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        if a in heavy_atoms and b in heavy_atoms:
            adj[a].add(b)
            adj[b].add(a)

    return heavy_atoms, adj


def _ring_systems(mol: Chem.Mol) -> list[set[int]]:
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    return _merge_overlapping_sets(rings)


def _merge_overlapping_sets(items: list[set[int]]) -> list[set[int]]:
    merged: list[set[int]] = []

    for item in items:
        item = set(item)
        changed = True

        while changed:
            changed = False
            kept = []

            for existing in merged:
                if item & existing:
                    item |= existing
                    changed = True
                else:
                    kept.append(existing)

            merged = kept

        merged.append(item)

    return merged


def _terminal_hetero_substituents(
    core_atoms: set[int],
    adj: dict[int, set[int]],
    mol: Chem.Mol,
) -> set[int]:
    """
    Include terminal hetero substituents directly attached to a ring/core.
    """

    out = set()

    for core_atom in core_atoms:
        for nbr in adj[core_atom]:
            if nbr in core_atoms:
                continue

            atom = mol.GetAtomWithIdx(nbr)
            if len(adj[nbr]) == 1 and atom.GetAtomicNum() in {7, 8, 9, 15, 16, 17, 35, 53}:
                out.add(nbr)

    return out


def _contains_p_or_s_oxo_tail(mol: Chem.Mol, atoms: set[int]) -> bool:
    for atom_idx in atoms:
        atom = mol.GetAtomWithIdx(atom_idx)

        if atom.GetAtomicNum() not in {15, 16}:
            continue

        oxygen_neighbors = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 8)
        if oxygen_neighbors >= 2:
            return True

    return False


def _split_single_anchor_acyclic_component(
    *,
    mol: Chem.Mol,
    adj: dict[int, set[int]],
    component: set[int],
    anchor: int,
) -> list[dict]:
    starts = sorted(nbr for nbr in adj[anchor] if nbr in component)

    if len(starts) != 1:
        return [
            {
                "role": "branch_bundle",
                "atoms": set(component) | {anchor},
                "anchor_atoms": {anchor},
                "reason": "multiple equivalent starts from one anchor",
            }
        ]

    path_atoms, branch_specs = _trace_main_path_and_branches(
        mol=mol,
        adj=adj,
        component=component,
        anchor=anchor,
        start=starts[0],
    )

    out = [
        {
            "role": "path",
            "atoms": set(path_atoms),
            "anchor_atoms": {anchor},
            "reason": "largest acyclic path away from core",
        }
    ]

    branch_specs.sort(key=lambda x: (-len(x["atoms"]), x["role"], min(x["atoms"])))
    out.extend(branch_specs)
    return out


def _trace_main_path_and_branches(
    *,
    mol: Chem.Mol,
    adj: dict[int, set[int]],
    component: set[int],
    anchor: int,
    start: int,
) -> tuple[set[int], list[dict]]:
    path = [anchor]
    branches: list[dict] = []
    parent = anchor
    current = start

    while True:
        path.append(current)
        forward = sorted(nbr for nbr in adj[current] if nbr != parent and nbr in component)

        if not forward:
            break

        if len(forward) == 1:
            parent, current = current, forward[0]
            continue

        downstream = []
        for child in forward:
            atoms = _component_after_cut(adj=adj, start=child, cut=(current, child), allowed=component)
            downstream.append({"child": child, "atoms": atoms, "score": _branch_score(mol, atoms)})

        downstream.sort(key=lambda x: (x["score"], -x["child"]), reverse=True)
        best = downstream[0]
        second = downstream[1]

        if best["score"] > second["score"]:
            for item in downstream[1:]:
                branches.append(
                    {
                        "role": "branch",
                        "atoms": set(item["atoms"]) | {current},
                        "anchor_atoms": {current},
                        "reason": f"smaller side branch from atom {current}",
                    }
                )

            parent, current = current, best["child"]
            continue

        bundled = set()
        for item in downstream:
            bundled |= set(item["atoms"])

        branches.append(
            {
                "role": "branch_bundle",
                "atoms": bundled | {current},
                "anchor_atoms": {current},
                "reason": f"tied downstream branches from atom {current}",
            }
        )
        break

    return set(path), branches


def _branch_score(mol: Chem.Mol, atoms: set[int]) -> tuple[int, int, int]:
    n_atoms = len(atoms)
    n_carbons = sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in atoms)
    n_heavy_hetero = sum(mol.GetAtomWithIdx(i).GetAtomicNum() not in {1, 6} for i in atoms)
    return n_atoms, n_carbons, n_heavy_hetero


def _component_after_cut(
    *,
    adj: dict[int, set[int]],
    start: int,
    cut: tuple[int, int],
    allowed: set[int],
) -> set[int]:
    cut_a, cut_b = cut
    seen = {start}
    queue = deque([start])

    while queue:
        atom_idx = queue.popleft()

        for nbr in adj[atom_idx]:
            if nbr not in allowed:
                continue

            if (atom_idx == cut_a and nbr == cut_b) or (atom_idx == cut_b and nbr == cut_a):
                continue

            if nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)

    return seen


def _connected_components(nodes: set[int], adj: dict[int, set[int]]) -> list[set[int]]:
    remaining = set(nodes)
    components = []

    while remaining:
        start = remaining.pop()
        comp = {start}
        queue = deque([start])

        while queue:
            atom_idx = queue.popleft()

            for nbr in adj[atom_idx]:
                if nbr in remaining:
                    remaining.remove(nbr)
                    comp.add(nbr)
                    queue.append(nbr)

        components.append(comp)

    components.sort(key=lambda c: (-len(c), min(c)))
    return components


def _build_semantic_block_graph(
    mol: Chem.Mol,
    block_atom_sets: list[set[int]],
) -> dict[int, dict[int, int]]:
    """
    Weighted semantic block graph.

    edge weight 0 = blocks overlap at an anchor/branch point
    edge weight 1 = blocks are only connected by a real bond

    This matters because anchor-inclusive child blocks can also be one real bond
    away from the root core.  The overlap route is the semantic parent route.
    """

    graph: dict[int, dict[int, int]] = defaultdict(dict)
    atom_to_blocks: dict[int, list[int]] = defaultdict(list)

    for block_id, atoms in enumerate(block_atom_sets):
        for atom_idx in atoms:
            atom_to_blocks[atom_idx].append(block_id)

    def add_edge(i: int, j: int, weight: int) -> None:
        if i == j:
            return
        graph[i][j] = min(graph[i].get(j, weight), weight)
        graph[j][i] = min(graph[j].get(i, weight), weight)

    # Overlap edges, caused by anchor-inclusive blocks.
    for overlap_blocks in atom_to_blocks.values():
        for i in overlap_blocks:
            for j in overlap_blocks:
                add_edge(i, j, 0)

    # Real bond edges, e.g. purine core directly bonded to ribose core.
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        for ba in atom_to_blocks.get(a, []):
            for bb in atom_to_blocks.get(b, []):
                add_edge(ba, bb, 1)

    for i in range(len(block_atom_sets)):
        graph.setdefault(i, {})

    return graph

def _choose_semantic_root_block(*, specs: list[dict], root_atom_idx: int | None) -> int:
    if root_atom_idx is not None:
        containing = [i for i, spec in enumerate(specs) if root_atom_idx in spec["atoms"]]
        if containing:
            return max(containing, key=lambda i: (specs[i]["role"] == "core", len(specs[i]["atoms"])))

    core_blocks = [i for i, spec in enumerate(specs) if spec["role"] == "core"]
    if core_blocks:
        return max(core_blocks, key=lambda i: (len(specs[i]["atoms"]), -min(specs[i]["atoms"])))

    return 0


def _orient_semantic_blocks(
    block_adj: dict[int, dict[int, int]],
    root_block: int,
) -> tuple[dict[int, int | None], dict[int, list[int]]]:
    # Dijkstra over 0/1 semantic edges.  Overlap edges are preferred over direct
    # bond shortcuts, which keeps side branches parented to the path block they
    # share an anchor with.
    best_dist: dict[int, tuple[int, int]] = {root_block: (0, 0)}
    parent: dict[int, int | None] = {root_block: None}
    heap: list[tuple[int, int, int]] = [(0, 0, root_block)]
    step = 0

    while heap:
        dist, _, block_id = heapq.heappop(heap)
        if dist != best_dist[block_id][0]:
            continue

        for nbr, weight in sorted(block_adj[block_id].items()):
            if nbr == root_block:
                continue

            new_dist = dist + weight
            old = best_dist.get(nbr)
            old_parent = parent.get(nbr)
            old_parent_sort = old_parent if old_parent is not None else 10**9
            should_update = (
                old is None
                or new_dist < old[0]
                or (new_dist == old[0] and block_id < old_parent_sort)
            )

            if not should_update:
                continue

            parent[nbr] = block_id
            step += 1
            best_dist[nbr] = (new_dist, step)
            heapq.heappush(heap, (new_dist, step, nbr))

    children: dict[int, list[int]] = defaultdict(list)
    for child, par in parent.items():
        if par is not None:
            children[par].append(child)

    return parent, children

def _graph_center(nodes: set[int], adj: dict[int, set[int]]) -> int:
    best = min(nodes)
    best_score = (10**9, 10**9, best)

    for start in nodes:
        distances = {start: 0}
        queue = deque([start])

        while queue:
            atom_idx = queue.popleft()
            for nbr in adj[atom_idx]:
                if nbr in nodes and nbr not in distances:
                    distances[nbr] = distances[atom_idx] + 1
                    queue.append(nbr)

        if len(distances) != len(nodes):
            continue

        score = (max(distances.values()), sum(distances.values()), start)
        if score < best_score:
            best = start
            best_score = score

    return best


if __name__ == "__main__":
    examples = {
        "catechol_sidechain": "CC(C)NCC(c1ccc(c(c1)O)O)O",
        "atp": "c1nc(c2c(n1)n(cn2)C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N",
    }

    for name, smiles in examples.items():
        print("\n" + "=" * 80)
        print(name)
        mol = Chem.MolFromSmiles(smiles)
        blocks = build_semantic_anchor_blocks(mol)
        print_semantic_blocks(mol, blocks)
        draw_semantic_blocks_grid(mol, blocks, filename=f"{name}_semantic_blocks.png", mols_per_row=2)
