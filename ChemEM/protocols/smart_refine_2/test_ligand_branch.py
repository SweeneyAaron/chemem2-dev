from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Literal

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor


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
class SemanticBlock:
    block_id: int
    role: BlockRole
    atom_indices: tuple[int, ...]
    anchor_atoms: tuple[int, ...]
    parent: int | None
    children: tuple[int, ...]
    reason: str = ""


def build_semantic_anchor_blocks(
    mol: Chem.Mol,
    *,
    root_atom_idx: int | None = None,
    keep_p_s_tails_intact: bool = True,
) -> tuple[SemanticBlock, ...]:
    """
    Build general anchor-inclusive ligand blocks.

    This is for ligand decomposition/debugging, not for applying torsion moves.
    Blocks may share anchor atoms.

    Rules:
      1. Find all ring systems.
      2. Make each ring system a core block.
      3. Add terminal hetero substituents directly attached to each ring core.
      4. Decompose remaining acyclic components into paths, branches, tails, or linkers.
      5. Orient the block graph from the chosen root block.
    """

    if mol is None:
        raise ValueError("mol is None")

    heavy_atoms, adj = _heavy_adjacency(mol)

    if not heavy_atoms:
        return ()

    ring_systems = _ring_systems(mol)

    specs: list[dict] = []
    assigned: set[int] = set()

    # ---- 1. Ring/core blocks first ----
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

    # ---- 3. Build block graph from overlaps and real bonds ----
    block_atom_sets = [set(s["atoms"]) for s in specs]
    block_adj = _build_semantic_block_graph(mol, block_atom_sets)

    root_block = _choose_semantic_root_block(
        specs=specs,
        root_atom_idx=root_atom_idx,
    )

    parent, children = _orient_semantic_blocks(block_adj, root_block)

    blocks = []
    for block_id, spec in enumerate(specs):
        blocks.append(
            SemanticBlock(
                block_id=block_id,
                role=spec["role"],
                atom_indices=tuple(sorted(spec["atoms"])),
                anchor_atoms=tuple(sorted(spec["anchor_atoms"])),
                parent=parent.get(block_id),
                children=tuple(sorted(children.get(block_id, ()))),
                reason=spec["reason"],
            )
        )

    return tuple(blocks)


def print_semantic_blocks(mol: Chem.Mol, blocks: tuple[SemanticBlock, ...]) -> None:
    for block in blocks:
        labels = [
            f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}"
            for i in block.atom_indices
        ]

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
    blocks can share atoms.
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
        legends.append(
            f"B{block.block_id} {block.role}: {block.atom_indices}"
        )
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

def _heavy_adjacency(mol: Chem.Mol) -> tuple[set[int], dict[int, set[int]]]:
    heavy_atoms = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    }

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

    Examples:
      - phenolic OH on aromatic ring
      - exocyclic amino group on adenine
      - ribose OH directly attached to ribose ring carbon
    """

    out = set()

    for core_atom in core_atoms:
        for nbr in adj[core_atom]:
            if nbr in core_atoms:
                continue

            atom = mol.GetAtomWithIdx(nbr)

            if len(adj[nbr]) == 1 and atom.GetAtomicNum() in {
                7, 8, 9, 15, 16, 17, 35, 53
            }:
                out.add(nbr)

    return out


def _contains_p_or_s_oxo_tail(mol: Chem.Mol, atoms: set[int]) -> bool:
    """
    Keep phosphate/sulfate-like tails intact.

    This is why ATP gives one phosphate-tail block rather than lots of tiny
    P-O and terminal oxygen blocks.
    """

    for atom_idx in atoms:
        atom = mol.GetAtomWithIdx(atom_idx)

        if atom.GetAtomicNum() not in {15, 16}:
            continue

        oxygen_neighbors = 0

        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() == 8:
                oxygen_neighbors += 1

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

    branch_specs.sort(
        key=lambda x: (-len(x["atoms"]), x["role"], min(x["atoms"]))
    )

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

        forward = sorted(
            nbr
            for nbr in adj[current]
            if nbr != parent and nbr in component
        )

        if not forward:
            break

        if len(forward) == 1:
            parent, current = current, forward[0]
            continue

        downstream = []
        for child in forward:
            atoms = _component_after_cut(
                adj=adj,
                start=child,
                cut=(current, child),
                allowed=component,
            )

            downstream.append(
                {
                    "child": child,
                    "atoms": atoms,
                    "score": _branch_score(mol, atoms),
                }
            )

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
    n_carbons = sum(
        mol.GetAtomWithIdx(i).GetAtomicNum() == 6
        for i in atoms
    )
    n_heavy_hetero = sum(
        mol.GetAtomWithIdx(i).GetAtomicNum() not in {1, 6}
        for i in atoms
    )

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

            if (atom_idx == cut_a and nbr == cut_b) or (
                atom_idx == cut_b and nbr == cut_a
            ):
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
) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = defaultdict(set)

    atom_to_blocks: dict[int, list[int]] = defaultdict(list)
    for block_id, atoms in enumerate(block_atom_sets):
        for atom_idx in atoms:
            atom_to_blocks[atom_idx].append(block_id)

    # Overlap edges, caused by anchor-inclusive blocks.
    for blocks in atom_to_blocks.values():
        for i in blocks:
            for j in blocks:
                if i != j:
                    graph[i].add(j)

    # Real bond edges, e.g. purine core directly bonded to ribose core.
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        for ba in atom_to_blocks.get(a, []):
            for bb in atom_to_blocks.get(b, []):
                if ba != bb:
                    graph[ba].add(bb)
                    graph[bb].add(ba)

    for i in range(len(block_atom_sets)):
        graph.setdefault(i, set())

    return graph


def _choose_semantic_root_block(
    *,
    specs: list[dict],
    root_atom_idx: int | None,
) -> int:
    if root_atom_idx is not None:
        containing = [
            i for i, spec in enumerate(specs)
            if root_atom_idx in spec["atoms"]
        ]

        if containing:
            return max(
                containing,
                key=lambda i: (
                    specs[i]["role"] == "core",
                    len(specs[i]["atoms"]),
                ),
            )

    core_blocks = [
        i for i, spec in enumerate(specs)
        if spec["role"] == "core"
    ]

    if core_blocks:
        return max(
            core_blocks,
            key=lambda i: (len(specs[i]["atoms"]), -min(specs[i]["atoms"])),
        )

    return 0


def _orient_semantic_blocks(
    block_adj: dict[int, set[int]],
    root_block: int,
) -> tuple[dict[int, int | None], dict[int, list[int]]]:
    parent: dict[int, int | None] = {root_block: None}
    children: dict[int, list[int]] = defaultdict(list)
    queue = deque([root_block])

    while queue:
        block_id = queue.popleft()

        for nbr in sorted(block_adj[block_id]):
            if nbr in parent:
                continue

            parent[nbr] = block_id
            children[block_id].append(nbr)
            queue.append(nbr)

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

from rdkit import Chem

#atp_smiles = (
#    "c1nc(c2c(n1)n(cn2)C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N"
#)
atp_smiles = ("CC(C)NCC(c1ccc(c(c1)O)O)O")

mol = Chem.MolFromSmiles(atp_smiles)

blocks = build_semantic_anchor_blocks(mol)

print_semantic_blocks(mol, blocks)

img = draw_semantic_blocks_grid(
    mol,
    blocks,
    filename="/Users/aaron.sweeney/Documents/chemem2_build/ChemEM2_feb26/chemem2-dev/atp_semantic_blocks.png",
    mols_per_row=2,
)

img