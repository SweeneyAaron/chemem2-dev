# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>


from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple
import ast
import copy
import os

from ChemEM.data.system import System
from ChemEM.parsers.protein_parser import ProteinParser
from ChemEM.parsers.ligand_parser import LigandParser
from ChemEM.parsers.EMMap import EMMap
from ChemEM.parsers.models import CovalentLinkSpec, CovalentFragment, LigandList
from ChemEM.parsers.covalent_fragment import (
    apply_protein_deletions,
    build_and_parameterize_fragment,
)
from ChemEM.data.data import SYSTEM_ATTRS
from ChemEM.tools.resources import apply_cpu_budget, default_cpu_budget


@dataclass
class Config:
    # Track which keys were explicitly set in the config file / python API
    _provided: Set[str] = field(default_factory=set, init=False, repr=False)
    _config_dir: Optional[str] = field(default=None, init=False, repr=False)

    # File paths / IO
    protein: Optional[str] = None
    ligand: List[str] = field(default_factory=list)
    system_ligand_file: List[str] = field(default_factory=list)
    densmap: Optional[str] = None
    resolution: Optional[float] = None
    centroid: List[float] = field(default_factory=list)
    output: Optional[str] = None
    ligands_from_dir: Optional[str] = None
    system_ligands_from_dir: Optional[str] = None
    difference_map: List[str] = field(default_factory=list)
    local_resolution: List[float] = field(default_factory=list)
    full_map_id: Optional[str] = None

    # Other parameters
    platform: str = "auto"  # "auto" = choose best, "CPU" = force CPU, etc.
    cutoff: Optional[float] = None
    flexible_side_chains: Optional[bool] = None
    solvent: Optional[bool] = None
    ncpu: Optional[int] = None
    n_cpu: Optional[int] = None
    n_cpus: Optional[int] = None
    post_process_solution: List[str] = field(default_factory=list)
    hold_fragment: List[str] = field(default_factory=list)

    # Protein preparation. These must be settable before create_system() builds
    # the protein, which is why they live on Config rather than on system.options
    # like most protocol knobs -- see ProteinParser / remodel.determinism.
    prep_platform: str = "CPU"
    prep_threads: int = 1
    prep_seed: int = 1234567
    deterministic_prep: bool = True
    prep_clash_relief_steps: Optional[int] = None
    prep_h_implicit: bool = True
    cache_protein: bool = True
    protein_cache_dir: Optional[str] = None
    refresh_protein_cache: bool = False

    protonation: Optional[bool] = None
    chirality: Optional[bool] = None
    rings: Optional[bool] = None
    pH:  float = 7.4
    pKa_prec: Optional[float] = None
    max_ligand_varients: int = 1
    forcefield: List[str] = field(default_factory=list)

    # Map and contour parameters
    map_contour: List[float] = field(default_factory=list)
    local_contour: List[float] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)

    # Per-ligand covalent link specs. Each entry in ligand = … may have a
    # corresponding covalent block. Lists are parallel to `ligand`; entries
    # may be None for non-covalent ligands. In the config file, a covalent
    # block is opened with `covalent_ligand_atom = …` and its fields apply
    # to the most recently declared `ligand = …`.
    covalent_ligand_atom: List[Optional[str]] = field(default_factory=list)
    covalent_protein_atom: List[Optional[str]] = field(default_factory=list)
    covalent_bond_order: List[Optional[str]] = field(default_factory=list)
    covalent_delete_ligand_atoms: List[Optional[List[str]]] = field(default_factory=list)
    covalent_delete_protein_atoms: List[Optional[List[str]]] = field(default_factory=list)

    LIST_FIELDS = {
        "ligand",
        "system_ligand_file",
        "centroid",
        "map_contour",
        "local_contour",
        "local_resolution",
        "exclude",
        "difference_map",
        "post_process_solution",
        "hold_fragment",
        "forcefield",
        "covalent_ligand_atom",
        "covalent_protein_atom",
        "covalent_bond_order",
        "covalent_delete_ligand_atoms",
        "covalent_delete_protein_atoms",
    }

    COVALENT_FIELDS = (
        "covalent_ligand_atom",
        "covalent_protein_atom",
        "covalent_bond_order",
        "covalent_delete_ligand_atoms",
        "covalent_delete_protein_atoms",
    )

    SCALAR_PATH_FIELDS = {
        "protein",
        "densmap",
        "output",
        "ligands_from_dir",
        "system_ligands_from_dir",
    }

    LIST_PATH_FIELDS = {
        "system_ligand_file",
        "difference_map",
    }

    LIGAND_PATH_SUFFIXES = {
        ".sdf",
        ".mol",
        ".mol2",
        ".pdb",
        ".pdbqt",
        ".mae",
        ".smi",
    }

    CPU_FIELDS = {"ncpu", "n_cpu", "n_cpus"}

    # ---------- Generic setters / loaders ----------

    def reset(self) -> "Config":
        """Reset to dataclass defaults so the same Config instance can be reused safely."""
        default = type(self)()
        for f in fields(self):
            if f.name == "_provided":
                continue
            setattr(self, f.name, copy.deepcopy(getattr(default, f.name)))
        self._provided.clear()
        self._config_dir = None
        return self

    def _resolve_config_path(self, value: str) -> str:
        expanded = os.path.expanduser(value)
        if os.path.isabs(expanded) or self._config_dir is None:
            return os.path.abspath(expanded)
        return os.path.abspath(os.path.join(self._config_dir, expanded))

    def _looks_like_ligand_path(self, value: str) -> bool:
        expanded = os.path.expanduser(value.strip())
        if not expanded:
            return False
        if os.path.isabs(expanded):
            return True
        if expanded.startswith(".") or expanded.startswith("~"):
            return True
        # NOTE: do NOT treat a bare os.sep as path-like. Isomeric SMILES use '/'
        # and '\\' as E/Z double-bond markers (e.g. retinal 'CC1=C(/C=C/...C=O)'),
        # which must stay SMILES, not be resolved against the config dir. Real
        # relative paths are still caught below by their ligand suffix or by
        # existing on disk.
        if os.path.splitext(expanded)[1].lower() in self.LIGAND_PATH_SUFFIXES:
            return True
        if self._config_dir is not None and os.path.exists(os.path.join(self._config_dir, expanded)):
            return True
        return False

    def _normalize_config_value(self, attr_id: str, value: Any) -> Any:
        if self._config_dir is None or not isinstance(value, str):
            return value

        if attr_id in self.SCALAR_PATH_FIELDS or attr_id in self.LIST_PATH_FIELDS:
            return self._resolve_config_path(value)

        if attr_id == "ligand" and self._looks_like_ligand_path(value):
            return self._resolve_config_path(value)

        return value

    def _set_value(
        self,
        attr_id: str,
        value: Any,
        *,
        append_list_fields: bool = False,
        mark_provided: bool = True,
    ) -> None:
        """
        Internal unified setter.
        - append_list_fields=True reproduces config-file repeated-line behaviour
        - append_list_fields=False replaces list fields (better for Python API)
        """
        if attr_id in self.CPU_FIELDS:
            budget = apply_cpu_budget(self, value)
            if mark_provided:
                self._provided.update(self.CPU_FIELDS)
            return

        if not hasattr(self, attr_id):
            raise RuntimeError(f"[Error] Unknown attribute '{attr_id}'.")

        if attr_id in self.LIST_FIELDS:
            if append_list_fields:
                current_val = getattr(self, attr_id, None)
                if current_val is not None and isinstance(current_val, list):
                    current_val.append(value)
                else:
                    setattr(self, attr_id, [value])
            else:
                # Python API mode: replace list field directly.
                if value is None:
                    setattr(self, attr_id, [])
                elif isinstance(value, list):
                    setattr(self, attr_id, value)
                elif isinstance(value, tuple):
                    setattr(self, attr_id, list(value))
                else:
                    # allow scalar convenience: ligand="a.sdf" -> ["a.sdf"]
                    setattr(self, attr_id, [value])
        else:
            setattr(self, attr_id, value)

        if mark_provided:
            self._provided.add(attr_id)

    def _ensure_default_ncpus(self) -> None:
        """Set ncpu and legacy aliases if not explicitly provided."""
        if self.ncpu is None and self.n_cpu is None and self.n_cpus is None:
            apply_cpu_budget(self, default_cpu_budget())
        else:
            apply_cpu_budget(
                self,
                self.ncpu if self.ncpu is not None
                else self.n_cpu if self.n_cpu is not None
                else self.n_cpus,
            )
        self._provided.update(self.CPU_FIELDS)

    def apply_inputs(
        self,
        inputs: Mapping[str, Any],
        *,
        reset: bool = False,
        append_list_fields: bool = False,
    ) -> "Config":
        """
        Populate config from a Python dict-like object.

        Example:
            cfg.apply_inputs({
                "protein": "rec.pdb",
                "ligand": ["a.sdf", "b.sdf"],
                "densmap": "map.mrc",
                "resolution": 3.2,
                "platform": "CPU",
            })
        """
        if reset:
            self.reset()

        for k, v in inputs.items():
            self._set_value(k, v, append_list_fields=append_list_fields, mark_provided=True)

        self._ensure_default_ncpus()
        return self

    def load_inputs(self, **kwargs) -> System:
        """
        Python API entrypoint (kwargs instead of config file).
        """
        self.apply_inputs(kwargs, reset=False, append_list_fields=False)
        return self.create_system()

    @classmethod
    def from_inputs(cls, **kwargs) -> System:
        """
        Convenience classmethod:
            system = Config.from_python_inputs(...)
        """
        cfg = cls()
        cfg.apply_inputs(kwargs, reset=False, append_list_fields=False)
        return cfg.create_system()

    # ---------- Config file parsing ----------

    def _pad_covalent_lists(self) -> None:
        """Ensure per-ligand covalent lists are the same length as `ligand`.

        Each slot holds a LIST OF BLOCKS (one per covalent bond on that ligand),
        so slots for ligands with no covalent block are padded with []."""
        target = len(self.ligand)
        for attr in self.COVALENT_FIELDS:
            lst = getattr(self, attr)
            while len(lst) < target:
                lst.append([])

    def _set_covalent_for_last_ligand(self, attr_id: str, value: Any) -> None:
        """Attach a covalent_* value to the most recently declared ligand.

        A `covalent_ligand_atom` line OPENS a new bond block; the other covalent_*
        fields write into the block most recently opened for that ligand. Repeating
        `covalent_ligand_atom` therefore declares a second bond on the same ligand
        (a crosslinker), and a conf with one block per ligand parses exactly as it
        did before blocks existed.
        """
        if not self.ligand:
            raise RuntimeError(
                f"[Error] '{attr_id}' must appear after a 'ligand = ...' entry."
            )
        self._pad_covalent_lists()
        idx = len(self.ligand) - 1

        if attr_id == "covalent_ligand_atom":
            # Opens a new block, and extends every parallel list to match so the
            # blocks stay index-aligned across the five fields.
            for attr in self.COVALENT_FIELDS:
                slot = getattr(self, attr)[idx]
                slot.append(value if attr == attr_id else None)
            self._provided.add(attr_id)
            return

        slot = getattr(self, attr_id)[idx]
        if not slot:
            raise RuntimeError(
                f"[Error] '{attr_id}' must follow a 'covalent_ligand_atom = ...' line "
                "for the same ligand."
            )
        slot[-1] = value
        self._provided.add(attr_id)

    def _covalent_block_count(self, i: int) -> int:
        """Number of covalent bonds declared on ligand i."""
        if i >= len(self.covalent_ligand_atom):
            return 0
        return len(self.covalent_ligand_atom[i] or [])

    def _normalise_covalent_slots(self) -> None:
        """Coerce Python-API covalent values into the per-ligand block form.

        The config-file parser already produces blocks. Through `load_inputs`, a
        slot may instead be a bare scalar (one bond, the historical form) or a list
        of scalars (several bonds). `covalent_delete_*` is itself list-valued per
        bond, so a slot there is treated as a single bond's atom list unless it is
        a list of lists.
        """
        n = len(self.ligand)
        per_bond_lists = {"covalent_delete_ligand_atoms", "covalent_delete_protein_atoms"}

        for attr in self.COVALENT_FIELDS:
            slots = list(getattr(self, attr) or [])
            while len(slots) < n:
                slots.append([])
            out = []
            for i, slot in enumerate(slots):
                blocks = self._covalent_block_count(i) if attr != "covalent_ligand_atom" else None
                if slot is None:
                    out.append([])
                elif attr in per_bond_lists:
                    if isinstance(slot, (list, tuple)) and any(
                        isinstance(x, (list, tuple)) or x is None for x in slot
                    ):
                        out.append([list(x) if x is not None else None for x in slot])
                    elif isinstance(slot, (list, tuple)) and slot:
                        out.append([list(slot)])          # one bond, several atoms
                    else:
                        out.append([])
                elif isinstance(slot, (list, tuple)):
                    out.append(list(slot))
                else:
                    out.append([slot])                    # scalar -> one bond
                # Pad a short parallel list so block indices line up.
                if blocks:
                    while len(out[-1]) < blocks:
                        out[-1].append(None)
            setattr(self, attr, out)

    def _process_line(self, line: str) -> None:
        if "=" not in line:
            return

        attr_id, value = line.split("=", maxsplit=1)
        attr_id = attr_id.strip()
        value = value.strip()

        try:
            parsed_value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed_value = value

        parsed_value = self._normalize_config_value(attr_id, parsed_value)

        if attr_id in self.COVALENT_FIELDS:
            self._set_covalent_for_last_ligand(attr_id, parsed_value)
            return

        # Keep original config-file semantics: repeated list fields append entries
        self._set_value(attr_id, parsed_value, append_list_fields=True, mark_provided=True)

        # After a new ligand line, keep covalent lists in sync (filled with None
        # for this new slot until its covalent_* lines arrive).
        if attr_id == "ligand":
            self._pad_covalent_lists()

    def load_config(self, config_file: str, overrides=None) -> System:
        """Parse a config file and build the System.

        `overrides` carries the handful of CLI settings that must be known
        *before* the protein is built (prep platform/seed/cache), since
        create_system() builds it. Everything else reaches protocols later via
        system.options.
        """
        self.reset()

        config_path = os.path.abspath(os.path.expanduser(config_file))
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        self._config_dir = os.path.dirname(config_path)

        with open(config_path, "r") as f:
            for raw in f:
                line = raw.strip()
                if line and not line.startswith("#"):
                    self._process_line(line)

        self._ensure_default_ncpus()

        # CLI wins over the config file for these, and they are applied before
        # create_system() so protein prep can see them.
        for name, value in (overrides or {}).items():
            if value is not None and hasattr(self, name):
                setattr(self, name, value)
                self._provided.add(name)

        return self.create_system()

    # ---------- Helpers ----------

    def get_ligands_from_dir(self, path: str) -> List[str]:
        if not os.path.isdir(path):
            print(f"Warning: {path} is not a directory. No ligands loaded.")
            return []
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".sdf")]

    # ---------- System creation ----------

    def create_system(self) -> System:
        system = System()

        # Protein is required for most workflows
        if not self.protein:
            raise ValueError("Config error: 'protein' must be set.")

        system.protein = self.add_protein(self.protein, self.forcefield)

        # Ligands
        if self.ligands_from_dir is not None:
            self.ligand.extend(self.get_ligands_from_dir(self.ligands_from_dir))

        # Make sure covalent lists are the same length as ligand list
        self._normalise_covalent_slots()
        self._pad_covalent_lists()

        ligand_objects = []
        for i, lig_input in enumerate(self.ligand):
            covalent_specs = self._build_covalent_specs(i)
            # The ligand parser only needs one spec, to decide whether an addition
            # warhead needs saturating before OpenFF sees it.
            lig_obj = self.add_ligand(
                lig_input, name=f"LIG{i}",
                covalent_spec=covalent_specs[0] if covalent_specs else None,
            )
            # add_ligand returns a LigandList (one or more ligands from one input).
            # Attach the covalent specs only when exactly one ligand came from the
            # input — multi-variant covalent inputs are not supported.
            if covalent_specs:
                if len(lig_obj) != 1:
                    raise RuntimeError(
                        f"[Error] covalent ligand entry {i} produced {len(lig_obj)} "
                        "variants; covalent ligands must yield exactly one molecule."
                    )
                lig_obj[0].covalent_links = covalent_specs
            ligand_objects += lig_obj
        system.ligand = LigandList(ligand_objects)

        # ---- Covalent ligands: apply protein deletions + build junction fragments.
        # Must happen after the Protein and all Ligands are built (so we can
        # parameterize the capped junction fragment) and before any
        # ChemEMSimulationSetup is constructed.
        #
        # Protein deletions are per bond; the fragment is built ONCE per ligand
        # spanning every junction, so terms bridging two junctions are collected
        # once and the ligand charges are redistributed once.
        for lig in system.ligand:
            specs = getattr(lig, "covalent_links", None)
            if not specs:
                continue
            for spec in specs:
                apply_protein_deletions(system.protein.complex_structure, spec)
            terms = build_and_parameterize_fragment(lig.mol, specs)
            lig.covalent_fragment = CovalentFragment(
                fragment_structure=terms["fragment_structure"],
                junction_angles=terms["angles"],
                junction_dihedrals=terms["dihedrals"],
                fragment_ligand_charges=terms["ligand_charges"],
            )

        # Map
        if self.densmap is not None and self.resolution is not None:
            system.density_map = EMMap.from_mrc(self.densmap, resolution=self.resolution)

        # Apply only attributes explicitly set in config/python API
        for attr in SYSTEM_ATTRS:
            if attr in self._provided:
                value = getattr(self, attr, None)
                if value is not None:
                    setattr(system, attr, value)

        apply_cpu_budget(system, self.ncpu)

        # Platform selection
        try:
            from ChemEM.tools.util import resolve_platform_name
            system.platform = resolve_platform_name(self.platform)
        except Exception as e:
            raise RuntimeError(f"[Error] Failed to resolve OpenMM platform '{self.platform}': {e}") from e

        return system

    def add_protein(self, protein_file: str, forcefield: List[str]):
        from ChemEM.parsers.remodel.determinism import PrepOptions
        from ChemEM.parsers.remodel.prep_cache import ProteinPrepCache

        prep = PrepOptions(
            platform=self.prep_platform,
            threads=self.prep_threads,
            seed=self.prep_seed,
            pH=self.pH,
            deterministic=self.deterministic_prep,
            clash_relief_steps=self.prep_clash_relief_steps,
            h_placement_implicit=self.prep_h_implicit,
        )
        cache = None
        if self.cache_protein:
            cache = ProteinPrepCache(
                root=self.protein_cache_dir,
                refresh=self.refresh_protein_cache,
            )
        return ProteinParser.load_protein_structure(
            protein_file, forcefield=forcefield, prep=prep, cache=cache,
        )

    def _build_covalent_specs(self, i: int) -> List[CovalentLinkSpec]:
        """Construct the CovalentLinkSpec list for ligand i (empty if non-covalent).

        One spec per declared block, so a crosslinker with two `covalent_ligand_atom`
        lines yields two specs.
        """
        if i >= len(self.covalent_ligand_atom):
            return []

        def _block(attr, b):
            slot = getattr(self, attr)
            if i >= len(slot):
                return None
            blocks = slot[i] or []
            return blocks[b] if b < len(blocks) else None

        def _as_list(v):
            if v is None:
                return []
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
            return [str(v)]

        specs: List[CovalentLinkSpec] = []
        seen_ligand_atoms, seen_protein_atoms = set(), set()

        for b in range(self._covalent_block_count(i)):
            lig_atom = _block("covalent_ligand_atom", b)
            prot_atom = _block("covalent_protein_atom", b)
            if lig_atom is None and prot_atom is None:
                continue
            if lig_atom is None or prot_atom is None:
                raise RuntimeError(
                    f"[Error] Ligand {i} covalent bond {b}: both covalent_ligand_atom "
                    "and covalent_protein_atom must be specified."
                )
            # Two bonds sharing an endpoint cannot both be formed; catch it here
            # rather than letting the fragment builder produce a malformed molecule.
            if str(lig_atom) in seen_ligand_atoms:
                raise RuntimeError(
                    f"[Error] Ligand {i}: ligand atom '{lig_atom}' is used by more "
                    "than one covalent bond."
                )
            if str(prot_atom) in seen_protein_atoms:
                raise RuntimeError(
                    f"[Error] Ligand {i}: protein atom '{prot_atom}' is used by more "
                    "than one covalent bond."
                )
            seen_ligand_atoms.add(str(lig_atom))
            seen_protein_atoms.add(str(prot_atom))

            bond_order = _block("covalent_bond_order", b)
            specs.append(
                CovalentLinkSpec(
                    ligand_atom_spec=str(lig_atom),
                    protein_atom_spec=str(prot_atom),
                    bond_order=str(bond_order) if bond_order else "SINGLE",
                    delete_ligand_atoms=_as_list(_block("covalent_delete_ligand_atoms", b)),
                    delete_protein_atoms=_as_list(_block("covalent_delete_protein_atoms", b)),
                )
            )
        return specs

    def add_ligand(self, ligand_input: str, name: str = "LIG",
                   covalent_spec: Optional[CovalentLinkSpec] = None):
        return LigandParser.load_ligands(
            ligand_input,
            protonation=True if self.protonation is None else self.protonation,
            chirality=True if self.chirality is None else self.chirality,
            rings=True if self.rings is None else self.rings,
            pH=self.pH,
            pka_prec=0.0 if self.pKa_prec is None else self.pKa_prec,
            max_varients=self.max_ligand_varients,
            name=name,
            covalent_spec=covalent_spec,
        )
