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
from ChemEM.parsers.models import CovalentLinkSpec, LigandList
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
        if os.sep in expanded or (os.altsep and os.altsep in expanded):
            return True
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
        """Ensure per-ligand covalent lists are the same length as `ligand`,
        padding with None for ligands declared without a covalent block."""
        target = len(self.ligand)
        for attr in self.COVALENT_FIELDS:
            lst = getattr(self, attr)
            while len(lst) < target:
                lst.append(None)

    def _set_covalent_for_last_ligand(self, attr_id: str, value: Any) -> None:
        """Attach a covalent_* value to the most recently declared ligand."""
        if not self.ligand:
            raise RuntimeError(
                f"[Error] '{attr_id}' must appear after a 'ligand = ...' entry."
            )
        self._pad_covalent_lists()
        idx = len(self.ligand) - 1
        lst = getattr(self, attr_id)
        lst[idx] = value
        self._provided.add(attr_id)

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

    def load_config(self, config_file: str) -> System:
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
        self._pad_covalent_lists()

        ligand_objects = []
        for i, lig_input in enumerate(self.ligand):
            covalent_spec = self._build_covalent_spec(i)
            lig_obj = self.add_ligand(lig_input, name=f"LIG{i}", covalent_spec=covalent_spec)
            # add_ligand returns a LigandList (one or more ligands from one input).
            # Attach the covalent spec only when exactly one ligand came from the
            # input — multi-variant covalent inputs are not supported.
            if covalent_spec is not None:
                if len(lig_obj) != 1:
                    raise RuntimeError(
                        f"[Error] covalent ligand entry {i} produced {len(lig_obj)} "
                        "variants; covalent ligands must yield exactly one molecule."
                    )
                lig_obj[0].covalent_link = covalent_spec
            ligand_objects += lig_obj
        system.ligand = LigandList(ligand_objects)

        # ---- Covalent ligands: apply protein deletions + build junction fragments.
        # Must happen after the Protein and all Ligands are built (so we can
        # parameterize the capped junction fragment) and before any
        # ChemEMSimulationSetup is constructed.
        for lig in system.ligand:
            spec = getattr(lig, "covalent_link", None)
            if spec is None:
                continue
            apply_protein_deletions(system.protein.complex_structure, spec)
            build_and_parameterize_fragment(lig.mol, spec)

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
        return ProteinParser.load_protein_structure(protein_file, forcefield=forcefield)

    def _build_covalent_spec(self, i: int) -> Optional[CovalentLinkSpec]:
        """Construct a CovalentLinkSpec from the i-th covalent_* slot, or None."""
        if i >= len(self.covalent_ligand_atom):
            return None
        lig_atom = self.covalent_ligand_atom[i]
        prot_atom = self.covalent_protein_atom[i]
        if lig_atom is None and prot_atom is None:
            return None
        if lig_atom is None or prot_atom is None:
            raise RuntimeError(
                f"[Error] Ligand {i}: both covalent_ligand_atom and "
                "covalent_protein_atom must be specified."
            )
        bond_order = self.covalent_bond_order[i] if i < len(self.covalent_bond_order) else None
        del_lig = self.covalent_delete_ligand_atoms[i] if i < len(self.covalent_delete_ligand_atoms) else None
        del_prot = self.covalent_delete_protein_atoms[i] if i < len(self.covalent_delete_protein_atoms) else None

        def _as_list(v):
            if v is None:
                return []
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
            return [str(v)]

        return CovalentLinkSpec(
            ligand_atom_spec=str(lig_atom),
            protein_atom_spec=str(prot_atom),
            bond_order=str(bond_order) if bond_order else "SINGLE",
            delete_ligand_atoms=_as_list(del_lig),
            delete_protein_atoms=_as_list(del_prot),
        )

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
