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
import multiprocessing
import os

from ChemEM.data.system import System
from ChemEM.parsers.protein_parser import ProteinParser
from ChemEM.parsers.ligand_parser import LigandParser
from ChemEM.parsers.EMMap import EMMap
from ChemEM.data.data import SYSTEM_ATTRS


@dataclass
class Config:
    # Track which keys were explicitly set in the config file / python API
    _provided: Set[str] = field(default_factory=set, init=False, repr=False)

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
    n_cpus: Optional[int] = None
    post_process_solution: List[str] = field(default_factory=list)
    hold_fragment: List[str] = field(default_factory=list)

    protonation: Optional[bool] = None
    chirality: Optional[bool] = None
    rings: Optional[bool] = None
    pH: Tuple[float, float] = (6.4, 8.4)
    pKa_prec: Optional[float] = None
    forcefield: List[str] = field(default_factory=list)

    # Map and contour parameters
    map_contour: List[float] = field(default_factory=list)
    local_contour: List[float] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)

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
    }

    # ---------- Generic setters / loaders ----------

    def reset(self) -> "Config":
        """Reset to dataclass defaults so the same Config instance can be reused safely."""
        default = type(self)()
        for f in fields(self):
            if f.name == "_provided":
                continue
            setattr(self, f.name, copy.deepcopy(getattr(default, f.name)))
        self._provided.clear()
        return self

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
        """Set n_cpus if not explicitly provided."""
        if self.n_cpus is None:
            self.n_cpus = max(1, multiprocessing.cpu_count() - 2)
            # Marking provided is optional; I usually leave it unmarked because it's implicit.
            # But if you want it propagated via SYSTEM_ATTRS, mark it:
            self._provided.add("n_cpus")

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

        # Keep original config-file semantics: repeated list fields append entries
        self._set_value(attr_id, parsed_value, append_list_fields=True, mark_provided=True)

    def load_config(self, config_file: str) -> System:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, "r") as f:
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

        ligand_objects = []
        for i, lig_input in enumerate(self.ligand):
            try:
                lig_obj = self.add_ligand(lig_input, name=f"LIG{i}")
            except Exception:
                print(f"ChemEM Warning: Ligand Parser failed for ligand {i}: {lig_input} (skipping)")
                continue
            ligand_objects += lig_obj
        system.ligand = ligand_objects

        # Map
        if self.densmap is not None and self.resolution is not None:
            system.density_map = EMMap.from_mrc(self.densmap, resolution=self.resolution)

        # Apply only attributes explicitly set in config/python API
        for attr in SYSTEM_ATTRS:
            if attr in self._provided:
                value = getattr(self, attr, None)
                if value is not None:
                    setattr(system, attr, value)

        # Platform selection
        try:
            from ChemEM.tools.util import resolve_platform_name
            system.platform = resolve_platform_name(self.platform)
        except Exception as e:
            raise RuntimeError(f"[Error] Failed to resolve OpenMM platform '{self.platform}': {e}") from e

        return system

    def add_protein(self, protein_file: str, forcefield: List[str]):
        return ProteinParser.load_protein_structure(protein_file, forcefield=forcefield)

    def add_ligand(self, ligand_input: str, name: str = "LIG"):
        return LigandParser.load_ligands(
            ligand_input,
            protonation=True if self.protonation is None else self.protonation,
            chirality=True if self.chirality is None else self.chirality,
            rings=True if self.rings is None else self.rings,
            pH=self.pH,
            pka_prec=1.0 if self.pKa_prec is None else self.pKa_prec,
            name=name,
        )

'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
import ast
import multiprocessing
import os

from ChemEM.data.system import System
from ChemEM.parsers.protein_parser import ProteinParser
from ChemEM.parsers.ligand_parser import LigandParser
from ChemEM.parsers.EMMap import EMMap
from ChemEM.data.data import SYSTEM_ATTRS


@dataclass
class Config:
    # Track which keys were explicitly set in the config file
    _provided: Set[str] = field(default_factory=set, init=False, repr=False)
    
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
    n_cpus: Optional[int] = None
    post_process_solution: List[str] = field(default_factory=list)
    hold_fragment: List[str] = field(default_factory=list)
    
    protonation: Optional[bool] = None
    chirality: Optional[bool] = None
    rings: Optional[bool] = None
    pH: Tuple[float, float] = (6.4, 8.4)
    pKa_prec: Optional[float] = None #TODO! default should be 1.0?
    forcefield: List[str] = field(default_factory=list)
    
    # Map and contour parameters
    map_contour: List[float] = field(default_factory=list)
    local_contour: List[float] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    
    #move to data
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
    }
    
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
   
        if attr_id in self.LIST_FIELDS:
            current_val = getattr(self, attr_id, None)
            if current_val is not None and isinstance(current_val, list):
                current_val.append(parsed_value)  # keep your desired list-of-lists behaviour
            else:
                setattr(self, attr_id, [parsed_value])
            self._provided.add(attr_id)
            return
        if hasattr(self, attr_id):
           setattr(self, attr_id, parsed_value)
           self._provided.add(attr_id)
           return
        
        raise RuntimeError(f"[Error] Unknown attribute '{attr_id}' found in config.")
    
    def load_config(self, config_file: str) -> System:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, "r") as f:
            for raw in f:
                line = raw.strip()
                if line and not line.startswith("#"):
                    self._process_line(line)

        # Determine CPUs if not set explicitly
        if self.n_cpus is None:
            cpus = round(multiprocessing.cpu_count() - 2)
        

        return self.create_system()
    
    def get_ligands_from_dir(self, path: str) -> List[str]:
        if not os.path.isdir(path):
            print(f"Warning: {path} is not a directory. No ligands loaded.")
            return []
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".sdf")]
    
    def create_system(self) -> System:
        system = System()

        # Protein is required for most workflows
        if not self.protein:
            raise ValueError("Config error: 'protein' must be set.")

        system.protein = self.add_protein(self.protein, self.forcefield)

        # Ligands
        if self.ligands_from_dir is not None:
            self.ligand.extend(self.get_ligands_from_dir(self.ligands_from_dir))

        ligand_objects = []
        for i, lig_path in enumerate(self.ligand):
            try:
                lig_obj = self.add_ligand(lig_path, name=f"LIG{i}")
            except Exception:
                print(f"ChemEM Warning: Ligand Parser failed for ligand {i}: {lig_path} (skipping)")
                continue
            ligand_objects += lig_obj
        system.ligand = ligand_objects

        # Map
        if self.densmap is not None and self.resolution is not None:
            system.density_map = EMMap.from_mrc(self.densmap, resolution=self.resolution)

        # Apply only attributes explicitly set in config file
        for attr in SYSTEM_ATTRS:
            if attr in self._provided:
                value = getattr(self, attr, None)
                if value is not None:
                    setattr(system, attr, value)

        # Platform selection (moved out, fixed semantics)
        # - "auto" chooses best (CUDA/OpenCL/CPU)
        # - "CPU" forces CPU only
        try:
            from ChemEM.tools.util import resolve_platform_name
            system.platform = resolve_platform_name(self.platform)
        except Exception as e:
            # TODO! if openmm is not avalible or you want HTS mode make a fast cpu only protocol
            
            raise RuntimeError(f"[Error] Failed to resolve OpenMM platform '{self.platform}': {e}") from e

        return system

    def add_protein(self, protein_file: str, forcefield: List[str]):
        return ProteinParser.load_protein_structure(protein_file, forcefield=forcefield)

    def add_ligand(self, ligand_input: str, name: str = "LIG"):
        return LigandParser.load_ligands(
            ligand_input,
            protonation=True if self.protonation is None else self.protonation,
            chirality=True if self.chirality is None else self.chirality,
            rings=True if self.rings is None else self.rings,
            pH=self.pH,
            pka_prec=1.0 if self.pKa_prec is None else self.pKa_prec,
            name=name,
        )
    
    '''

    