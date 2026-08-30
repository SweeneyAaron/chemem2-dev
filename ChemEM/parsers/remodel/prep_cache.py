# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Cache for prepared (repaired + protonated) proteins.

Preparation is the one expensive step in loading a protein, and its result is a
pure function of the input file plus the prep settings -- once preparation is
deterministic (see determinism.py). So it can be computed once and reused.

Exactly one artefact is cached: the output of ``remodel_from_fixer``, i.e. the
prepared OpenMM Topology and positions. Everything downstream (force field,
System, ParmEd Structure, SSE groups, residue map) is rebuilt every run, because
it is cheap, deterministic given the coordinates, and serialising it would tie
the cache to ParmEd and OpenMM versions for no real saving.

**The residue map is never cached.** It has to be rebuilt against the *original*
input file, and it doubles as the integrity check on a cache hit.

Why not just cache a PDB
------------------------
Two independent reasons, both silent-corruption class:

  * ``build_residue_map_by_positions`` matches original<->prepared backbone atoms
    to **1e-5 A**. That works only because untouched atoms round-trip through PDB
    '%.3f' formatting exactly. Any format that re-quantises coordinates returns an
    EMPTY residue map with no error. Positions are therefore stored as float64
    nanometres -- OpenMM's native units, no conversion, no rounding.
  * ``split_chains_on_breaks`` gives split segments the *same* chain id and drops
    insertion codes, so chains are only separable by TER placement. A PDB reparse
    is a lossy guess at the topology; pdbfixer and addHydrogens also add bonds
    that ``createStandardBonds()`` cannot reconstruct. The topology is therefore
    stored explicitly.

A human-readable ``prepared.pdb`` is written alongside for inspection and is
never read back.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile

import numpy as np

CACHE_FORMAT_VERSION = 1

_COMPLETE_MARKER = ".complete"


def default_cache_root():
    """Cache root: $CHEMEM_CACHE_DIR, else $XDG_CACHE_HOME/chemem, else ~/.cache/chemem.

    Deliberately not the run's output directory -- the point is reuse across runs.
    """
    env = os.environ.get("CHEMEM_CACHE_DIR")
    if env:
        return os.path.join(env, "prepared_protein")
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = xdg if xdg else os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "chemem", "prepared_protein")


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class ProteinPrepCache:
    """Content-addressed store of prepared proteins."""

    def __init__(self, root=None, refresh=False):
        self.root = root or default_cache_root()
        self.refresh = bool(refresh)

    # ------------------------------------------------------------------- key

    def key(self, protein_file, *, forcefield_files, request_implicit, force_ff, prep):
        """Hash of everything that changes the prepared coordinates.

        The file *path* is deliberately excluded, so the same structure at two
        paths shares one entry. Anything not listed here that changes the prep
        pipeline must bump PREP_SCHEMA_VERSION in determinism.py.
        """
        from .determinism import PREP_SCHEMA_VERSION, PrepOptions

        if prep is None:
            prep = PrepOptions()

        payload = {
            "input_sha256": _sha256_file(protein_file),
            "input_bytes": os.path.getsize(protein_file),
            "forcefield_files": [str(f) for f in (forcefield_files or [])],
            "request_implicit": bool(request_implicit),
            "force_ff": bool(force_ff),
            "prep": prep.key_fields(),
            "versions": _versions(),
            "schema": PREP_SCHEMA_VERSION,
            "cache_format": CACHE_FORMAT_VERSION,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def _dir(self, key):
        return os.path.join(self.root, key)

    def _is_complete(self, key):
        return os.path.exists(os.path.join(self._dir(key), _COMPLETE_MARKER))

    # ------------------------------------------------------------------ read

    def expected_mapped(self, key):
        """Residue count the cold path mapped, for the integrity check."""
        try:
            with open(os.path.join(self._dir(key), "meta.json")) as fh:
                return json.load(fh).get("n_mapped_residues")
        except Exception:
            return None

    def load(self, key):
        """Return an app.Modeller, or None on any miss//corruption."""
        if key is None or self.refresh or not self._is_complete(key):
            return None
        try:
            path = self._dir(key)
            positions_nm = np.load(os.path.join(path, "positions.npy"))
            with open(os.path.join(path, "topology.json")) as fh:
                topology = _topology_from_json(json.load(fh))

            if topology.getNumAtoms() != positions_nm.shape[0]:
                raise ValueError("atom count disagrees with positions")

            from openmm import unit
            from openmm.app import Modeller
            print(f"[prep] reusing cached prepared protein ({key[:12]})")
            return Modeller(topology, positions_nm * unit.nanometer)
        except Exception as exc:
            print(f"[prep] cached protein unusable ({type(exc).__name__}: {exc}); re-preparing.")
            return None

    # ----------------------------------------------------------------- write

    def store(self, key, modeller, *, n_mapped=None):
        """Write atomically: build in a temp dir, rename into place, marker last.

        A reader that finds no marker treats the entry as a miss, so a crashed or
        racing writer can never be read as a valid entry.
        """
        if key is None:
            return
        try:
            from openmm import unit
            from openmm.app import PDBFile

            os.makedirs(self.root, exist_ok=True)
            tmp = tempfile.mkdtemp(prefix=f".tmp-{os.getpid()}-", dir=self.root)

            positions_nm = np.asarray(
                modeller.positions.value_in_unit(unit.nanometer), dtype=np.float64
            )
            np.save(os.path.join(tmp, "positions.npy"), positions_nm)
            with open(os.path.join(tmp, "topology.json"), "w") as fh:
                json.dump(_topology_to_json(modeller.topology), fh)

            meta = {
                "version": CACHE_FORMAT_VERSION,
                "key": key,
                "n_atoms": int(positions_nm.shape[0]),
                "n_residues": modeller.topology.getNumResidues(),
                "n_bonds": modeller.topology.getNumBonds(),
                "n_mapped_residues": n_mapped,
                "versions": _versions(),
            }
            with open(os.path.join(tmp, "meta.json"), "w") as fh:
                json.dump(meta, fh, indent=2)

            # Inspection only -- never read back. Do not be tempted to load from
            # this: '%.3f' rounding would break the 1e-5 A residue match.
            try:
                with open(os.path.join(tmp, "prepared.pdb"), "w") as fh:
                    PDBFile.writeFile(modeller.topology, modeller.positions, fh, keepIds=True)
            except Exception:
                pass

            open(os.path.join(tmp, _COMPLETE_MARKER), "w").close()

            dest = self._dir(key)
            if os.path.exists(dest):
                shutil.rmtree(dest, ignore_errors=True)
            try:
                os.replace(tmp, dest)
            except OSError:
                # Another process won the race; its entry is as good as ours.
                shutil.rmtree(tmp, ignore_errors=True)
                return
            print(f"[prep] cached prepared protein ({key[:12]})")
        except Exception as exc:
            print(f"[prep] could not cache prepared protein ({type(exc).__name__}: {exc}); continuing.")

    def invalidate(self, key):
        if key:
            shutil.rmtree(self._dir(key), ignore_errors=True)


def _versions():
    versions = {}
    try:
        import ChemEM
        versions["chemem"] = getattr(ChemEM, "__version__", "?")
    except Exception:
        versions["chemem"] = "?"
    try:
        import openmm
        versions["openmm"] = openmm.version.version
    except Exception:
        versions["openmm"] = "?"
    try:
        from importlib.metadata import version
        versions["pdbfixer"] = version("pdbfixer")
    except Exception:
        versions["pdbfixer"] = "?"
    return versions


# --------------------------------------------------------------------------- #
# Topology (de)serialisation. Explicit rather than via PDB because chain ids are
# not unique after split_chains_on_breaks, and prep adds bonds that
# createStandardBonds() cannot rebuild.
# --------------------------------------------------------------------------- #

def _topology_to_json(topology):
    chains, residues, atoms = [], [], []
    chain_index, residue_index = {}, {}

    for chain in topology.chains():
        chain_index[chain] = len(chains)
        chains.append({"id": chain.id})

    for residue in topology.residues():
        residue_index[residue] = len(residues)
        residues.append({
            "name": residue.name,
            "id": residue.id,
            "insertionCode": getattr(residue, "insertionCode", ""),
            "chain": chain_index[residue.chain],
        })

    atom_index = {}
    for atom in topology.atoms():
        atom_index[atom] = len(atoms)
        atoms.append({
            "name": atom.name,
            "element": atom.element.symbol if atom.element is not None else None,
            "residue": residue_index[atom.residue],
            "id": atom.id,
        })

    bonds = []
    for bond in topology.bonds():
        bonds.append([
            atom_index[bond[0]],
            atom_index[bond[1]],
            str(bond.type) if getattr(bond, "type", None) is not None else None,
            bond.order if getattr(bond, "order", None) is not None else None,
        ])

    box = topology.getPeriodicBoxVectors()
    if box is not None:
        from openmm import unit
        box = np.asarray(box.value_in_unit(unit.nanometer), dtype=float).tolist()

    return {"chains": chains, "residues": residues, "atoms": atoms,
            "bonds": bonds, "box": box}


def _topology_from_json(data):
    from openmm import unit
    from openmm.app import Topology, element as omm_element

    topology = Topology()

    chains = [topology.addChain(c["id"]) for c in data["chains"]]

    residues = []
    for r in data["residues"]:
        residues.append(topology.addResidue(
            r["name"], chains[r["chain"]], r["id"], r.get("insertionCode", "")
        ))

    atoms = []
    for a in data["atoms"]:
        element = omm_element.get_by_symbol(a["element"]) if a["element"] else None
        atoms.append(topology.addAtom(a["name"], element, residues[a["residue"]], a["id"]))

    for bond in data["bonds"]:
        i, j = bond[0], bond[1]
        order = bond[3] if len(bond) > 3 else None
        topology.addBond(atoms[i], atoms[j], order=order)

    if data.get("box") is not None:
        topology.setPeriodicBoxVectors(np.asarray(data["box"], dtype=float) * unit.nanometer)

    return topology
