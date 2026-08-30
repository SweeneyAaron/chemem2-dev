
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.interchange import Interchange
import io
import math
from itertools import combinations
from openmm.app import PDBFile
from openmm import unit as _ou
import openmm as _mm
from rdkit import Chem
from rdkit.Chem import AllChem
import parmed


def _openff_structure(molecule, allow_undefined_stereo=True):
    """Normal OpenFF/SMIRNOFF parameterization. Raises for elements/valences the
    force field cannot handle (e.g. boron)."""
    ligand_off_molecule = Molecule.from_rdkit(molecule, allow_undefined_stereo=allow_undefined_stereo)
    ligand_off_molecule.assign_partial_charges("mmff94")

    force_field = ForceField("openff_unconstrained-2.0.0.offxml")
    interchange = Interchange.from_smirnoff(
        topology=[ligand_off_molecule],
        force_field=force_field,
        charge_from_molecules=[ligand_off_molecule],
    )

    ligand_system = interchange.to_openmm()

    virtual_file = io.StringIO(Chem.MolToPDBBlock(molecule))
    ligand_pdbfile = PDBFile(virtual_file)
    ligand_structure = parmed.openmm.load_topology(
        ligand_pdbfile.topology, ligand_system, xyz=ligand_pdbfile.positions
    )
    return ligand_structure, ligand_system


# --- generic approximate-harmonic fallback -----------------------------------
# When OpenFF/SMIRNOFF cannot parameterize a ligand (boron, exotic elements or
# valences), build a geometry-restrained harmonic model instead of crashing:
# each bond/angle is pinned to the input geometry and nonbonded uses generic LJ
# + Gasteiger charges. This is intentionally approximate — the ligand is held
# near its input shape while the covalent anchor and density position the pose.

_HARMONIC_BOND_K = 300000.0   # kJ/mol/nm^2  (~stiff bond)
_HARMONIC_ANGLE_K = 500.0     # kJ/mol/rad^2
_LJ_EPS = 0.30                # kJ/mol       (modest, clash-avoidance)
# element -> van der Waals radius (Angstrom); default 1.75 for anything missing
_VDW_R = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "P": 1.80, "S": 1.80,
    "CL": 1.75, "BR": 1.85, "I": 1.98, "B": 1.92, "SI": 2.10, "SE": 1.90,
}


def _lj_sigma_nm(symbol: str) -> float:
    r = _VDW_R.get(symbol.upper(), 1.75)  # Angstrom vdW radius
    # sigma = R_min / 2^(1/6), R_min = 2*R_vdw ; convert Angstrom -> nm
    return (2.0 * r * 0.1) / (2.0 ** (1.0 / 6.0))


def _harmonic_fallback_structure(molecule, reason=""):
    mol = Chem.Mol(molecule)
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    conf = mol.GetConformer()
    pos_nm = [(conf.GetAtomPosition(i).x * 0.1,
               conf.GetAtomPosition(i).y * 0.1,
               conf.GetAtomPosition(i).z * 0.1) for i in range(mol.GetNumAtoms())]

    # topology + positions from the same PDB path the normal route uses (atom
    # order matches the RDKit mol, so bond/charge indices line up).
    pdb = PDBFile(io.StringIO(Chem.MolToPDBBlock(mol)))
    top = pdb.topology

    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    system = _mm.System()
    for atom in mol.GetAtoms():
        system.addParticle(atom.GetMass())

    nb = _mm.NonbondedForce()
    nb.setNonbondedMethod(_mm.NonbondedForce.NoCutoff)
    for atom in mol.GetAtoms():
        q = 0.0
        if atom.HasProp("_GasteigerCharge"):
            gq = atom.GetDoubleProp("_GasteigerCharge")
            q = gq if math.isfinite(gq) else 0.0
        nb.addParticle(q * _ou.elementary_charge,
                       _lj_sigma_nm(atom.GetSymbol()) * _ou.nanometer,
                       _LJ_EPS * _ou.kilojoule_per_mole)
    system.addForce(nb)

    def _d(i, j):
        a, b = pos_nm[i], pos_nm[j]
        return math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)))

    bond_pairs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    hb = _mm.HarmonicBondForce()
    for i, j in bond_pairs:
        hb.addBond(i, j, _d(i, j) * _ou.nanometer,
                   _HARMONIC_BOND_K * _ou.kilojoule_per_mole / _ou.nanometer ** 2)
    system.addForce(hb)

    adj = {i: [] for i in range(mol.GetNumAtoms())}
    for i, j in bond_pairs:
        adj[i].append(j)
        adj[j].append(i)
    ha = _mm.HarmonicAngleForce()
    for j, nbrs in adj.items():
        for i, k in combinations(nbrs, 2):
            v1 = [pos_nm[i][t] - pos_nm[j][t] for t in range(3)]
            v2 = [pos_nm[k][t] - pos_nm[j][t] for t in range(3)]
            n1 = math.sqrt(sum(x * x for x in v1)) or 1e-9
            n2 = math.sqrt(sum(x * x for x in v2)) or 1e-9
            cosang = max(-1.0, min(1.0, sum(v1[t] * v2[t] for t in range(3)) / (n1 * n2)))
            ha.addAngle(i, j, k, math.acos(cosang) * _ou.radian,
                        _HARMONIC_ANGLE_K * _ou.kilojoule_per_mole / _ou.radian ** 2)
    system.addForce(ha)

    nb.createExceptionsFromBonds(bond_pairs, 0.833, 0.5)

    positions = pdb.positions
    structure = parmed.openmm.load_topology(top, system, xyz=positions)
    print(f"[ligand] approximate harmonic parameters (OpenFF failed: {reason})")
    return structure, system


def load_ligand_structure(molecule, allow_undefined_stereo=True):
    """Parameterize a ligand. Falls back to a generic geometry-restrained harmonic
    model if OpenFF/SMIRNOFF cannot handle it (boron, exotic elements/valences)."""
    try:
        return _openff_structure(molecule, allow_undefined_stereo=allow_undefined_stereo)
    except Exception as exc:
        return _harmonic_fallback_structure(molecule, reason=f"{type(exc).__name__}: {exc}")
