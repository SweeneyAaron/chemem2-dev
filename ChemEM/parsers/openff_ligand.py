 
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.interchange import Interchange
import io
from openmm.app import PDBFile 
from rdkit import Chem 
import parmed

def load_ligand_structure(molecule,  allow_undefined_stereo= True):
     
     ligand_off_molecule = Molecule.from_rdkit(molecule, allow_undefined_stereo=True)
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
