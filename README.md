## Local mamba build + install (from this repo)

This repo is for the development of  `ChemEM2`.
These instructions build a local conda package from this source tree (including the compiled C++/pybind11 extensions), then install it into a fresh environment **with all runtime dependencies**.
> Installation requires mamba, this is a development version, please report bugs to aaron sweeney ucbtasw@ucl.ac.uk
> Assumes you are in the repo root (the folder that contains `meta.yaml` and `CMakeLists.txt`).

---

## 1) Build environment (conda/mamba)

Create a build env with the *host/build* requirements:

```bash
mamba create -n chemem-build -c conda-forge \
  python=3.11.11 \
  cmake ninja scikit-build-core pip \
  pybind11 eigen \
  rdkit=2024.03.3 rdkit-dev=2024.03.3 \
  "numpy>=1.19" \
  boost-cpp \
  llvm-openmp \
  conda-build conda-index mamba
```

Activate it:

```bash
mamba activate chemem-build
```

Notes:
- `llvm-openmp` is mainly for macOS. On Linux you can usually drop it.

---

## 2) Configure + build with CMake + Ninja

From the repo root:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
```

Build:

```bash
cmake --build build 
```

---

## 3) Runtime environment (conda/mamba)

Create a runtime env with the *run* requirements:

```bash
mamba create -n chemem -c conda-forge \
  python=3.11.11 \
  numpy=1.26.4 scipy openmm \
  rdkit=2024.03.3 \
  "dimorphite-dl>=2.0.1" \
  "spyrmsd>=0.8.0" \
  ambertools=23.6 \
  openff-toolkit=0.16.0 openff-toolkit-base=0.16.0 \
  openff-amber-ff-ports=0.0.4 \
  openff-forcefields=2024.09.0 \
  openff-interchange=0.3.29 openff-interchange-base=0.3.29 \
  openff-models=0.1.2 openff-units=0.2.2 openff-utilities=0.1.12 \
  "pdbfixer>=1.11" "mrcfile>=1.5.0" \
  scikit-image kneed mdtraj networkx tqdm scikit-learn grand \
  boost-cpp \
  llvm-openmp
```

Activate it:

```bash
mamba activate chemem
```


---

## 4) Recommended rebuild + local install

If you want the exact command sequence for a fresh local package build and install, use:

```bash
mamba activate chemem-build
conda config --set channel_priority strict
conda build . --output-folder conda-dist --no-anaconda-upload
conda index conda-dist

mamba create -n chemem-test -c conda-forge python=3.11.11
mamba install -n chemem-test --override-channels -c file://$PWD/conda-dist -c conda-forge chemem=2.0.0

conda run -n chemem-test python -c "import openmm; from ChemEM.config import Config; print('runtime imports ok')"
conda run -n chemem-test chemem --help
```

If that passes, activate the runtime env and run ChemEM normally:

```bash
mamba activate chemem-test
chemem <conf_file> <options>
```

---

## 5) Build the local conda package

Build the package from this repo:

```bash
conda build . --output-folder conda-dist --no-anaconda-upload
```


If you want `mamba` to install from your local build output, index it once:

```bash
conda index conda-dist
python -m conda_index conda-dist
```

Then install into your target environment from the local channel:

```bash
mamba install -c "file://${PWD}/conda-dist" -c conda-forge \
  --channel-priority flexible \
  "chemem==2.0.0=py311hcb8d3e5_8" --force-reinstall


```

You can also install the built artifact directly if you prefer:

```bash
mamba install -n chemem ./conda-dist/osx-arm64/chemem-2.0.0-*_8.tar.bz2
```

Adjust the subdirectory (`osx-arm64`, `osx-64`, `linux-64`, etc.) to match your platform.

---

## 6) Run ChemEM

After installation, use the console entry point:

```bash
chemem <conf_file> <options>
```

`python -m ChemEM` still works as well:

```bash
python -m ChemEM <conf_file> <options>
```

Paths inside the config file can now be either absolute paths or relative paths.
Relative paths are resolved relative to the config file's own directory, not the shell's
current working directory.

Example:

```bash
python -m ChemEM 7jjo_conf.txt --dock --no-map --minimize-docking --rescore
```

Quick smoke test for a fresh install:

```bash
python -c "import openmm; from ChemEM.config import Config; print('runtime imports ok')"
chemem --help
```

Note: for module execution, use a module name, not a path:

```bash
python -m ChemEM 7jjo_conf.txt --dock
```

---

## Why the macOS `zsh: killed` bug happened

In simple terms, ChemEM was not crashing because the Python code was wrong in one obvious
place. The real problem was that some compiled libraries inside the conda environment
ended up with invalid macOS code signatures after installation.

Packages like OpenMM and RDKit ship native `.so` and `.dylib` files. When Python imports
one of those files, macOS checks the signature. If the signature is broken, macOS kills
the whole process immediately. That is why the error looked like this:

```bash
zsh: killed chemem ...
```

instead of showing a normal Python traceback.

There were two parts to the bug:

1. The CLI startup path imported too much too early.
   Even `chemem --help` could pull in heavy runtime modules.

2. Fresh macOS conda installs could contain invalidly signed native libraries.
   As soon as one of those libraries was imported, macOS killed Python.

So the package fix in ChemEM is:

1. Keep the CLI lazy so basic commands do not import the full runtime stack.
2. Keep packaging conda-native (no install hooks that rewrite env files).

If you still hit signature failures on very old conda installations, this is usually an
installer/toolchain problem in that conda stack, not a ChemEM recipe bug.

Recommended before retrying:

```bash
conda config --set channel_priority strict
conda update -n base -c conda-forge conda
```

Last-resort recovery for a broken env (macOS only):

```bash
find "$CONDA_PREFIX" -type f \( -name "*.dylib" -o -name "*.so" \) -exec codesign --force --sign - {} \;
```

---




### Dependencies

Some protocols require others to run first. For example, docking requires a binding site:

So `chemem <conf_file> --dock` results in the ordered pipeline:

1. `binding_site`
2. `dock`

---

## CLI syntax

```bash
chemem <config> [shared options] [protocol selection flags] [protocol options]
```

- `<config>`: path to your ChemEM configuration file.
- Protocol options are all registered on the single CLI parser (so flags exist even if you’re not running that protocol; the protocol decides what it uses).

---

## Shared options (apply to multiple protocols)

| Option | Type | Default | Meaning |
|---|---:|---:|---|
| `--platform` | str | `None` | OpenMM platform selection (e.g., `CPU`, `OpenCL`, `CUDA`). |
| `--output` | str | `None` | Output directory (overrides config/system default). |
| `--ncpu` | int | `max(1, os.cpu_count() - 2)` | CPU count used by protocols that parallelize. |
| `--no-map` | flag | `False` | Disable density-map usage (sets `system.density_map = None`). |


---

## Protocol: `binding_site`

**Action:** Prepare / identify a binding site to dock into.

**Run it:**

```bash
chemem <conf_file> --binding-site
# or
chemem <conf_file> -b
```

### Options

| Option | Type | Default | Notes |
|---|---:|---:|---|
| `--probe-sphere-min` | float | `3.0` | Minimum probe sphere radius used during site detection. |
| `--probe-sphere-max` | float | `6.0` | Maximum probe sphere radius used during site detection. |
| `--first-pass-thr` | float | `1.73` | Threshold for the first pass of site detection. |
| `--fist-pass-cluster-size` | int | `35` | Cluster size cutoff for first pass (*note: flag name is `fist` in code*). |
| `--second-pass-thr` | float | `4.5` | Threshold for the second pass. |
| `--third-pass-thr` | float | `2.5` | Threshold for the third pass. |
| `--binding-site-padding` | float | `6.0` | Padding around detected site region (Å). |
| `--binding-site-grid-spacing` | float | `0.5` | Grid spacing for site maps / masks (Å). |
| `--n-overlaps` | int | `2` | Overlap requirement used in site assembly/merging. |
| `--n-opening-voxels` | int | `10` | Morphological opening strength in voxels (site mask cleanup). |
| `--voxel-buffer` | float | `1.5` | Additional buffer around voxelized site (Å). |
| `--fall-back_radius` | float | `15.0` | Fallback radius (Å) if site finding fails / is forced. |
| `--lining_residue_distance` | float | `2.0` | Distance cutoff (Å) for lining-residue identification. |
| `--force-new-site` | flag | `False` | Force creation of a new site instead of reusing cached/previous. |

---

## Protocol: `dock`

**Action:** Dock ligands into the prepared binding site.

**Run it (explicitly):**

```bash
chemem <conf_file> --dock
# or
chemem <conf_file> -d
```

> Because docking depends on `binding_site`, selecting `--dock` will run `binding_site` first.

### Options

| Option | Type | Default | Notes |
|---|---:|---:|---|
| `--rescore` | flag | `False` | Rescore generated poses with a single frame MMGBSA). |
| `--flexible-rings` / `-fr` | flag | `False` | Allow hetrocyclic ring flexibility. |
| `--split-site` / `-ss` | flag | `False` | Split a large binding site into sub-sites for docking. |
| `--no-para` / `-np` | flag | `False` | Disable protocol parallelization (run serially). |
| `--n-global-search` | int | `1000` | Global search budget (number of confomrations generated in the ACO step). |
| `--n-local-search` | int | `10` | Local refinement per iteration (number of solutions taken forawrd to nealder-mead local optimisation). |
| `--bias-radius` / `-br` | float | `12.0` | Radius (Å) for biasing/sampling around a site center. |
| `--cluster-docking` | float | `2.0` | Clustering cutoff (Å) used to merge similar poses. |
| `--energy-cutoff` | float | `1.0` | Energy cutoff (units depend on scoring) for filtering poses. |
| `--minimize-docking` | flag | `False` | Run minimization after docking (OpenMM/OpenFF-based). |
| `--aggregate-sites` | flag | `False` | Aggregate docking across multiple sites into one result set. |

---



## Covalent ligands

ChemEM can treat a ligand as covalently attached to a protein atom. When a
covalent link is declared, the refine / annealing protocols see a real
topology bond between the ligand atom and the protein atom — bond, angle,
and torsion terms straddling the junction are parameterized from a capped
junction fragment via OpenFF, correct 1-2/1-3/1-4 nonbonded exclusions are
auto-generated by OpenMM, and the ligand's partial charges are re-polarized
near the junction and renormalized to integer net charge.

Non-covalent ligands are unaffected — when no covalent block is present for
a ligand, the existing code path runs unchanged.

### Config syntax

Covalent fields are **per ligand** and attach to the most recently declared
`ligand = …` entry. You can mix covalent and non-covalent ligands in the
same config.

```
ligand = warhead.sdf
covalent_ligand_atom = LIG:0:C3
covalent_protein_atom = A:CYS:145:SG
covalent_bond_order = SINGLE                 # optional, default SINGLE
# covalent_delete_ligand_atoms  = [C8, O2]   # optional, only for multi-atom leaving groups
# covalent_delete_protein_atoms = [HG]       # optional, overrides auto H-removal
```

**Atom spec grammar:**

| Spec | Format | Example |
|---|---|---|
| `covalent_ligand_atom` | `LIG:<index>:<ATOMNAME>` | `LIG:0:C3` |
| `covalent_protein_atom` | `CHAIN:RESNAME:RESNUM:ATOMNAME` | `A:CYS:145:SG` |

Ligand atom names use ChemEM's own convention: `element-upper + 1-based
count-within-element` (e.g. the third carbon is `C3`, the first sulfur is
`S1`).

**Field reference:**

| Field | Type | Default | Notes |
|---|---|---|---|
| `covalent_ligand_atom` | str | — | Required. Ligand-side attachment atom spec. |
| `covalent_protein_atom` | str | — | Required. Protein-side attachment atom spec. |
| `covalent_bond_order` | str | `SINGLE` | One of `SINGLE`, `DOUBLE`, `TRIPLE`. |
| `covalent_delete_ligand_atoms` | list[str] | auto | Ligand atom names to remove on bond formation. If omitted, ChemEM auto-removes one H from the ligand attachment atom when valence demands it. Required for multi-atom leaving groups (activated esters, halide + context, etc.). |
| `covalent_delete_protein_atoms` | list[str] | auto | Protein atom names to remove from the target residue. If omitted, ChemEM auto-removes one H (e.g. CYS HG, SER HG, LYS HZ1). Use this to override auto-selection when a residue has multiple equivalent Hs. |

**Auto-detection of leaving Hs.** The common case — nucleophilic attack on
Cys SG, Ser/Thr OG, Lys NZ, His NE/ND, Tyr OH — does not require any
deletion fields. ChemEM counts current valence vs. standard valence and
auto-removes one bonded H from each side when the new bond would exceed it.
Removals are logged as `[covalent] auto-removing …`.

**Supported residue templates** (for junction fragment parameterization):
CYS, CYX, SER, THR, TYR, LYS, HIS/HIE/HID, ASP, GLU. Unknown residues fall
back to a generic methyl cap on the attachment element.

### Running a refine job with a covalent ligand

Covalent attachment plugs into the existing `refine` protocol — no new CLI
flag is required. Write a config file, then:

```bash
chemem my_covalent_conf.txt --refine
```

Minimal example config (`acrylamide_cys145.txt`):

```
protein = mpro.pdb
ligand  = acrylamide.sdf
output  = run_covalent
densmap = mpro.mrc       # optional — omit with no_map = True for pure MM refine
resolution = 2.2         # required if densmap is set

# Covalent attachment: Michael acceptor C on the ligand → Cys145 SG
covalent_ligand_atom  = LIG:0:C3
covalent_protein_atom = A:CYS:145:SG
# No deletion fields: the β-H on C3 and HG on Cys145 are both auto-detected.
```

Then run:

```bash
chemem acrylamide_cys145.txt --refine
# or for a pure-MM refine (no density term):
chemem acrylamide_cys145.txt --refine --no-map
# annealing variant:
chemem acrylamide_cys145.txt --refine --annealing
```

**What you should see in the log:**

```
[covalent] protein deletions: removed ['HG'] from A:CYS:145 (auto)
[covalent] fragment parameterized: bond=ok, angles=2, dihedrals=3
[covalent] injected bond A:CYS:145:SG — LIG_0:C3 (k=269.8, r0=1.760)
[covalent] LIG_0 net charge: -0.023 (pre) → -0.118 (polarized) → +0.000 (renormalized)
```

If you see `bond=missing` or `[covalent] WARNING: no junction bond params`,
the fragment parameterization failed — most commonly because the target
residue isn't in the template table, or because the protein atom spec
doesn't resolve to a real atom after the parser's residue renaming.

### Python / standalone test script

You can exercise the covalent fragment pipeline end-to-end without a full
refine run, which is useful for sanity-checking bonded term extraction on
a new warhead before committing to a density refinement:

```python
from ChemEM.parsers.models import CovalentLinkSpec
from ChemEM.parsers.ligand_parser import LigandParser
from ChemEM.parsers.covalent_fragment import build_and_parameterize_fragment

# N-methylacrylamide-like warhead; C1 is the Michael-acceptor terminal carbon.
ligs = LigandParser.load_ligands(
    "C=CC(=O)N",
    protonation=False, chirality=False, rings=False, name="LIG_0",
)
lig = ligs[0]

spec = CovalentLinkSpec(
    ligand_atom_spec="LIG:0:C1",
    protein_atom_spec="A:CYS:145:SG",
)
# build_and_parameterize_fragment expects the protein-side fields to be
# pre-resolved (normally Config.create_system does this via apply_protein_deletions).
spec.resolved_protein_chain    = "A"
spec.resolved_protein_resname  = "CYS"
spec.resolved_protein_resnum   = 145
spec.resolved_protein_atom_name = "SG"
lig.covalent_link = spec

terms = build_and_parameterize_fragment(lig.mol, spec)
print("bond:", terms["bond_params"])              # (k_kcal/Å², r0_Å)
print("angles:",    len(terms["angles"]))         # list of (names, k, theta0)
print("dihedrals:", len(terms["dihedrals"]))      # list of (names, k, phase, per)
print("ligand charges:", terms["ligand_charges"]) # dict name -> partial charge
```

Expected output for this warhead (Michael acceptor sp² carbon → Cys SG):

```
bond: (269.8, 1.760)
angles: 2          # C2-C1-SG and H2-C1-SG (sp² C1 has exactly two non-C2 neighbors)
dihedrals: 2       # C3-C2-C1-SG and H3-C2-C1-SG (π-system 2-fold barrier)
```

A **saturated** warhead carbon (sp³) would instead give three angles and
more dihedrals because the attachment atom has three non-H neighbors plus
three Hs.

### Caveats

1. **Ligand variants.** A covalent ligand entry must resolve to exactly one
   molecule. If the SDF or SMILES produces multiple protonation variants,
   `create_system` raises an error — re-declare the ligand with a single
   variant (`max_ligand_varients = 1`, or pre-protonated SDF).
2. **Protein-side junction charges.** Ligand-side atoms near the junction
   get re-polarized from the fragment. Protein heavy atoms keep their
   AMBER charges (localized approximation; total system charge is still
   integer).
3. **Residue template coverage.** Residues outside
   CYS/CYX/SER/THR/TYR/LYS/HIS/ASP/GLU fall back to a generic methyl cap.
   Extend `_SIDECHAIN_TEMPLATES` in
   `ChemEM/parsers/covalent_fragment.py` to add more.
4. **Docking.** Covalent support currently only wires into the refine /
   annealing path. `CovalentLinkSpec` is designed as standalone data so
   docking can consume it later without a second refactor.

---

## Notes for maintainers

- All protocol specs live in `ChemEM/protocol_spec.py` in the `REGISTRY`.
- Protocol selection flags are auto-generated from registry keys (underscore → hyphen).
- If you add a new protocol:
  1) Create a new `ProtocolSpec(...)` in `REGISTRY`.
  2) Provide `deps(args)` and (optionally) `add_args(parser)`.
