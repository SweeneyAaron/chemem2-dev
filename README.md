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
python -m ChemEM 7jjo_conf.txt --dock --no-map --minimize-docking --dock-rescore-mmgbsa
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
| `--sigma-ref` | float | `0.6` | Q-score reference Gaussian width (Å). Read by `--score`, `--orchestrate` and `smart_ligand_refine2`. |

### Protein preparation determinism

Unlike every other shared option, these are applied *before* the protein is built,
so they are read when the config is loaded rather than by `apply_overrides`.

| Option | Type | Default | Meaning |
|---|---:|---:|---|
| `--prep-platform` | str | `CPU` | OpenMM platform for the two minimisations inside protein preparation. `Reference` also gives cross-machine identity; `inherit` restores the old auto-selection, which is **not** reproducible. |
| `--prep-threads` | int | `1` | Thread count for the prep platform. |
| `--prep-seed` | int | `1234567` | Seed for PDBFixer's rebuilt-atom dynamics and hydrogen placement. Must be non-zero. |
| `--no-deterministic-prep` | flag | `False` | Restore the previous, irreproducible preparation. |
| `--prep-clash-relief-steps` | int | *unset* | Cap PDBFixer's clash-relief dynamics. Large speedup, **not safe by default**. See below. |
| `--no-prep-h-implicit` | flag | `False` | Drop implicit solvent from hydrogen placement. Faster but **shifts scores**. See below. |
| `--no-cache-protein` | flag | `False` | Re-prepare every run instead of reusing the cache (cache is **on** by default). |
| `--protein-cache-dir` | str | *see below* | Cache location. Defaults to `$CHEMEM_CACHE_DIR`, else `$XDG_CACHE_HOME/chemem`, else `~/.cache/chemem`. |
| `--refresh-protein-cache` | flag | `False` | Ignore any cached entry and rewrite it. |

**Why this exists.** Protein preparation used to produce different coordinates in
every process, which put a floor of ~1.9 ECHO score units under every result —
enough to swamp the gap between competing docked poses, and enough to make an
absolute-score benchmark or a weight refit partly fit noise. Three independent
causes, all in the OpenMM work done during preparation:

1. `PDBFixer.addMissingAtoms()` ran with no seed. Its `LangevinIntegrator` seed
   defaulted to 0, which OpenMM reads as "choose a fresh seed per Context", and
   when a rebuilt heavy atom lands within 1.3 Å of a neighbour it runs **up to
   2000 fs of 300 K Langevin dynamics**. That is what moved atoms up to 9.8 Å
   between runs — minimisation alone could not.
2. Both that minimisation and `Modeller.addHydrogens` built their `Context` with
   `platform=None`, so OpenMM auto-selected the fastest platform (OpenCL here),
   which is single-precision and explicitly allowed to be non-reproducible.
3. `Modeller.addHydrogens` seeds each new hydrogen at a **random** offset from its
   parent before relaxing it — OpenMM's own comment reads "The hydrogens were
   added at random positions" — drawing from Python's global `random`.

Only (3) is seedable from Python, which is why seeding `random` alone never fixed
it: (1) and (2) live in OpenMM's C++ kernels. All three are now pinned.

**Scope of the guarantee.** Repeated runs *on one machine* agree. Identical
coordinates across machines, OpenMM builds or pdbfixer versions are not promised —
CPU SIMD dispatch differs. Use `--prep-platform Reference` if you need that.

**What this buys, measured on 9e26:**

| | before | after |
|---|---:|---:|
| prepared protein, two processes | up to **9.8 Å** apart | **0.000e+00** (bitwise identical) |
| ECHO score of a fixed pose, 3 processes | **1.89** units spread | **0.000e+00** on every term |
| `--dock` scores, two runs at fixed `--ncpu` | varied | **0.000e+00** (bit-identical) |
| residue map coverage | 1827 / 2532 | **2532 / 2532** |

The residue-map jump is a side-effect worth knowing about: the auto-selected
OpenCL platform is single-precision, so it perturbed *every* coordinate by roughly
1e-4 Å at these magnitudes — past the 1e-5 Å tolerance `build_residue_map_by_positions`
matches on. 705 residues were silently failing to map, which quietly broke
`get_residue_mapping`, `--manual-site` and covalent atom specs for those residues.

**The docking search is reproducible from its seed alone** — see
`--dock-seed` below.

### Preparation cost

Determinism is not free — the CPU platform is far slower than the auto-selected
one. On 9e26 (39318 atoms) preparation went from seconds to 492 s. **The cache is
the safe mitigation and is on by default**; the two speed flags below are opt-in
because neither is free.

| configuration | prep on 9e26 | score impact |
|---|---:|---|
| **default** (PDBFixer's own clash relief, GB hydrogen placement) | 492 s | — |
| warm cache — **the default path after the first run** | **60 s** | none |
| `--prep-clash-relief-steps 600` | 236 s | none *on 9e26*; **degrades 7bxu** — see below |
| `--no-prep-h-implicit` | 387 s | **+0.6 units** — needs a weight refit |
| both | 131 s | both caveats |

**Where the time goes.** 76% of preparation is `PDBFixer.addMissingAtoms`, and 97%
of *that* is a clash-relief loop: up to 10 rounds of 200×5 fs Langevin dynamics
trying to push rebuilt atoms ≥1.3 Å apart. On a heavily-repaired receptor it never
gets there. The closest-contact trace on 9e26, in Å:

```
steps:     0     200    400    600    800   1000  1200  1400  1600  1800  2000
nearest: 0.821  0.564  0.188  0.952  0.574 0.712 0.733 0.756 0.855 0.893 0.890
```

It gets *worse* before better, peaks at 600, and never beats that again — and the
loop keeps its best snapshot, so the last 1400 steps cost ~250 s for nothing.

**But capping is not safe as a default**, which is why it is opt-in. The useful
snapshot lands at a structure-dependent iteration:

| | uncapped | capped at 600 |
|---|---|---|
| 9e26 worst contact | 1.252 Å | 1.252 Å — *identical structure* |
| 7bxu worst contact | 1.052 Å | **0.655 Å**, plus ~39 more sub-2 Å contacts |

A "stop when it stops improving" rule does not rescue it either: on 9e26 the peak
arrives only after two consecutive worsening rounds. If you set
`--prep-clash-relief-steps`, check the resulting contacts on your structure. `0`
skips the dynamics entirely.

**Why `--no-prep-h-implicit` is off despite being nearly 2× faster.** Hydrogen
placement minimises a system built from whatever force field it is handed, and
ChemEM's includes `implicit/gbn2.xml` — a `CustomGBForce` over every atom with no
interaction group, minimised 50 times. Dropping it leaves `hbond_raw` and every
non-electrostatic term bit-identical, but still moves `echo_total` by up to 0.63
units, because the ECHO electrostatic grid is built with `collapse_hydrogens=False`
— per-atom charges *including* hydrogens — so any change in H positions rewrites
it. That is comparable to the run-to-run noise this work removed, so it is opt-in
until the ECHO weights are refit.

**What is being rebuilt.** Not loops — ChemEM never rebuilds those, because
`model_to_fixer_interchange` round-trips through a temp PDB and `PDBFile` writes no
SEQRES, so `findMissingResidues` has nothing to diff against. On 9e26 it completes
**1152 missing side-chain heavy atoms across 354 residues**. This cannot be
skipped: OpenMM's `ForceField` matches residues by exact atom signature plus bond
graph, so a residue missing one heavy atom raises `No template found for residue`.

The cache stores only
the prepared topology and positions, keyed on the input file, force field, prep
settings and library versions; everything downstream is rebuilt each run. It
stores positions as float64 nanometres rather than a PDB **on purpose** — the
original↔prepared residue map matches backbone atoms to 1e-5 Å, so a format that
re-quantises coordinates would return an empty map with no error. A `prepared.pdb`
is written alongside for inspection and is never read back.

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
| `--dock-rescore-mmgbsa` | flag | `False` | After docking, score the generated poses with a single-frame MM-GBSA into `mmgbsa_rescore.txt`. (Was `--rescore`, still accepted.) To score poses from a *file*, use `--score --score-with mmgbsa` instead. |
| `--dock-seed` | int | *random* | Seed for the ACO search. Random each run and always logged. See below. |
| `--echo-lattice-anchor` | str | `off` | `off` / `global` / `centroid`. Anchor the ECHO grid lattice so distant atoms cannot shift it. See below. |
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
| `--dock-full-map` | flag | `False` | Score the map term against the cropped binding-site map instead of the alpha-mask. See below. |
| `--sci-weight` | float | `2.5` | Scale factor for the SCI map score. |
| `--mi-weight` | float | `100.0` | Scale factor for the mutual-information map score. |
| `--inner-map-score` | `0`\|`1` | `1` | Map score driving the search (ant sampling + inner Nelder-Mead): `0` = MI, `1` = SCI. |
| `--outer-map-score` | `0`\|`1` | `0` | Map score driving the final polish, i.e. how returned poses are **ranked**: `0` = MI, `1` = SCI. |

### Choosing the map term

The density term has two independent scorers, and by default the search and the ranking
use *different* ones: SCI guides sampling (`--inner-map-score 1`) while mutual information
decides the final order (`--outer-map-score 0`). Set both to the same value to use one
term throughout, and scale each with `--sci-weight` / `--mi-weight` respectively (setting
the corresponding weight to `0` mutes that term). The `[dock-map]` line in the log reports
the effective source, terms and weights for every site, e.g.

```
[dock-map] site 0: source=segmented box(zyx)=(10, 11, 11) ... inner=SCI outer=MI mi_weight=100.0 sci_weight=2.5
```


### `--dock-full-map`

By default the docking map term (MI/SCI) scores against the **alpha-masked,
blob-segmented** per-site map: the confidence map cropped to the site, multiplied by an
alpha-sphere (or SES) mask, an Otsu gate and a boundary distance transform, with
everything outside the accepted density blobs zeroed. That envelope stops the scorer
fitting to off-blob density, but it also means docking can never be rewarded for
occupying ligand density that the mask clipped.

`--dock-full-map` instead scores against the **FDR confidence map cropped to the same
site box**, with no masking. The crop is bit-exact on the same voxel grid, so nothing is
resampled or shifted. It also makes binding sites that segmentation dropped entirely
dockable again — their box is taken from the binding-site geometry and snapped onto the
map grid.

Caveats:

- The full map contains **protein density**, and the segmented map is amplitude-rescaled
  by the boundary distance transform, so the two are not on the same scale even inside
  the blob. `--mi-weight` (default `100.0`) and `--sci-weight` (default `2.5`) are
  calibrated for the segmented map and will need re-tuning; try `--mi-weight 10` first.
- **Absolute dock scores are not comparable across modes** — MI re-derives its histogram
  bounds from the box on every evaluation. Compare RMSD-to-native and pose ranking only.
- `--refine-to-diff-map` becomes a no-op (both branches resolve to the same map).
- Sites kept only by the relaxed gate get no orchestrator density-fit metrics
  (coverage / precision / CCC), since those are defined on the segmented envelope.

---



## Protocol: `score`

Score poses that already exist. One protocol, one CSV, any combination of scorers:

```bash
chemem <conf_file> --score --score-with echo,mmgbsa,qscore,density
# or
chemem <conf_file> -sc --score-with all
```

Poses come from the config, one pose per SDF record — either `ligand = poses.sdf`
(a multi-record SDF, e.g. the output of a previous `--dock`) or
`ligands_from_dir = <dir>`. `--score` also composes with `--dock` in one run, in
which case it scores the poses docking just produced.

This replaces four separate things: the `--rescore-poses` and `--mapq-score`
protocols and the unregistered `score_poses` / `echo_terms` scripts.
`--rescore-poses`/`-rp` and `--mapq-score` still work as deprecated aliases; they
select the right scorer, print a deprecation notice and write the new output layout.

### Scorers

`--score-with` takes a comma-separated list (repeatable, and `all` selects every
scorer). **The order matters**: it is the column order in the CSV, and the first
scorer's headline column is the default ranking. Bare `--score` means `echo`.

| Scorer | Headline column | Needs | Cost |
|---|---|---|---|
| `echo` | `echo_total` | binding site + segmentation | ~15 s/pose |
| `mmgbsa` | `mmgbsa` | prepared receptor | ~20 s/pose |
| `qscore` | `qscore` | density map | ~0.2 s/pose |
| `density` | `density_overlap` | density map | ~1 s/pose |
| `strain` | `ligand_strain` | nothing | negligible |
| `clash` | `protein_ligand_clash_penalty` | prepared receptor | negligible |

**Dependencies are per-scorer.** `--score` is the only protocol whose `deps()`
reads the CLI flags: `--score-with echo` pulls in
`binding_site` → `alpha_mask` → `confidence_map`, while `--score-with qscore`
runs none of them. Scoring Q-score alone therefore no longer pays for
segmentation, which `--mapq-score` did not either but `--rescore-poses` always did.

### Output

Lands in `<output>/score/` (`--score-out` to change the subdirectory):

| File | When | Contents |
|---|---|---|
| `pose_scores.csv` | always | one row per pose: identity, then each scorer's headline, then each scorer's detail block, then the status columns |
| `score_run.json` | always | selected scorers, resolved options, the ECHO weights used, per-scorer timings and failure counts |
| `pose_scores.json` | `--score-json` | everything the flat CSV cannot hold: per-atom Q-scores, per-feature density metrics, SCI sub-terms |
| `<stem>_scored.sdf` | `--score-sdf` | poses ranked best-first, every score as an SD property |
| `echo_protein_h.csv` | `--score-echo-protein-h` | per-donor attribution of the receptor-H relaxation |
| `<stem>_mmgbsa_minimised.sdf` + `mmgbsa_receptor.pdb` | `--score-mmgbsa-write-minimised` | the pocket-relaxed ligand poses, and the receptor to load them against |

Column names are unchanged from the tools this replaces (`echo_total`, `raw_*`,
`w_*`, `qscore`, `density_*`, `mmgbsa*`), so existing analysis scripts keep working.

**A scorer that fails does so alone.** Each scorer is guarded per pose; a failure
writes `<scorer>_error` into that row and the other scorers still report. One
unscorable pose never costs you the other ninety-nine.

### The ECHO terms

`--score-with echo` reports the weighted total **plus every individual term, both
weighted and unweighted**. They reconcile with the total exactly, matching
`ECHOScore::score`:

```
echo_total  = -(echo_linear + map_score) + bias + constraint + covalent
echo_linear = sum(w_<term>)          # over the 17 weighted channels
```

`raw_aromatic` and `raw_nonbond` are lumped duplicates of their split channels
(`aromatic_attr + aromatic_clash`, `nonbond_attr + nonbond_rep + clash`); they
are reported for convenience and deliberately excluded from `echo_linear`.

### Options

Every option is `--score-<scorer>-*` prefixed. The old `--rescore-*` and
`--per-atom` spellings still work and land on the same values.

| Option | Type | Default | Notes |
|---|---:|---:|---|
| `--score-with` | list | `echo` | Scorers to run, in column order. `all` selects every one. |
| `--score-out` | str | `score` | Output subdirectory. |
| `--score-site` | str | *auto* | Force one binding-site key. Default: the site whose box contains the pose, else the nearest site centroid. |
| `--score-json` | flag | `False` | Also write `pose_scores.json`. |
| `--score-sdf` | flag | `False` | Also write ranked SDFs. |
| `--score-rank-by` | str | *first headline* | Column the SDFs are sorted by. |
| `--score-echo-engine` | str | `docking` | `docking` or `docking_v2`. Must match the engine that produced the poses. |
| `--score-echo-rep-max` | float | *`--repulsion-cap-polish`* | Repulsion cap. See below — this is **not** 5.0. |
| `--score-echo-interaction-cutoff` | float | `6.0` | ECHO interaction cutoff (Å). |
| `--score-echo-electro-clamp` | float | `2.0` | Electrostatic repulsion clamp. |
| `--score-echo-minimise-hydrogens` | flag | `False` | Relax the ligand's polar-H torsions first. See below. |
| `--score-echo-protein-h` | flag | `False` | Also relax the receptor's rotatable donor H's. |
| `--score-qscore-sigma-ref` | float | *`--sigma-ref`* | Q-score reference Gaussian width. |
| `--score-qscore-per-atom` | flag | `False` | Record per-atom Q-scores; implies `--score-json`. |
| `--score-density-region` | str | `full` | `full`, `box` or `site`. See the warning below. |
| `--score-density-box-size` | float | `24.0` | Cube edge for `--score-density-region box`. |
| `--score-mmgbsa-minimise` | flag | `False` | Relax the ligand in a pinned pocket first. Strongly recommended on raw docked poses. |
| `--score-mmgbsa-pocket-radius` | float | `12.0` | Pocket radius for that relaxation. |
| `--score-mmgbsa-write-minimised` | flag | `False` | Write the relaxed poses out. Requires `--score-mmgbsa-minimise`. |

### Writing out the minimised poses

`--score-mmgbsa-write-minimised` (which requires `--score-mmgbsa-minimise`) writes
the relaxed geometry the energy was actually evaluated on:

```bash
chemem <conf> --score --score-with mmgbsa \
    --score-mmgbsa-minimise --score-mmgbsa-write-minimised
```

* `<stem>_mmgbsa_minimised.sdf` — one file per input source, holding each pose's
  relaxed ligand with `mmgbsa` and `mmgbsa_min_shift_A` as SD properties. Atom order
  matches the parsed input pose, so it diffs cleanly against `--score-sdf` output.
* `mmgbsa_receptor.pdb` — the receptor, so the SDF has something to load against.

**Only the ligand is minimised.** The pocket residues within
`--score-mmgbsa-pocket-radius` are included in the truncated minimisation system but
held by a positional restraint, and `_pocket_minimise` discards their relaxed
coordinates in favour of the reference ones — the returned complex is
`reference receptor + relaxed ligand`. So `mmgbsa_receptor.pdb` is the *prepared
input structure, unchanged*; it is written for convenience, not because it moved.
The log says so on every run.

The `mmgbsa_minimised_sdf` column in the CSV points each row at its file, and
`mmgbsa_min_shift_A` is the all-atom RMSD between the input and relaxed pose —
verified to match the written coordinates to 6 decimal places.

### Three different maps in one row

A row from `--score-with echo,qscore,density` reports three density numbers
against **three different maps**, and they are not comparable:

* `qscore` — the full `system.density_map`.
* `density_*` — whatever `--score-density-region` selects. `full` (the default)
  is the whole map; `box` is a cube around the pose; `site` is the segmented
  binding-site map, which alpha-mask has multiplied by a boundary EDT (~38×), so
  its amplitudes are on a completely different scale. `site` is what
  `--orchestrate` scores against internally.
* `map_score` (ECHO's) — the segmented site map, or the cropped confidence map
  under `--dock-full-map`.

`density_region` is recorded on every row and in `score_run.json`.

### Optional hydrogen relaxation

`--score-echo-minimise-hydrogens` relaxes the ligand's polar (donor N/O/S–H)
torsions against ECHO before scoring, so a pose is not penalised for whatever H
placement its SDF happened to carry. These are the same torsions the ACO search
already samples, so it puts an externally-supplied pose on the same footing as a
docked one.

Every candidate torsion is filtered through `only_h_moves_on_rotation`, so
**"ligand hydrogens only" is guaranteed by topology, not by a restraint** — no
heavy atom *can* move. Measured on a real pose, the only channels that respond
are `hbond_raw`, `ligand_torsion` and `nonbond_rep`; `clash`, `nonbond_attr`,
`aromatic*`, `electro*`, the hydrophobic channels, `saltbridge_raw`,
`unsat_polar`, `ligand_intra` and `desolvation_penalty_scaled` all come back
identical to the last bit. Relaxation therefore cannot flatter a pose's shape,
sterics or electrostatics.

Limits, by construction: only *rotatable donor* H's move (a `-NH2` is excluded,
since rotating its pivot moves both H's), and bond lengths and X–H angles are
never touched. Because this optimises ECHO with ECHO, the pre-relaxation total
is always reported alongside as `echo_total_prehmin`.

Relaxation runs **before any scorer scores the pose**, and the receptor is
restored only **after every scorer has read it**. That ordering matters when you
combine scorers: Q-score, density, strain and clash are heavy-atom only and are
unaffected, but MM-GBSA is all-atom and does see the relaxed hydrogens.

**Cost:** roughly 10–15 s per pose. Nearly all of it is API overhead —
`run_echo_score` rebuilds the entire `PreComputedData` object from Python on
every call, so one evaluation costs ~200 ms even though the scoring itself is
~0.05 ms. `--score-echo-h-min-grid` / `-passes` / `-maxiter` are the cost knobs.

### Reproducing a docking score

Two things stand between `echo_total` and the number `--dock` printed:

1. **The repulsion cap.** `run_aco_docking` ranks and returns its poses from a
   final Nelder-Mead polish at `repCap_final_nm` (`--repulsion-cap-polish`,
   default 15.0), *not* the `rep_max=5.0` default baked into the
   `run_echo_score` pybind signature. `--score-echo-rep-max` therefore defaults
   to `--repulsion-cap-polish`; scoring at 5.0 makes every pose look several
   score units better than docking said it was.
2. **Protein preparation.** It is deterministic and cached by default, but a cold
   preparation on a different machine still yields slightly different
   coordinates, and an ECHO score moves with them. This is upstream of scoring
   and affects `--dock` ranking identically.

`--outer-map-score 1` (SCI) also cannot be reproduced: `run_echo_score` has no
`use_map_score` argument and always uses mutual information. The protocol logs a
warning when you ask for it. Every non-map term is unaffected. Note this is a
different quantity from the `density_sci` column, which the density scorer
computes against its own region map.

### Adding a scorer

One file plus one line. Subclass `PoseScorer`
(`ChemEM/protocols/score/scorers/base.py`), declare `NAME`, `COLUMNS`,
`HEADLINE` and `DEPS`, implement `score(pose, row)`, and add an entry to
`SCORER_REGISTRY` in `ChemEM/protocols/score/scorers/__init__.py`. The registry
imports lazily, so a new scorer costs nothing to the runs that do not select it.

---



## Docking reproducibility (`--dock-seed`)

The ACO search is stochastic. **By default each run draws a fresh seed**, which is
always written to the log and the docking summary:

```
[dock] seed: 8216489998807125027 (random) -- reproduce this run with --dock-seed 8216489998807125027
```

Pass that number back with `--dock-seed` and you get the run again, byte for byte.

Random-by-default is deliberate: the seed used to be a hard-coded `1234567`, so
every run on every machine replayed one identical trajectory. Rerunning could never
reveal that a pose was a lucky draw rather than a converged answer, and a benchmark
or weight fit built on it was fitting a single realisation. Run a few seeds to see
the real spread; pass a fixed one when you need repeatability.

**The seed alone determines the result.** Ants are seeded from
`(seed, iteration, ant index)`, so:

| | before | after |
|---|---|---|
| same command twice | identical (fixed seed) | identical *given the same seed* |
| `--ncpu 4` vs `--ncpu 8`, same seed | differed by up to **0.5 units** | **identical** |
| same ligand listed twice in one run | −6.217939 vs **−6.221098** | **identical** |

The last two were real bugs. Ants were seeded per *thread*, so the thread count
decided which ant drew which stream; and the generator was `static thread_local`,
so it was seeded once per thread per *process* and the stream carried across
ligands — meaning adding or reordering a ligand silently changed every later
ligand's result. Both are fixed, along with a tie-ordering hazard in the final
polish (candidates were merged under an `omp critical` and then sorted with a
non-stable sort, so exactly-equal scores could permute).

**Note for anyone comparing against older numbers:** per-ant seeding is a different
RNG stream, so `--dock-seed 1234567` does *not* reproduce pre-change results. The
committed benchmark CSVs under `ChemEM/benchmark/` predate this and need
regenerating.

---

## ECHO grid lattice anchoring (`--echo-lattice-anchor`)

The ECHO electrostatic, hydrophobic and environment grids are built on a lattice
whose origin is `min(all protein atoms) - padding`. The lattice **phase** is
therefore set by whichever atom sits at the protein's extremity — often a rebuilt,
poorly-determined atom tens of Ångström from the binding site. Move it and the
whole sampling lattice slides; since the scorer reads the grids by trilinear
interpolation, `electro_attractive` changes even though no physics did.

Measured on 9e26 before protein preparation was made deterministic: the parent
origin moved 3.02 Å between runs, the grid shape changed by up to 4 voxels, and
the cropped site sub-box picked up a 0.358 Å (≈1 voxel) shift.

| Mode | Behaviour |
|---|---|
| `off` (default) | Previous behaviour, bit-for-bit. |
| `global` | Snap the origin to the absolute lattice `{i·spacing}`. The phase can never move. |
| `centroid` | Make the binding-site centroid an exact lattice node, so the grid follows the site under rigid translation of the receptor. |

Under `global`/`centroid` the origin may still track the bounding box, but only in
**whole voxels** — so the set of sampled physical points is unchanged and every
trilinear lookup is identical.

Prefer `global` unless the config supplies an explicit `centroid =`: a derived
centroid comes from the translation points and is itself segmentation-dependent.

**Why it is off by default.** Anchoring shifts absolute ECHO scores by a small
constant (measured: ±0.01–0.04 units per pose on 9e26 — mixed sign, so it is a
resampling shift rather than a bias). That is small, but it is enough to make
published numbers and the fitted `default_v1` weights slightly stale, so it is
opt-in until the weights are refit. Note this is *not* needed for reproducibility:
once protein preparation is deterministic (see **Protein preparation determinism**
above), the run-to-run variance is already zero with the flag off. Anchoring is
defence-in-depth against distant-atom jitter — a re-cropped receptor, a translated
copy, a differently-repaired loop.

The `electro_cutoff` is **not** the leak here, despite appearances: the Coulomb
kernel hard-rejects `r2 > cutoff2`, so an atom 30 Å away contributes exactly 0.0.
Separately, `depth_norm` and `constriction` are normalised by protein-wide maxima;
that is intended behaviour and is deliberately left alone.

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

**Several bonds on one ligand (crosslinkers).** A `covalent_ligand_atom` line
*opens a bond block*; the fields after it belong to that bond. Repeat it to
declare another bond on the same ligand — no new keywords, and a config with one
block per ligand behaves exactly as before:

```
ligand = C=CCN
covalent_ligand_atom = LIG:0:C2
covalent_protein_atom = M:CYS:823:SG
covalent_bond_order = SINGLE
covalent_ligand_atom = LIG:0:N1              # opens the SECOND bond
covalent_protein_atom = C:GLY:75:C
covalent_bond_order = SINGLE
covalent_delete_protein_atoms = ['OXT']
```

All of a ligand's bonds are parameterized from a **single** capped fragment
spanning every junction, so terms bridging two junctions are counted once and the
ligand's charges are redistributed once. Two bonds may not share a ligand atom or
a protein atom. See `test/covalent_ligand/9r85/9r85_covalent_2link.txt`.

> **Attaching to a carbonyl/carboxyl carbon.** Force-field structures carry no
> bond orders, so the valence-based auto-detection cannot tell that a backbone
> `C` is already double-bonded to `O` and will not free a valence for you. ChemEM
> warns when it spots this (an attachment carbon with two terminal oxygens); set
> `covalent_delete_protein_atoms` explicitly, e.g. `['OXT']` at a C-terminus.

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
4. **Engine coverage.** `--dock`, `--refine` (incl. annealing),
   `--smart-ligand-refine2` and `--export` all consume the covalent link.
   `--dock2` does **not** — it warns and docks the ligand non-covalently, so
   use `--dock` for covalent ligands. `geodock` has no covalent support.
5. **Boron warheads.** OpenFF Sage 2.0 carries no boron valence parameters, so
   boronic-acid warheads (e.g. benchmark case 6WQH) fail fragment
   parameterization with `UnassignedBondError`.

---

## Notes for maintainers

- All protocol specs live in `ChemEM/protocol_spec.py` in the `REGISTRY`.
- Protocol selection flags are auto-generated from registry keys (underscore → hyphen).
- If you add a new protocol:
  1) Create a new `ProtocolSpec(...)` in `REGISTRY`.
  2) Provide `deps(args)` and (optionally) `add_args(parser)`.
