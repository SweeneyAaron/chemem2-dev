# Running ChemEM2 on an HPC with Apptainer

This directory builds a single self-contained Apptainer/Singularity image
(`.sif`) so you can run ChemEM2 on a cluster **without** creating a conda
environment. Everything — the conda-forge runtime stack (OpenMM, RDKit,
AmberTools, OpenFF, ...) and the compiled C++/pybind11 extensions — is baked
into the image.

This image is **CPU-only**. OpenMM uses the CPU platform.

Files:
- [`chemem.def`](chemem.def) — the Apptainer definition (the build recipe).
- [`environment-container.yml`](environment-container.yml) — the conda-forge env
  spec (build toolchain + runtime deps) installed inside the image.

---

## 1. Build the image

> Apptainer does **not** run on macOS, so you can't build the `.sif` on your
> Mac. Build it on a Linux host that has Apptainer — easiest is your **HPC
> login/build node** (it builds in place, so there's no multi-GB `scp`).

From the **repo root** (the folder containing `meta.yaml` and `CMakeLists.txt`):

```bash
apptainer build --fakeroot chemem.sif container/chemem.def
```

- `--fakeroot` lets you build unprivileged; most clusters enable it. If yours
  doesn't, use Apptainer's remote builder instead:
  ```bash
  apptainer build --remote chemem.sif container/chemem.def
  ```
- The build compiles the C++ extensions and resolves the full conda env, so it
  takes a while (often 15–40 min) and needs a few GB of scratch. If your build
  node restricts `/tmp`, point Apptainer at scratch:
  ```bash
  export APPTAINER_TMPDIR=$SCRATCH/apptainer-tmp
  export APPTAINER_CACHEDIR=$SCRATCH/apptainer-cache
  mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
  ```

The build runs a `%test` section at the end (`chemem --help` + a runtime import
check); a clean build means those passed.

---

## 2. Run it

Apptainer automatically bind-mounts `$HOME`, the current directory, and `/tmp`.
Run **from your data directory** and pass `--pwd "$PWD"` so the config's relative
paths (protein, ligand, map, output) resolve the same way they do natively:

```bash
cd /path/to/my/job          # contains conf.txt + inputs
apptainer run --pwd "$PWD" /path/to/chemem.sif conf.txt --dock
```

`apptainer run` invokes the `chemem` entry point, so everything after the `.sif`
is passed straight to the CLI. `--help` works too:

```bash
apptainer run /path/to/chemem.sif --help
```

If your inputs/outputs live on a project or scratch filesystem that Apptainer
doesn't bind by default, add it explicitly:

```bash
apptainer run --bind /scratch/myproj:/scratch/myproj \
    --pwd /scratch/myproj/run1 \
    /path/to/chemem.sif conf.txt --refine
```

You can also use `apptainer exec` to run anything inside the image, e.g. a quick
sanity check:

```bash
apptainer exec /path/to/chemem.sif \
    python -c "import openmm; from ChemEM.config import Config; print('ok')"
```

---

## 3. CPUs / threads under a scheduler

ChemEM derives its default CPU budget from `os.cpu_count() - 2`
(`default_cpu_budget()` in `ChemEM/tools/resources.py`). Inside a Slurm/PBS
allocation that often reports the **whole node**, not the cores you were
granted, which oversubscribes the job. **Always pass `--ncpu` explicitly** to
match your allocation:

```bash
apptainer run --pwd "$PWD" /path/to/chemem.sif conf.txt --dock \
    --ncpu "$SLURM_CPUS_PER_TASK"
```

ChemEM sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc. internally from `--ncpu`.
For split-site docking you can also tune `--cpus-per-site`.

### Sample Slurm job script

```bash
#!/bin/bash
#SBATCH --job-name=chemem
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00

module load apptainer        # or: module load singularity

SIF=/path/to/chemem.sif
cd "$SLURM_SUBMIT_DIR"        # job dir with conf.txt + inputs

apptainer run --pwd "$PWD" "$SIF" conf.txt --dock --minimize-docking \
    --ncpu "$SLURM_CPUS_PER_TASK"
```

---

## 4. Rebuilding after code changes

The image bakes in the source at build time, so rebuild the `.sif` whenever you
update ChemEM:

```bash
apptainer build --fakeroot chemem.sif container/chemem.def
```

To change pinned dependency versions, edit
[`environment-container.yml`](environment-container.yml) and rebuild.

---

## Notes

- **GPU:** this image is CPU-only by design. A CUDA variant would only need a
  CUDA-enabled `openmm` build in the env file plus `apptainer run --nv ...`; the
  rest of the recipe is unchanged.
- **Reproducibility:** the env is resolved fresh each build. For
  bit-reproducible rebuilds, generate a lock (e.g. `conda-lock` /
  `micromamba env export --explicit`) and install from that instead of the
  `.yml`.
- **Image size:** the build toolchain (compilers, headers) stays in the image
  for simplicity. If size matters, split into a throwaway build stage that
  keeps only the runtime env.
