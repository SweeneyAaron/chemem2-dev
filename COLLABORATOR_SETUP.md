# ChemEM-X collaborator setup (macOS Apple Silicon)

This guide gets a collaborator from zero to a working **ChemEM-X ChimeraX
plugin** on an Apple-Silicon Mac. It covers the backend, the plugin, and how the
two connect.

## How the pieces fit together

The ChimeraX **plugin** is only a UI. To do real work it runs the **ChemEM
backend** — the `chemem` command-line program — as a local subprocess
(e.g. `chemem ChemEMChimera.conf --dock`). There is **no server, no network, and
no ports**: everything happens on your own machine.

The plugin finds the `chemem` program by scanning your conda environments. So
the whole job is: **install the backend into a conda env**, then point the plugin
at it (usually automatic).

## Prerequisites

- **Miniforge** (provides `conda`/`mamba`) — https://github.com/conda-forge/miniforge
- **Xcode Command Line Tools** — needed to compile the C++ extensions:
  ```bash
  xcode-select --install
  ```
- **ChimeraX ≥ 1.1** — https://www.cgl.ucsf.edu/chimerax/download.html

## 1. Install the backend

Clone the repo and run the installer from its root:

```bash
git clone https://github.com/SweeneyAaron/chemem2-dev.git
cd chemem2-dev
./install_macos_arm64.sh
```

The script:
1. finds your conda/mamba (even in non-standard locations),
2. creates a conda env named **`chemem`** from `environment-dev.yml` (the full
   build + runtime stack — expect ~10–20 min and several GB the first time),
3. compiles the C++/pybind11 extensions and installs the `chemem` CLI,
4. smoke-tests it and prints the executable path.

> Conda installed somewhere unusual (Homebrew, `/opt`, custom dir)? Either
> activate it first (`source <conda>/etc/profile.d/conda.sh`) or pass the binary
> directly: `CONDA_BIN=/full/path/to/mamba ./install_macos_arm64.sh`.

If you prefer to do it by hand, the three steps the script runs are:

```bash
mamba env create -n chemem -f environment-dev.yml
mamba install  -n chemem -c conda-forge llvm-openmp
mamba run      -n chemem pip install . --no-build-isolation
```

Verify:

```bash
mamba run -n chemem chemem --help
```

## 2. Install the ChimeraX plugin

Get the plugin source (the `ChemEM-X_v2` folder containing `bundle_info.xml`),
then from **inside ChimeraX** run:

```
devel install /full/path/to/ChemEM-X_v2
```

This builds and installs the bundle and pulls its dependency
(`scikit-spatial==7.0`) into ChimeraX's Python. Restart ChimeraX, then open the
tool from **Tools → Structure Prediction → ChemEM**.

## 3. Connect the plugin to the backend

The plugin auto-discovers the `chemem` env in most cases. To make it reliable
regardless of where conda lives, **launch ChimeraX from the activated env**:

```bash
conda activate chemem
open -na ChimeraX
```

That sets `CONDA_PREFIX`, which the plugin reads directly — so it finds the
backend no matter where conda is installed.

If the backend still isn't listed, select it manually in the plugin's backend
selector: paste the executable path that the installer printed, e.g.
`…/envs/chemem/bin/chemem`.

## 4. Smoke test end-to-end

In ChimeraX with the ChemEM tool open:
1. Load a small protein and a ligand.
2. Define a binding site / centroid and run a **Dock**.
3. Watch the **Jobs** tab — you should see the backend's output stream in. That
   confirms the plugin is invoking `chemem` successfully.

## Troubleshooting

**The plugin can't find the backend.** See step 3 — launch ChimeraX from the
activated env, or paste the `…/envs/chemem/bin/chemem` path into the backend
selector. The plugin only auto-scans `CONDA_PREFIX`, conda/mamba on `PATH`, and a
few default home-directory conda roots, so a Homebrew/`/opt`/custom install needs
one of those two steps.

**`zsh: killed chemem …` on macOS.** Some fresh conda installs ship native
libraries with invalid code signatures, which macOS kills on import. Re-sign the
env's libraries:

```bash
conda activate chemem
find "$CONDA_PREFIX" -type f \( -name "*.dylib" -o -name "*.so" \) \
    -exec codesign --force --sign - {} \;
```

**Build fails.** Make sure Xcode Command Line Tools are installed
(`xcode-select --install`). The installer automatically retries the build with
pip build isolation if the first attempt fails.
