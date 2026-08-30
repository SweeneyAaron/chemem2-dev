# ChemEM-X collaborator setup (macOS Apple Silicon)

This guide gets you from zero to a working **ChemEM-X ChimeraX plugin** on an
Apple-Silicon Mac. It covers the backend, the plugin, and how the two connect.
No GitHub account or `git` is required — you download the code as ZIP files.

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

## 1. Download the code

You don't need a GitHub account — download each branch as a ZIP from the browser
(both repos are public). Click each link, then unzip:

- **Backend** (`chemem2-dev`, branch `feature/mapq_score`):
  https://github.com/SweeneyAaron/chemem2-dev/archive/refs/heads/feature/mapq_score.zip
- **Plugin** (`ChemEM-X`, branch `folder-branch`):
  https://github.com/SweeneyAaron/ChemEM-X/archive/refs/heads/folder-branch.zip

(Or, on each repo page: pick the branch in the dropdown, then **Code → Download
ZIP**.) Unzipping gives two folders — the examples below assume they're in
`~/Downloads`:

- `chemem2-dev-feature-mapq_score/` — the backend
- `ChemEM-X-folder-branch/` — the plugin (the bundle is inside `ChemEM-X_v2/`)

## 2. Install the backend

Open **Terminal**, go into the backend folder, and run the installer:

```bash
cd ~/Downloads/chemem2-dev-feature-mapq_score
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

## 3. Install the ChimeraX plugin

The plugin is the `ChemEM-X_v2` folder inside the unzipped `ChemEM-X-folder-branch`
folder (it contains `bundle_info.xml`). From **inside ChimeraX** run:

```
devel install ~/Downloads/ChemEM-X-folder-branch/ChemEM-X_v2
```

(Adjust the path to wherever you unzipped it.) This builds and installs the
bundle and pulls its dependency (`scikit-spatial==7.0`) into ChimeraX's Python.
Restart ChimeraX, then open the tool from **Tools → Structure Prediction → ChemEM**.

## 4. Connect the plugin to the backend

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

## 5. Smoke test end-to-end

In ChimeraX with the ChemEM tool open:
1. Load a small protein and a ligand.
2. Define a binding site / centroid and run a **Dock**.
3. Watch the **Jobs** tab — you should see the backend's output stream in. That
   confirms the plugin is invoking `chemem` successfully.

## Troubleshooting

**The plugin can't find the backend.** See step 4 — launch ChimeraX from the
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
