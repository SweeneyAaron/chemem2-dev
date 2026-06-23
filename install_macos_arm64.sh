#!/usr/bin/env bash
#
# install_macos_arm64.sh — one-shot installer for the ChemEM2 backend on
# macOS (Apple Silicon), for use with the ChemEM-X ChimeraX plugin.
#
# What it does:
#   1. Locates a conda/mamba binary (robust against non-standard install paths).
#   2. Creates a conda env (default name: "chemem") from environment-dev.yml,
#      which bundles both the C++ build toolchain and the full runtime stack.
#   3. Ensures llvm-openmp (the macOS-critical OpenMP runtime) is present.
#   4. Compiles the C++/pybind11 extensions and installs the `chemem` CLI.
#   5. Smoke-tests the install and prints the executable path for the plugin.
#
# Usage:
#   ./install_macos_arm64.sh [ENV_NAME]
#
#   ENV_NAME            optional conda env name to create (default: chemem)
#   CONDA_BIN=/path     env var override pointing directly at conda/mamba
#                       e.g.  CONDA_BIN=/opt/conda/bin/mamba ./install_macos_arm64.sh
#
set -euo pipefail

ENV_NAME="${1:-chemem}"

say()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARN:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------------------
# 1. Locate a conda/mamba binary (handles non-standard install locations).
# ----------------------------------------------------------------------------
find_conda_bin() {
    # (a) explicit override
    if [[ -n "${CONDA_BIN:-}" ]]; then
        [[ -x "$CONDA_BIN" ]] && { printf '%s\n' "$CONDA_BIN"; return 0; }
        die "CONDA_BIN='$CONDA_BIN' is not an executable file."
    fi

    # (b) env vars set by `conda init` (prefer mamba)
    local cand
    for cand in "${MAMBA_EXE:-}" "${CONDA_EXE:-}"; do
        [[ -n "$cand" && -x "$cand" ]] && { printf '%s\n' "$cand"; return 0; }
    done

    # (c) the currently-activated env's bin
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        for cand in "$CONDA_PREFIX/bin/mamba" "$CONDA_PREFIX/bin/conda"; do
            [[ -x "$cand" ]] && { printf '%s\n' "$cand"; return 0; }
        done
    fi

    # (d) PATH
    local name p
    for name in mamba conda; do
        p="$(command -v "$name" 2>/dev/null || true)"
        [[ -n "$p" ]] && { printf '%s\n' "$p"; return 0; }
    done

    # (e) probe common + non-standard roots the plugin does NOT auto-discover
    local root sub
    local roots=(
        "/opt/homebrew/Caskroom/miniforge/base"
        "/opt/conda"
        "/usr/local/miniconda3"
        "/usr/local/miniforge3"
        "$HOME/miniforge3"
        "$HOME/miniconda3"
        "$HOME/anaconda3"
        "$HOME/mambaforge"
        "$HOME/opt/miniforge3"
        "$HOME/opt/miniconda3"
        "$HOME/opt/anaconda3"
    )
    for root in "${roots[@]}"; do
        for sub in condabin/mamba bin/mamba condabin/conda bin/conda; do
            [[ -x "$root/$sub" ]] && { printf '%s\n' "$root/$sub"; return 0; }
        done
    done

    return 1
}

CONDA="$(find_conda_bin || true)"
if [[ -z "$CONDA" ]]; then
    die "Could not find conda or mamba.

Fix one of the following, then re-run:
  • Install Miniforge:  https://github.com/conda-forge/miniforge
  • Activate conda first: source <your-conda>/etc/profile.d/conda.sh
    (or run 'conda init zsh' and open a new terminal)
  • Point this script straight at the binary:
        CONDA_BIN=/full/path/to/mamba ./install_macos_arm64.sh"
fi
say "Using conda/mamba: $CONDA"

# ----------------------------------------------------------------------------
# 2. Sanity checks: repo root + Xcode Command Line Tools.
# ----------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
for f in meta.yaml CMakeLists.txt environment-dev.yml; do
    [[ -f "$f" ]] || die "Missing '$f'. Run this script from the chemem2-dev repo root."
done

if ! xcode-select -p >/dev/null 2>&1; then
    warn "Xcode Command Line Tools not detected. The C++ build needs the macOS SDK.
       Install them with:  xcode-select --install
       (then re-run this script)."
fi

# ----------------------------------------------------------------------------
# 3. Create the conda env (idempotent: skip if it already exists).
# ----------------------------------------------------------------------------
CONDA_BASE="$("$CONDA" info --base 2>/dev/null || true)"
ENV_PREFIX=""
[[ -n "$CONDA_BASE" ]] && ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"

if [[ -n "$ENV_PREFIX" && -d "$ENV_PREFIX" ]]; then
    say "Conda env '$ENV_NAME' already exists at $ENV_PREFIX — skipping creation."
else
    say "Creating conda env '$ENV_NAME' from environment-dev.yml."
    echo "    (downloads OpenMM/RDKit/AmberTools/OpenFF — expect ~10-20 min and several GB)"
    "$CONDA" env create -n "$ENV_NAME" -f environment-dev.yml
fi

# ----------------------------------------------------------------------------
# 4. macOS OpenMP runtime (usually pulled in by openmm; ensure it explicitly).
# ----------------------------------------------------------------------------
say "Ensuring llvm-openmp is installed (macOS OpenMP runtime)."
"$CONDA" install -n "$ENV_NAME" -y -c conda-forge llvm-openmp

# ----------------------------------------------------------------------------
# 5. Compile the C++/pybind11 extensions and install the `chemem` CLI.
#    --no-build-isolation uses the conda toolchain already in the env, so the
#    build links against conda's clang/openmp/boost (important on arm64).
# ----------------------------------------------------------------------------
say "Building C++ extensions and installing ChemEM into '$ENV_NAME'."
if ! "$CONDA" run -n "$ENV_NAME" pip install . --no-build-isolation; then
    warn "Build with --no-build-isolation failed; retrying with build isolation."
    "$CONDA" run -n "$ENV_NAME" pip install .
fi

# ----------------------------------------------------------------------------
# 6. Smoke test + report the executable path.
# ----------------------------------------------------------------------------
say "Verifying the install."
"$CONDA" run -n "$ENV_NAME" python -c \
    "import openmm; from ChemEM.config import Config; print('runtime imports ok')"

if "$CONDA" run -n "$ENV_NAME" chemem --help >/dev/null 2>&1; then
    echo "    chemem CLI OK"
else
    die "'chemem --help' failed inside env '$ENV_NAME'. See output above."
fi

CHEMEM_PATH="$("$CONDA" run -n "$ENV_NAME" python -c \
    "import shutil; print(shutil.which('chemem') or '')")"

printf '\n\033[1;32mDone.\033[0m ChemEM backend installed into conda env %s.\n' "'$ENV_NAME'"
cat <<EOF

  Backend executable:
      ${CHEMEM_PATH:-<not found on PATH inside env>}

Next steps for the ChimeraX plugin:
  • Easiest:  activate this env, THEN launch ChimeraX, so the plugin finds it:
        conda activate $ENV_NAME && open -na ChimeraX
  • Otherwise, paste the path above into the plugin's backend selector.

See COLLABORATOR_SETUP.md for the full plugin install + wiring steps.
EOF
