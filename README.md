## Local conda build + install (from this repo)

These instructions build a local conda package from this source tree (including the compiled C++/pybind11 extensions), then install it into a fresh environment **with all runtime dependencies**.
> Installation requires mamba, this is a development version, please report bugs to aaron sweeney ucbtasw@ucl.ac.uk
> Important: do **not** install the `.conda` artifact by filename (e.g. `mamba install ./conda-dist/.../chemem-*.conda`).
> Installing via an explicit package file path does not reliably resolve dependencies.  
> Instead, build an indexed local channel and install **by package name**.

### 1) Create a build environment

```bash
cd /path/to/ChemEM2_build

mamba create -n chemem-build -c conda-forge python=3.11 conda-build conda-index
conda activate chemem-build
conda build . -c conda-forge --output-folder conda-dist


#if conda-dist/noarch doesn't exist build it with:
	mkdir -p conda-dist/noarch

#index conda-dist as a local package
python -m conda_index "$PWD/conda-dist"

#check this file exists
ls -l conda-dist/linux-64/repodata.json

#check the local pacakge is visalbe to conda
conda search -c "file://$PWD/conda-dist" chemem --info

#you should see something like: chemem 2.0.0 ... url: file://.../conda-dist/...


mamba create -n chemem -c conda-forge python=3.11
mamba activate chemem

#remove the build environment 
conda remove chemem-build --all
```

### create a runtime environment 
```bash
mamba create -n chemem -c conda-forge python=3.11
mamba activate chemem
mamba install \
  --override-channels \
  --strict-channel-priority \
  -c "file://$PWD/conda-dist" \
  -c conda-forge \
  "chemem=2.0.0"
#verify package install
chemem -h

#end-to-end test (60 secs)
chemem 7jjo_conf.txt --dock --no-map --minimize-docking --rescore

```

# ChemEM Docking CLI — Protocols & Options

This document describes the **protocol pipeline** and the **command-line options** exposed by the `chemem` entrypoint.

---

## Quickstart



Run only binding-site preparation:

```bash
chemem <conf_file> --binding-site
```

Run docking explicitly:

```bash
chemem <conf_file> --dock
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



## Notes for maintainers

- All protocol specs live in `ChemEM/protocol_spec.py` in the `REGISTRY`.
- Protocol selection flags are auto-generated from registry keys (underscore → hyphen).
- If you add a new protocol:
  1) Create a new `ProtocolSpec(...)` in `REGISTRY`.
  2) Provide `deps(args)` and (optionally) `add_args(parser)`.

