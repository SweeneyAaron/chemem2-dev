# Smart Ligand Refinement

`SmartLigandRefinement` is a near-fit ligand repair protocol for cases where a
ligand is already approximately placed in cryoEM density, but local regions such
as tails, linkers, sugars, phosphate groups, or flexible substituents need
controlled local repair.

The design rule is:

```text
ChemEM map scoring decides whether the ligand fits the density.
OpenMM/OpenFF decides whether the ligand is chemically reasonable.
The search controller proposes small, reversible, logged moves.
```

Density scoring is not moved into OpenMM.

## CLI Usage

From the project command line:

```bash
chemem my_config.txt --smart-ligand-refine --output ./out
```

Short alias:

```bash
chemem my_config.txt -slr --output ./out
```

Run one conservative macrocycle:

```bash
chemem my_config.txt -slr \
  --slr-max-macrocycles 1 \
  --slr-no-openmm-geometry \
  --output ./out
```

Enable the branch rebuild stage:

```bash
chemem my_config.txt -slr \
  --slr-branch-rebuild \
  --output ./out
```

Use the raw density map instead of the confidence-map preference:

```bash
chemem my_config.txt -slr \
  --slr-map-source raw \
  --output ./out
```

Print human-readable progress while it runs:

```bash
chemem my_config.txt -slr \
  --slr-progress \
  --output ./out
```

Write diagnostic JSON and compare against a deposited/reference SDF:

```bash
chemem my_config.txt -slr \
  --slr-branch-rebuild \
  --slr-progress \
  --slr-write-diagnostics \
  --slr-reference-sdf /path/to/Ligand_dep.sdf \
  --output ./out
```

## Python API

The public entry point is:

```python
from ChemEM.protocols.refine.smart_ligand_refine import (
    SmartLigandRefinementConfig,
    smart_ligand_refine,
)

config = SmartLigandRefinementConfig(
    max_macrocycles=3,
    output_dir="./smart_ligand_refine",
    debug=True,
)

refined_coords, report = smart_ligand_refine(
    protein=protein,
    ligand=ligand,
    ligand_coords=ligand_coords,
    em_map=em_map,
    openmm_system=openmm_system,
    openmm_context=openmm_context,
    half_maps=None,
    config=config,
)
```

Returns:

```python
{
    "initial_ligand_ccc": float,
    "final_ligand_ccc": float,
    "initial_low_tail_q": float,
    "final_low_tail_q": float,
    "accepted_moves": int,
    "rejected_moves": int,
    "final_openmm_energy": float,
    "move_log_path": str,
}
```

## Outputs

CLI outputs are written under:

```text
<output>/smart_ligand_refine/
```

Per ligand:

```text
Ligand_<i>/smart_refined.sdf
Ligand_<i>/smart_ligand_refine_log.json
```

Summary:

```text
summary.json
```

Each log entry records attempted moves, acceptance/rejection, map-score deltas,
anchor RMSD, OpenMM/geometry energy deltas, clash changes, and rejection reason.

## Main Stages

1. Evaluate external map metrics: Q-score, ligand CCC, local CCC, map value,
   map gradient, optional half-map agreement, optional difference density.
2. Classify atoms as `ANCHOR`, `REPAIR`, `WEAK_OR_ABSENT`, or `UNCERTAIN`.
3. Run rigid micro-jiggle candidates.
4. Optionally try connected bad-atom subregion rigid-body proposals.
5. Rank bad torsions by the low-Q repair-like atoms they control.
6. Try torsion-profile minima, preferring OpenMM/OpenFF torsion-force profiles
   and falling back to RDKit/MMFF or UFF profiles when needed.
7. Optionally run targeted worst-atom branch rebuild over torsion-profile
   minima. Intermediate branch states are ranked by the current target atom,
   but final accepted candidates still use the normal acceptance filters.
8. Optionally run short no-map geometry cleanup.
9. Accept only candidates that improve map fit without disrupting anchors or
   violating geometry filters.

## Flags

### General

```text
--slr-map-source {confidence,raw,difference}
    Map source policy. Default: confidence.

--slr-max-macrocycles INT
    Maximum smart ligand refinement macrocycles. Default: 3.

--slr-debug
    Print verbose refinement messages. Reports initial metrics, macrocycle
    summaries, atom-class counts, stage candidate counts, top candidate scores,
    accepted moves, and compact rejection summaries.
    Default: off.

--slr-progress
    Print concise progress messages without verbose per-candidate dumps.
    Default: off.

--slr-write-diagnostics
    Write smart_ligand_refine_diagnostics.json containing progress events,
    per-atom metrics/classes, torsion-profile minima, branch-walk states, stage
    summaries, and optional reference-SDF comparisons. Default: off.

--slr-reference-sdf PATH
    Reference SDF for diagnostics only. Reports atom-count/formal-charge
    mismatch, heavy-atom RMSD, centroid shift, and worst displaced heavy atoms.
    Does not affect scoring or acceptance.

--slr-debug-candidate-limit INT
    Maximum number of top-ranked candidates printed per stage when
    --slr-debug is enabled. Default: 5.

--slr-pocket-radius FLOAT
    Pocket radius for the local OpenMM geometry environment in Å. Default: 12.0.

--slr-pin-k FLOAT
    Protein pin strength in the no-map OpenMM geometry environment. Default: 5000.0.
```

### Geometry

```text
--slr-use-openmm-geometry
    Use a no-map OpenMM context for geometry evaluation when available. Default: on.

--slr-no-openmm-geometry
    Disable OpenMM geometry context creation and use RDKit/simple checks.

--slr-clean-candidates
    Run short no-map geometry cleanup on generated candidates before scoring.
    Default: off.

--slr-clean-each-macrocycle
    Run short no-map geometry cleanup after accepted macrocycle moves. Default: on.

--slr-no-clean-each-macrocycle
    Disable no-map geometry cleanup after macrocycles.

--slr-clash-distance-a FLOAT
    Soft protein-ligand clash distance threshold in Å. Default: 1.6.
```

### Search

```text
--slr-branch-rebuild
    Enable targeted worst-atom branch rebuild over torsion-profile minima.
    Default: off.

--slr-branch-beam-width INT
    Number of intermediate branch states retained at each beam depth.
    Default: 16.

--slr-max-branch-torsions INT
    Maximum number of torsions walked in a targeted branch. Default: 6.

--slr-no-ring-flips
    Disable exocyclic 180-degree ring-flip proposals during branch rebuild.
    Default: ring flips on.

--slr-torsion-profile-source {auto,openmm,openff,rdkit}
    Torsion profile source. auto prefers OpenMM/OpenFF torsion-force profiles
    and falls back to RDKit profiles. Default: auto.

--slr-write-accepted-sdf
    Reserved for writing accepted intermediate SDF snapshots. Default: off.
```

### Atom Classification

```text
--slr-anchor-q-min FLOAT
    Q-score threshold for protected anchor atoms. Default: 0.75.

--slr-anchor-local-ccc-min FLOAT
    Local CCC threshold for protected anchor atoms. Default: 0.60.

--slr-repair-q-max FLOAT
    Q-score threshold below which density-supported atoms are repair targets.
    Default: 0.55.

--slr-min-halfmap-agreement FLOAT
    Minimum half-map agreement for confident classification. Default: 0.50.

--slr-min-density-value FLOAT
    Minimum sampled density value for density support. Default: 1e-6.

--slr-target-q FLOAT
    Target Q-score used when ranking bad torsions. Default: 0.75.

--slr-min-torsion-badness FLOAT
    Minimum torsion badness required for torsion-minima repair. Default: 0.05.
```

## Notes

- This is a local near-fit refinement protocol, not global docking.
- `WEAK_OR_ABSENT` atoms are deliberately not forced aggressively into noise.
- OpenMM/OpenFF energy is used as a filter/penalty, not as a density-biased
  objective.
- Candidate scoring is based on external ChemEM map metrics plus geometry and
  clash penalties.
