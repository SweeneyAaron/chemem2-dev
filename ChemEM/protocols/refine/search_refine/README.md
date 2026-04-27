# How to run `search_refine`

**Entrypoint:** `chemem <config> -sr [options...]` (or `python -m ChemEM`). The
`-sr` short alias selects the protocol; all `--sr-*` flags tune it.

## 1. Default run (v2 stage, SCI scorer, greedy acceptance, single-pose output)

```bash
chemem my.conf -sr --output ./out
```

Defaults out of the box:

- `--sr-stage v2` — new diagnostic-driven refinement stage.
- `--sr-return-n 1` — emit only the best refined pose per ligand.
- CCC scorer uses an **analytical per-atom gradient** (~50× faster than FD).
  SCI/MI/Q-score still fall back to central FD.
- All the new targeted-proposal features (diagnostic, dihedral, sub-region,
  directed kick) are **off by default** — opt in explicitly below. Without any
  extra flags, v2 is functionally equivalent to pre-refactor search_refine
  plus the CCC gradient speedup and single-pose output.

## 2. Stage preset: v2 vs legacy

```bash
chemem my.conf -sr --sr-stage v2        # default
chemem my.conf -sr --sr-stage legacy    # pre-refactor bit-exact path
```

`legacy` forces:
- CCC scorer routes through central FD (not the analytical gradient).
- Diagnostic, dihedral proposals, sub-region proposals, directed kicks all
  ignored regardless of their individual flags.

Use `legacy` for A/B regression against pre-refactor behavior.

## 3. Pick a different scorer

```bash
chemem my.conf -sr --sr-scorer ccc         # truncated CC
chemem my.conf -sr --sr-scorer mi          # mutual information
chemem my.conf -sr --sr-scorer qscore      # Q-score
```

Each metric drives **both** direction and ranking. Metric-specific knobs only
fire for their scorer:

- CCC: `--sr-ccc-mask-mode {nonzero,full}`
- MI: `--sr-mi-fd-step-a 0.1 --sr-mi-nbins 64 [--sr-mi-normalized]`
- Q-score: `--sr-qscore-sigma-ref 0.6 [--sr-qscore-radii 0.1,0.5,1.0,...]`
- SCI: `--sr-sci-sigma`, `--sr-w0/1/2`, `--sr-use-amp-eq`, etc.

## 4. Pick an acceptance strategy

```bash
# Greedy (default) — strict improvement wins
chemem my.conf -sr --sr-accept-strategy greedy

# Simulated annealing — occasionally accept worse proposals
chemem my.conf -sr --sr-accept-strategy metropolis \
    --sr-accept-temp-start 0.05 --sr-accept-temp-end 0.005

# Basin hopping — after N consecutive rejects, apply a kick
chemem my.conf -sr --sr-accept-strategy basin_hopping \
    --sr-basin-hop-stale 3 --sr-basin-hop-sigma-a 0.3
```

See section 8 for swapping the Gaussian kick for a **directed** one.

## 5. Gradient step size (FD-based scorers)

Central FD (default) is robust but costs `2·3·N_heavy` score evals per iter.
The CCC scorer in `--sr-stage v2` bypasses this with an analytical gradient
and does not consume `sr_fd_step_a`; the knob still applies to SCI, MI, and
Q-score, and to CCC when `--sr-stage legacy` is set.

```bash
--sr-fd-mode forward     # ~half the evals, noisier
--sr-fd-step-a 0.1       # larger step — smoother on discontinuous metrics
```

MI has its own larger default (`--sr-mi-fd-step-a 0.1`) since histogram MI is
noisy.

## 6. Diagnostic: identify bad atoms per iter

```bash
--sr-diagnostic                        # compute + log (requires --sr-verbose)
--sr-q-good-thresh 0.7                 # atoms with Q ≥ this are 'good'
--sr-q-bad-thresh  0.3                 # atoms with Q ≤ this are 'bad'
--sr-qscore-sigma-ref 0.6              # reference Gaussian width for Q-score
```

Logs a one-line per-atom summary (`q(mean=…, min=…) g_max=… good=… bad=…
neutral=…`) every iter. Useful to see whether any atoms are mispositioned and
how the refinement is shaping their Q distribution over time.

## 7. Target only bad atoms with the Cartesian pull

```bash
--sr-target-bad-only
```

Restricts the gradient-driven tether displacement to atoms classified as
"bad" by the diagnostic (see section 6). Good atoms are pinned at their
current position. Falls back to all-atom behavior when no atom is below the
bad threshold this iter. Implies the diagnostic is computed every iter.

## 8. Dihedral rearrangement proposals

```bash
--sr-dihedral-proposals-per-iter 2
```

Per iter, replace up to N gradient proposals with **rotation of a rotatable
bond** chosen to move bad atoms toward their CCC gradient direction. For each
bad atom the best-aligned rotatable bond is picked analytically
(`|axis × (x − pivot) · target| / v_norm`); angle steps are `±30°, ±60°, ±90°`,
with sign chosen to align with the target direction.

Heavy-atom rotatable bonds are enumerated directly from the bond graph
(`single, non-ring, non-aromatic, both endpoints have ≥2 heavy neighbors`);
explicit hydrogens on the input mol are handled correctly.

## 9. Sub-region rigid-body proposals

```bash
--sr-subregion-proposals-per-iter 1
--sr-subregion-min-size 3              # smallest cluster worth moving
--sr-subregion-max-size 8              # keep moves local
```

Clusters bad atoms by heavy-atom bond-graph connectivity, fits a closed-form
Kabsch transform that nudges the cluster along its per-atom gradient targets
(ε = `--sr-max-atom-delta-a`), and proposes the rotated/translated positions.
Ranked by total cluster badness; one proposal per cluster.

## 10. Directed basin-hopping kick

```bash
--sr-accept-strategy basin_hopping --sr-directed-kick
--sr-directed-kick-angle-deg 90        # default
```

Instead of a uniform Gaussian kick across all ligand atoms, apply a dihedral
rotation of the **worst-Q atom's** best-aligned rotatable bond. Falls back
automatically to the Gaussian kick when no viable dihedral exists (no
rotatable bonds, zero gradient, atom on every axis). The per-iter log line
shows which path fired (`directed bond=… q=… dtheta=…` vs `gaussian sigma=…`).

## 11. Output selection

```bash
--sr-return-n 1                        # default — single best pose
--sr-return-n 3                        # up to 3 near-tie distinct poses
--sr-return-score-margin 0.01          # max score gap from the best
--sr-rmsd-dedupe 0.5                   # min RMSD between kept poses
```

When `--sr-return-n > 1`, extra poses are emitted **only** if they are within
the score margin of the best **AND** separated by more than `--sr-rmsd-dedupe`.
Use this when you believe there are genuinely distinct equal-quality
solutions; otherwise stay at 1.

## 12. Outputs

Written under `<output>/search_refine/`:

- `Ligand_<i>/pose_001.sdf` … refined ligand poses (1 by default, see §11)
- `Ligand_<i>/scores.json` — every explored state with `scorer`, `score`,
  `final_score`, per-metric terms
- `summary.json` — per-ligand best + acceptance counters (`n_accept_major`,
  `n_accept_micro`, `n_perturb`)
- `search_refine_receptor.pdb` + `Ligand_<i>_best.sdf`

## 13. Debugging

```bash
--sr-verbose --sr-log-every 1
# Optional deep trace for NaN/instability debugging:
--sr-debug-relax
# Per-iter diagnostic (requires --sr-verbose):
--sr-diagnostic
```

Per-iteration lines log `grad(mean=…, max=…)` — that's how you confirm the
metric gradient is non-trivial. With `--sr-diagnostic`, you also get the
per-atom Q breakdown (§6). With `--sr-debug-relax`, each proposal prints
stage snapshots (`start`, `after-pre-min`, `after-md`, `after-post-min`) with
energy, ligand force norms, coordinate finiteness, and ligand displacement
from accepted/target states.

## 14. Recommended "hard case" preset

For a docked pose that is roughly right but has e.g. one bad dihedral or one
atom pulled out of density, enable the full v2 toolkit:

```bash
chemem my.conf -sr --sr-scorer ccc \
    --sr-accept-strategy basin_hopping \
    --sr-diagnostic \
    --sr-target-bad-only \
    --sr-dihedral-proposals-per-iter 2 \
    --sr-subregion-proposals-per-iter 1 \
    --sr-directed-kick \
    --sr-verbose
```

For a well-docked pose that only needs polishing, the defaults plus
`--sr-scorer ccc` and `--sr-verbose` are usually enough.

## 15. Sanity-check before a long run

```bash
python -m unittest discover ChemEM/tests -v
```

Should be 52 passing tests in ~0.5 s. The individual refactor suites are:
`test_search_refine_scorers`, `test_search_refine_acceptance`,
`test_search_refine_diagnostic`, `test_search_refine_direction`,
`test_search_refine_dihedral`, `test_search_refine_subregion`,
`test_search_refine_selection`.

## 16. Full options reference

Every `--sr-*` flag grouped by purpose, with the default in parentheses. Flag
definitions and defaults live in [../../protocol_spec.py](../../protocol_spec.py)
(lines 213–349).

### Outer loop / iteration control
- `--sr-max-outer-iter 50` — cap on outer iterations per ligand
- `--sr-patience 4` — early stop after this many consecutive stale iters
- `--sr-proposals-per-iter 6` — candidates evaluated per iteration
- `--sr-seed 1` — RNG seed

### Proposal geometry (how pulls are constructed)
- `--sr-max-atom-delta-a 1.5` — per-atom displacement cap (Å)
- `--sr-trust-k 250.0` — harmonic tether strength (kcal/mol/Å²)
- `--sr-pin-k 5000.0` — protein-atom restraint during relaxation

### Gradient estimation (FD-based scorers)
- `--sr-fd-step-a 0.25`, `--sr-fd-mode {central,forward}` — consumed by SCI /
  MI / Q-score, and by CCC only under `--sr-stage legacy`

### Map / pocket
- `--sr-map-source {confidence,raw,difference}` (default `confidence`)
- `--sr-pocket-radius 12.0` — pocket cutoff around ligand (Å)
- `--sr-map-pad-a 3.0` — sub-map padding (Å)
- `--sr-global-k 0.0` — OpenMM map-potential weight during MD (0 = off)

### MD / minimisation
- `--sr-md-steps-per-iter 250`
- `--sr-minimise-max-iters 200`
- `--sr-md-temp-k 150.0`

### Scorer-specific (only consumed by matching `--sr-scorer`)
- CCC: `--sr-ccc-mask-mode {nonzero,full}`
- MI: `--sr-mi-fd-step-a 0.1`, `--sr-mi-nbins 20`, `--sr-mi-normalized`
- Q-score: `--sr-qscore-sigma-ref 0.6`, `--sr-qscore-radii 0.1,0.5,1.0,...`
- SCI: `--sr-sci-sigma 1.0`, `--sr-sigma-coeff 0.356`, `--sr-sci-eps 1e-8`,
  `--sr-w0 1.0`, `--sr-w1 1.0`, `--sr-w2 1.0`,
  `--sr-use-amp-eq / --sr-no-amp-eq`,
  `--sr-normalise-sim-map / --sr-no-normalise-sim-map`

### Acceptance-specific
- Greedy: `--sr-min-delta 1e-6` (threshold between "major" and "micro"
  acceptances)
- Metropolis: `--sr-accept-temp-start 0.05`, `--sr-accept-temp-end 0.005`
- Basin hopping: `--sr-basin-hop-stale 3`, `--sr-basin-hop-sigma-a 0.3`

### Diagnostic + targeted proposals (all off by default)
- `--sr-diagnostic`, `--sr-q-good-thresh 0.7`, `--sr-q-bad-thresh 0.3`
- `--sr-target-bad-only`
- `--sr-dihedral-proposals-per-iter 0`
- `--sr-subregion-proposals-per-iter 0`, `--sr-subregion-min-size 3`,
  `--sr-subregion-max-size 8`
- `--sr-directed-kick`, `--sr-directed-kick-angle-deg 90.0`

### Hybrid reranking (global, after the outer loop)
`final_score = score − w_energy · zscore(energy) − w_clash · zscore(clash)`
- `--sr-w-energy 0.0`, `--sr-w-clash 0.0` (both off by default)
- `--sr-clash-distance-a 1.6`

### Output selection
- `--sr-return-n 1`, `--sr-return-score-margin 0.01`, `--sr-rmsd-dedupe 0.5`

### Debugging
- `--sr-verbose`, `--sr-log-every 1`, `--sr-debug-relax`

## Package layout

```
search_refine/
  __init__.py          # re-exports SearchRefine (backward compat)
  orchestrator.py      # outer loop + stage dispatch + kick routing
  types.py             # RefinedPose, ProposalRecord
  diagnostic.py        # per-atom fit quality, rotatable bonds, clustering,
                       # Kabsch fit, directed kick decision
  direction.py         # build_targets_from_{gradient,dihedral,subregion}
  acceptance.py        # GreedyAccept / MetropolisAccept / BasinHoppingAccept
  io.py                # pose SDFs, scores.json, summary.json
  scorers/
    base.py            # BaseScorer (default central-FD gradient)
    sci.py mi.py qscore.py
    ccc.py             # analytical per-atom gradient override (v2 path)
```
