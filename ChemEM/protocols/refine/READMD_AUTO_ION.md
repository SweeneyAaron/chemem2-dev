# Auto Ion Template Search + IonFixer Guide

This guide explains how to run the new `ion_template_search` protocol and how it interacts with `ion_fixer`.

## Protocol Selection

- `--ion-template-search` or `-its`
  Runs the template-mining protocol that searches PDB templates and proposes IonFixer atom specs.

- `--ion-fixer`
  Runs IonFixer.

When both are selected, `ion_template_search` runs first (deterministic order), then `ion_fixer`.

## Quick Start

1. Run template search only:

```bash
chemem <config_file> -its --output <out_dir>
```

2. Run template search, then IonFixer in the normal protocol pipeline:

```bash
chemem <config_file> -its --ion-fixer --output <out_dir>
```

3. Run template search and let it call IonFixer internally when confidence passes:

```bash
chemem <config_file> -its --its-auto-run-ion-fixer --output <out_dir>
```

Note: Avoid combining `--its-auto-run-ion-fixer` and `--ion-fixer` unless you intentionally want IonFixer to run twice.

## Output Files

`ion_template_search` writes to:

`<out_dir>/ion_template_search/`

Files:
- `report.json`
- `selected_template.json`
- `proposed_ion_fixer_args.json`

If confidence is below threshold, proposal files are still written, but IonFixer inputs are not auto-applied.

Auto-apply now requires:
- confidence at/above threshold
- at least one mapped protein coordinating atom
- at least one mapped ligand coordinating atom

## `ion_template_search` Flags

- `--its-auto-run-ion-fixer`
  Default: `False`
  If set, `ion_template_search` will execute IonFixer in the same run when confidence gating passes.

- `--its-confidence-thresh <float>`
  Default: `0.65`
  Minimum confidence required before auto-populating IonFixer options (`atom_specs`, and optionally `ion_type`/`coordination_geometry`).

- `--its-max-entry-candidates <int>`
  Default: `200`
  Maximum number of RCSB search hits retained before deeper evaluation.

- `--its-max-templates <int>`
  Default: `25`
  Maximum number of candidate templates evaluated in detail.

- `--its-seq-identity-min <float>`
  Default: `0.35`
  Minimum template-to-target chain sequence identity used for residue mapping.

- `--its-local-chain-radius-a <float>`
  Default: `12.0`
  Local-chain mode is mandatory: only target chains with residues within this ligand-neighborhood radius are eligible for sequence mapping.

- `--its-ion-elements <csv>`
  Default: empty string (uses built-in metals allowlist)
  Comma-separated metal allowlist override.
  Example: `--its-ion-elements ZN,MG,CA`

- `--its-similarity-enabled`
  Default: `True`
  Enables ligand similarity expansion after exact matching.

- `--its-no-similarity`
  Disables similarity search and keeps exact matching only.

### Mapping Behavior Notes

- Protein mapping uses both global sequence alignment and local-contact consistency checks.
- Ligand mapping is contact-anchored only (template contact atoms must be traceable); broad element fallback is disabled.
- Allowed ligand donor elements for mapping are `O/N/S`, but only if traceable from template contacts.

## `ion_fixer` Flags (Relevant Handoff Target)

- `--ion-type <str>`
- `--coordination-geometry <str>` (default: `Octahedral`)
- `--atom-spec <spec>` (repeatable)
- `--exclude-spec <spec>` (repeatable)
- `--pin-spec <spec>` (repeatable)
- `--distance-spec <atom1;atom2;distance_A>` (repeatable)
- `--ion-forcefield <xml>` (default: `amber14/tip3pfb.xml`)
- `--k_ang <float>`
- `--distance_fraction <float>` (default: `0.9`)
- `--n-cycles <int>` (default: `60`)

## Recommended Run Patterns

1. Safe default:
```bash
chemem <config_file> -its --ion-fixer
```
This lets confidence-gated option population happen before the regular IonFixer stage.

2. Diagnostics only:
```bash
chemem <config_file> -its
```
Use this when you want reports without modifying/running IonFixer.

3. Stricter template acceptance:
```bash
chemem <config_file> -its --its-confidence-thresh 0.8 --its-no-similarity
```

4. Broader search:
```bash
chemem <config_file> -its --its-max-entry-candidates 400 --its-max-templates 50
```

5. ATP/Mg-like systems (recommended):
```bash
chemem <config_file> -its --its-local-chain-radius-a 12.0 --its-seq-identity-min 0.4 --ion-fixer
```
For Mg/Ca, geometry inference tolerates under-coordination and reports missing sites (often water-completable) in the output metadata.
