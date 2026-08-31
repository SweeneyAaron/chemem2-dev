#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""CLI plumbing for the ``--score`` protocol.

Imported by ``ChemEM.protocol_spec`` while the parser is being built, so it must
stay stdlib-only -- no rdkit, no OpenMM, no compiled extensions. The scorer classes
themselves are reached through the lazy ``SCORER_REGISTRY``, which only imports the
scorers actually selected.

Three jobs:
  * ``resolve_scorers``  -- ``--score-with`` -> an ordered, deduplicated tuple
  * ``score_deps``       -- that tuple -> the protocols that must run first
  * ``apply_score_back_compat`` -- rewrite the deprecated ``--rescore-poses`` /
    ``--mapq-score`` spellings into ``--score`` before protocol selection happens
"""

from __future__ import annotations

import argparse

from ChemEM.protocols.score.scorers import SCORER_NAMES, load_scorer_cls

# The full segmentation chain, i.e. dock's dependencies minus dock. Anything that
# needs the site precompute or the segmented site maps needs all three.
SEGMENTATION = ("binding_site", "alpha_mask", "confidence_map")

# What bare `--score` means. `--score` replaces `--rescore-poses`, whose whole job
# was ECHO, and ECHO is the number `--dock` ranks by -- so "score these poses" with
# nothing further said is "score them the way docking would have".
DEFAULT_SCORERS = ("echo",)


def resolve_scorers(args) -> tuple[str, ...]:
    """``--score-with`` -> ordered, deduplicated scorer names.

    Pure and side-effect free: called once from ``score_deps`` during dependency
    resolution and again from the protocol itself. Accepts a comma-separated list,
    a repeated flag, or both, plus the literal ``all``.
    """
    raw = getattr(args, "score_with", None)
    if not raw:
        return DEFAULT_SCORERS
    if isinstance(raw, str):          # hand-built namespaces pass a bare string
        raw = [raw]

    out: list[str] = []
    for chunk in raw:
        for name in str(chunk).split(","):
            name = name.strip().lower()
            if not name:
                continue
            if name == "all":
                out.extend(n for n in SCORER_NAMES if n not in out)
            elif name in SCORER_NAMES:
                if name not in out:
                    out.append(name)
            else:
                raise SystemExit(
                    f"--score-with: unknown scorer {name!r} "
                    f"(choose from: {', '.join(SCORER_NAMES)}, all)"
                )
    return tuple(out) or DEFAULT_SCORERS


def score_deps(args) -> tuple[str, ...]:
    """Protocols that must run before ``--score``, given the selected scorers.

    This is the point of the whole exercise: `--score --score-with qscore` returns
    ``()`` and skips segmentation entirely, where the old `--rescore-poses` always
    paid for it.
    """
    deps: list[str] = []
    for name in resolve_scorers(args):
        for dep in load_scorer_cls(name).deps_for(args):
            if dep not in deps:
                deps.append(dep)
    return tuple(deps)


# --------------------------------------------------------------- deprecations

# Old protocol flag -> (scorer it selects, extra option defaults it implies).
_ALIASES = (
    ("score_alias_rescore_poses", "--rescore-poses", "echo", {"score_sdf": True}),
    ("score_alias_mapq_score", "--mapq-score", "qscore", {}),
)


def apply_score_back_compat(args) -> None:
    """Rewrite the deprecated protocol flags into ``--score``.

    Must run before ``selected_protocols()``, because ``--score``'s own
    dependencies are computed from ``args.score_with``.

    Idempotent: the alias flags are cleared once consumed, so calling this twice
    (or on a namespace that never saw them) does nothing.
    """
    picked: list[str] = []
    used: list[str] = []
    implied: dict = {}
    for dest, flag, scorer, extras in _ALIASES:
        if not getattr(args, dest, False):
            continue
        setattr(args, dest, False)
        used.append(flag)
        if scorer not in picked:
            picked.append(scorer)
        implied.update(extras)

    if not picked:
        return

    args.run_score = True
    if not getattr(args, "score_with", None):
        args.score_with = [",".join(picked)]
    # --rescore-poses used to write ranked SDFs unless --rescore-no-sdf was given.
    if getattr(args, "score_alias_rescore_no_sdf", False):
        implied.pop("score_sdf", None)
    for key, value in implied.items():
        if not getattr(args, key, False):
            setattr(args, key, value)

    replacement = f"--score --score-with {','.join(picked)}"
    print(
        f"[score] DEPRECATED: {' and '.join(used)} -> {replacement}. "
        "The old spelling still works and gives the same numbers, but the output "
        "now lands in <output>/score/pose_scores.csv, not in the old per-protocol "
        "files. The alias will be removed."
    )


# ------------------------------------------------------------------- argparse


def _deprecated_alias(old: str, new: str):
    """Argparse action that warns once, then behaves like a normal store."""

    class _Action(argparse.Action):
        _warned = False

        def __call__(self, parser, namespace, values, option_string=None):
            if option_string == old and not _Action._warned:
                _Action._warned = True
                print(f"[score] DEPRECATED: {old} -> {new}")
            setattr(namespace, self.dest, values)

    return _Action


def _add(group, new: str, old: str | None, **kwargs):
    """Register ``new``, optionally with ``old`` as a deprecated spelling.

    The alias carries ``default=SUPPRESS`` semantics by sharing ``new``'s dest and
    never supplying its own default, so an unused old flag cannot clobber the new
    flag's default.
    """
    flags = [new] + ([old] if old else [])
    if old and not kwargs.get("action"):
        kwargs["action"] = _deprecated_alias(old, new)
    return group.add_argument(*flags, **kwargs)


def add_score_args(p) -> None:
    """Register every ``--score*`` option on the (flat, global) parser."""
    g = p.add_argument_group("Score poses")

    # --- what to score with ---------------------------------------------------
    g.add_argument(
        "--score-with", action="append", default=None, metavar="LIST",
        help="Comma-separated scorers to run over every pose: "
             f"{', '.join(SCORER_NAMES)}, or 'all'. Repeatable. The order given is "
             "the column order in the CSV and decides the default ranking. "
             "Default: echo.",
    )

    # --- protocol-level -------------------------------------------------------
    _add(g, "--score-out", "--rescore-out", dest="score_out", default="score",
         help="Output subdirectory under the run output directory.")
    _add(g, "--score-site", "--rescore-site", dest="score_site", default=None,
         help="Force every pose to be scored against this binding-site key. "
              "Default: the site whose box contains the pose, else the nearest "
              "site centroid.")
    g.add_argument("--score-json", action="store_true",
                   help="Also write pose_scores.json, which carries the values a "
                        "flat CSV cannot hold: per-atom Q-scores, the per-feature "
                        "density metrics and the SCI sub-terms.")
    g.add_argument("--score-sdf", action="store_true",
                   help="Also write one SDF per input source, poses best-first, "
                        "with every score as an SD property.")
    g.add_argument("--score-rank-by", default=None, metavar="COLUMN",
                   help="Column the SDF poses are sorted by. Default: the first "
                        "selected scorer's headline column.")
    g.add_argument("--score-case-id", default=None,
                   help="Case id written into every row. Default: the output "
                        "directory's basename.")

    # --- echo -----------------------------------------------------------------
    e = p.add_argument_group("Score poses: ECHO")
    _add(e, "--score-echo-engine", "--rescore-engine", dest="score_echo_engine",
         choices=["docking", "docking_v2"], default="docking",
         help="Which compiled ECHO engine to score with. Must match the engine "
              "that produced the poses for the totals to be comparable.")
    _add(e, "--score-echo-rep-max", "--rescore-rep-max", dest="score_echo_rep_max",
         type=float, default=None,
         help="ECHO repulsion cap. Unset uses --repulsion-cap-polish (the cap the "
              "docking engine's final polish scores with, and so the one the "
              "returned poses are ranked by). Note this is NOT the run_echo_score "
              "pybind default of 5.0; scoring with 5.0 makes every pose look "
              "several units better than dock said.")
    _add(e, "--score-echo-interaction-cutoff", "--rescore-interaction-cutoff",
         dest="score_echo_interaction_cutoff", type=float, default=6.0,
         help="ECHO interaction cutoff in Angstrom.")
    _add(e, "--score-echo-electro-clamp", "--rescore-electro-clamp",
         dest="score_echo_electro_clamp", type=float, default=2.0,
         help="ECHO electrostatic repulsion clamp.")

    # Ligand polar-hydrogen torsion relaxation.
    e.add_argument("--score-echo-minimise-hydrogens", "--rescore-minimise-hydrogens",
                   dest="score_echo_minimise_hydrogens", action="store_true",
                   help="Relax the ligand's polar (donor N/O/S-H) torsions against "
                        "ECHO before scoring, so a pose is not penalised for the H "
                        "placement its SDF happened to carry. Only rotatable donor "
                        "H's move -- no heavy atom can, by construction. Because "
                        "this optimises ECHO with ECHO, the pre-relaxation total is "
                        "always reported alongside as echo_total_prehmin. Note the "
                        "relaxed hydrogens are what every other selected scorer "
                        "sees; mmgbsa is all-atom and is therefore affected.")
    _add(e, "--score-echo-h-min-grid", "--rescore-h-min-grid",
         dest="score_echo_h_min_grid", type=float, default=60.0,
         help="Coarse scan step in degrees used to seed Nelder-Mead. Cost is "
              "passes x torsions x (360/grid) evaluations.")
    _add(e, "--score-echo-h-min-passes", "--rescore-h-min-passes",
         dest="score_echo_h_min_passes", type=int, default=2,
         help="Sweeps over the torsion list during the coarse scan.")
    _add(e, "--score-echo-h-min-maxiter", "--rescore-h-min-maxiter",
         dest="score_echo_h_min_maxiter", type=int, default=100,
         help="Nelder-Mead iteration cap for the polish after the scan.")

    # Protein donor-hydrogen relaxation.
    e.add_argument("--score-echo-protein-h", "--rescore-protein-h",
                   dest="score_echo_protein_h", action="store_true",
                   help="Also relax the *protein's* rotatable donor hydrogens (Ser "
                        "OG, Thr OG1, Tyr OH, Cys SG, Lys NZ) against ECHO for each "
                        "pose. Only the hydrogen moves. Rotation is unpenalised, so "
                        "the relaxed total is an upper bound and "
                        "echo_total_prot_h_pre is always reported next to it. The "
                        "protein is restored between poses, so poses stay "
                        "comparable. Writes echo_protein_h.csv.")
    _add(e, "--score-echo-protein-h-grid", "--rescore-protein-h-grid",
         dest="score_echo_protein_h_grid", type=float, default=30.0,
         help="Coarse scan step in degrees for the protein donors.")
    _add(e, "--score-echo-protein-h-passes", "--rescore-protein-h-passes",
         dest="score_echo_protein_h_passes", type=int, default=2,
         help="Sweeps over the donor list during the coarse scan.")
    _add(e, "--score-echo-protein-h-maxiter", "--rescore-protein-h-maxiter",
         dest="score_echo_protein_h_maxiter", type=int, default=100,
         help="Nelder-Mead iteration cap for the polish after the scan.")

    # --- qscore ---------------------------------------------------------------
    q = p.add_argument_group("Score poses: Q-score")
    q.add_argument("--score-qscore-sigma-ref", type=float, default=None,
                   help="Reference Gaussian width. Unset uses the shared "
                        "--sigma-ref, which the orchestrator and smart_refine_2 "
                        "also read.")
    q.add_argument("--score-qscore-per-atom", "--per-atom",
                   dest="score_qscore_per_atom", action="store_true",
                   help="Also record the per-atom Q-scores. They cannot go in a "
                        "flat CSV, so this implies --score-json.")
    q.add_argument("--score-qscore-low-tail-fraction", type=float, default=0.3,
                   help="Fraction of the worst-fitting atoms averaged into "
                        "q_low_tail, which catches a pose that fits well overall "
                        "but has a group hanging out of density.")

    # --- density --------------------------------------------------------------
    d = p.add_argument_group("Score poses: density fit")
    d.add_argument("--score-density-region", choices=["full", "box", "site"],
                   default="full",
                   help="Region the density-fit terms are scored against. full "
                        "(default): the whole map, so the coverage denominator is a "
                        "per-case constant. box: a fixed-size cube around each pose. "
                        "site: the segmented binding-site map -- this is what "
                        "--orchestrate scores against, but that map is rescaled by a "
                        "boundary EDT, so its numbers are NOT comparable with the "
                        "other two, and it requires segmentation to have run.")
    d.add_argument("--score-density-box-size", type=float, default=24.0, metavar="A",
                   help="Cube edge for --score-density-region box.")
    d.add_argument("--score-density-threshold-frac", type=float, default=0.05,
                   help="Fraction of the map maximum below which voxels are treated "
                        "as background.")
    d.add_argument("--score-density-no-sci", action="store_true",
                   help="Skip the SCI score and its sub-terms.")
    d.add_argument("--score-density-no-shape", action="store_true",
                   help="Skip the shape/skeleton descriptors.")

    # --- mmgbsa ---------------------------------------------------------------
    m = p.add_argument_group("Score poses: MM-GBSA")
    m.add_argument("--score-mmgbsa-minimise", action="store_true",
                   help="Relax the ligand inside a pinned pocket before the "
                        "single-frame energy evaluation. MM-GBSA is defined on "
                        "refined geometry, so on raw docked poses a residual clash "
                        "gives an absurd deltaG; this removes it cheaply and reports "
                        "how far the ligand moved as mmgbsa_min_shift_A.")
    m.add_argument("--score-mmgbsa-min-iters", type=int, default=300,
                   help="Max iterations for the pocket minimisation.")
    m.add_argument("--score-mmgbsa-pocket-radius", type=float, default=12.0,
                   metavar="A",
                   help="Residues within this distance of the ligand are included "
                        "and pinned during --score-mmgbsa-minimise.")
    m.add_argument("--score-mmgbsa-no-cache", action="store_true",
                   help="Rebuild the OpenMM systems and Contexts for every pose "
                        "instead of reusing them per ligand. Bit-identical, just "
                        "much slower; for debugging the cache.")
    m.add_argument("--score-mmgbsa-write-minimised", action="store_true",
                   help="Write the pocket-relaxed poses produced by "
                        "--score-mmgbsa-minimise (which it requires) to "
                        "<stem>_mmgbsa_minimised.sdf, one file per input source, "
                        "plus mmgbsa_receptor.pdb to load them against. NOTE only "
                        "the ligand is minimised: the pocket residues are "
                        "positionally pinned and their relaxed coordinates are "
                        "discarded, so the receptor written is the prepared input, "
                        "unchanged.")

    # --- deprecated protocol-selection flags ----------------------------------
    # These used to be REGISTRY entries, so argparse generated `--rescore-poses`,
    # `-rp` and `--mapq-score` automatically. They are re-registered by hand here
    # and translated by apply_score_back_compat().
    dep = p.add_argument_group("Score poses: deprecated aliases")
    dep.add_argument("--rescore-poses", "-rp", dest="score_alias_rescore_poses",
                     action="store_true",
                     help="DEPRECATED: use --score --score-with echo.")
    dep.add_argument("--mapq-score", dest="score_alias_mapq_score",
                     action="store_true",
                     help="DEPRECATED: use --score --score-with qscore.")
    # Not SUPPRESSed: generate_custom_usage() prints action.help verbatim, so a
    # suppressed option shows up in the usage banner as a literal "==SUPPRESS==".
    dep.add_argument("--rescore-no-sdf", dest="score_alias_rescore_no_sdf",
                     action="store_true",
                     help="DEPRECATED: only meaningful with --rescore-poses, which "
                          "writes SDFs by default. --score does not, so there is "
                          "nothing to turn off.")
