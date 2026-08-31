#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>


from __future__ import annotations
import argparse
import os
import sys
import traceback

import ChemEM
from ChemEM.protocol_spec import REGISTRY, SHORT_ALIASES
from ChemEM.messages import Messages
from ChemEM.tools.resources import (
    apply_cpu_budget,
    apply_cpus_per_site,
    default_cpu_budget,
)



def generate_custom_usage() -> str:
    """Generates a clean, aligned usage string dynamically from the REGISTRY, including protocol options."""
    from ChemEM.protocol_spec import REGISTRY, SHORT_ALIASES

    lines = [
        "chemem <config_file> [protocols...] [options...]",
        "",
        "Available Protocols & Options:",
        "=============================="
    ]

    for key, spec in REGISTRY.items():
        # 1. Format the main protocol flag
        long_flag = "--" + key.replace("_", "-")
        short_flag = SHORT_ALIASES.get(key)
        flag_str = f"{long_flag}, {short_flag}" if short_flag else long_flag
        
        # Add the Protocol Header
        lines.append(f"• {flag_str}")
        if spec.help:
            lines.append(f"    {spec.help}")

        # 2. Extract arguments for this specific protocol
        if spec.add_args:
            # Create a dummy parser just to capture the arguments added by this protocol
            temp_p = argparse.ArgumentParser(add_help=False)
            spec.add_args(temp_p)
            
            # _actions contains all the arguments added
            actions = temp_p._actions
            
            if actions:
                lines.append("    Options:")
                # Find the longest option string so we can align the help text perfectly
                max_opt_len = max([len(", ".join(a.option_strings)) for a in actions])
                
                for action in actions:
                    opt_strs = ", ".join(action.option_strings)
                    padded_opt = opt_strs.ljust(max_opt_len + 2)
                    
                    help_txt = action.help or ""
                    
                    # Append default values (skip for boolean True/False flags to keep it clean)
                    if action.default is not None and action.default != argparse.SUPPRESS:
                        if not isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                            default_str = f"[default: {action.default}]"
                            help_txt = f"{help_txt} {default_str}".strip()

                    lines.append(f"      {padded_opt}{help_txt}")
        
        # Add a blank line between protocols to keep it from looking cluttered
        lines.append("")

    lines.extend([
        "Examples:",
        "  chemem config.txt --dock --minimize-docking",
        "  chemem config.txt --mapq-score --rescore",
        "  chemem config.txt -b -d  # Using short aliases"
    ])

    return "\n".join(lines)

# CLI flags that must reach Config BEFORE create_system() builds the protein.
# Everything else reaches protocols afterwards via system.options; these cannot,
# because protein preparation happens inside create_system().
PRE_PROTEIN_OVERRIDES = (
    "prep_platform",
    "prep_threads",
    "prep_seed",
    "deterministic_prep",
    "prep_clash_relief_steps",
    "prep_h_implicit",
    "cache_protein",
    "protein_cache_dir",
    "refresh_protein_cache",
)


def load_system(conf_file: str, args=None):
    from ChemEM.config import Config

    overrides = None
    if args is not None:
        overrides = {name: getattr(args, name, None) for name in PRE_PROTEIN_OVERRIDES}

    cfg = Config()
    return cfg.load_config(conf_file, overrides=overrides)

def _flag_name(proto_key: str) -> str:
    # binding_site -> --binding-site
    return "--" + proto_key.replace("_", "-")

def build_parser() -> argparse.ArgumentParser:
    
    p = argparse.ArgumentParser(
        prog="chemem",
        description="ChemEM command-line interface",
        usage = generate_custom_usage(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {ChemEM.__version__}",
    )

    p.add_argument("config", help="Path to ChemEM configuration file")

    # Shared options that are used by multiple protocols .
    
    shared = p.add_argument_group("shared protocol options")
    
    shared.add_argument("--platform", type=str, default=None,
                        help="OpenMM Platform: CPU, OpenCL, CUDA")
   
    shared.add_argument("--output", type=str, default=None,
                        help="Output directory")
    
    shared.add_argument("--ncpu", type=int, default=default_cpu_budget())

    shared.add_argument(
        "--cpus-per-site",
        type=int,
        default=None,
        help=(
            "CPUs allocated per split-site docking job. Defaults to "
            "max(2, ncpu // 4) to keep multi-job parallelism alive on small "
            "machines; raise it to give each site more cores at the cost of "
            "fewer parallel sites."
        ),
    )

    shared.add_argument("--no-map", action="store_true",
                        help="Disable density map usage")

    # Read by --score's Q-score scorer, by the orchestrator's triage and by
    # smart_refine_2's Q-score objective, so it is shared rather than owned by any
    # one protocol. --score-qscore-sigma-ref overrides it for scoring only.
    shared.add_argument("--sigma-ref", type=float, default=0.6,
                        help="Reference Gaussian width for Q-score, in Angstrom.")

    # --- protein preparation determinism ---
    # Applied before the protein is built, unlike every other option here, so they
    # are read by load_system() rather than apply_overrides().
    shared.add_argument("--prep-platform", type=str, default=None,
                        choices=["CPU", "Reference", "OpenCL", "CUDA", "inherit"],
                        help="OpenMM platform for the two minimisations in protein "
                             "preparation (PDBFixer's rebuilt-atom relaxation and "
                             "hydrogen placement). Default CPU, which is reproducible "
                             "and ~10x faster than Reference. Reference additionally "
                             "gives cross-machine identity. 'inherit' restores the old "
                             "auto-selection, which is NOT reproducible.")
    shared.add_argument("--prep-threads", type=int, default=None,
                        help="Thread count for the prep platform. 1 (default) removes "
                             "thread-count dependence from the result.")
    shared.add_argument("--prep-seed", type=int, default=None,
                        help="Seed for the Langevin dynamics PDBFixer runs on rebuilt "
                             "atoms. Must be non-zero: OpenMM reads 0 as 'pick a fresh "
                             "seed', which is what made preparation irreproducible.")
    shared.add_argument("--no-deterministic-prep", dest="deterministic_prep",
                        action="store_false", default=None,
                        help="Restore the previous, non-reproducible protein "
                             "preparation (auto-selected platform, unseeded dynamics).")
    shared.add_argument("--prep-clash-relief-steps", type=int, default=None,
                        help="EXPERT/UNSAFE. Cap the Langevin step budget for "
                             "PDBFixer's clash relief on rebuilt side-chain atoms. "
                             "That loop is 74%% of preparation time and never reaches "
                             "its 1.3 A target on a heavily-repaired receptor, so "
                             "capping it is a big win -- but the useful snapshot lands "
                             "at a structure-dependent iteration. Measured: a 600-step "
                             "cap reproduces the uncapped structure exactly on 9e26 "
                             "yet leaves a 0.655 A worst contact on 7bxu against 1.052 "
                             "A uncapped. Unset (default) keeps PDBFixer's behaviour. "
                             "If you set this, CHECK THE RESULTING CONTACTS. 0 skips "
                             "the dynamics; the minimisation after it always runs.")
    shared.add_argument("--no-prep-h-implicit", dest="prep_h_implicit",
                        action="store_false", default=None,
                        help="Drop implicit solvent from the force field used for "
                             "hydrogen placement. Cuts ~105 s of a 236 s preparation "
                             "(GBn2 is a CustomGBForce over every atom, minimised 50 "
                             "times), but it is NOT score-neutral: the ECHO "
                             "electrostatic grid is built from per-atom charges "
                             "INCLUDING hydrogens, so moving them shifts echo_total "
                             "by up to 0.6 units and would need the ECHO weights "
                             "refitting. On by default for that reason.")
    shared.add_argument("--no-cache-protein", dest="cache_protein",
                        action="store_false", default=None,
                        help="Re-prepare the protein every run instead of reusing a "
                             "cached result. The cache is on by default and is keyed "
                             "on the input file, force field, prep settings and "
                             "library versions.")
    shared.add_argument("--protein-cache-dir", type=str, default=None,
                        help="Where prepared proteins are cached. Defaults to "
                             "$CHEMEM_CACHE_DIR, else $XDG_CACHE_HOME/chemem, else "
                             "~/.cache/chemem.")
    shared.add_argument("--refresh-protein-cache", action="store_true", default=None,
                        help="Ignore any cached prepared protein and rewrite the entry.")

    shared.add_argument("--global-k", type=float, default=None,
                        help="Density-map restraint weight for every minimiser "
                             "(refine, smart_ligand_refine2, ion_fixer, dock, "
                             "lining_refine). Unset keeps each protocol's built-in "
                             "default of 150.0. For ion_fixer this sets the base "
                             "weight; its per-stage scale factors still apply.")

    shared.add_argument("--implicit-solvent", type=str, default=None,
                        choices=["none", "hct", "obc1", "obc2", "gbn", "gbn2"],
                        help="Implicit-solvent model for every minimiser. Unset keeps "
                             "current behaviour: gbn2 for refine/smart_ligand_refine2/"
                             "dock/lining_refine, vacuum for ion_fixer and export. Note "
                             "ion_fixer adds explicit dummy waters to complete the "
                             "coordination shell, so enabling GB there double-counts "
                             "solvation at those sites.")
    #now sure we will use this any more but leave it here untill the correct time.
    shared.add_argument("--dock-setup", type=str, default="sf",
                        help="Optional: controls docking dependency mode (e.g., sf/alpha)")

    # Protocol selection flags
    sel = p.add_argument_group("protocol selection")
    for key, spec in REGISTRY.items():
        long_flag = _flag_name(key)
        short_flag = SHORT_ALIASES.get(key)
        flags = [long_flag] + ([short_flag] if short_flag else [])
        sel.add_argument(*flags, dest=f"run_{key}", action="store_true", help=spec.help or f"Run {key}")

    # Register ALL protocol arguments on the single parser (Option B)
    for spec in REGISTRY.values():
        if spec.add_args:
            spec.add_args(p)

    return p

def selected_protocols(args: argparse.Namespace) -> list[str]:
    picked = [k for k in REGISTRY.keys() if getattr(args, f"run_{k}", False)]
    return picked or ["dock"]

def resolve_protocol_order(selected: list[str], args: argparse.Namespace) -> list[str]:
    ordered: list[str] = []
    temp: set[str] = set()
    perm: set[str] = set()

    def visit(name: str) -> None:
        if name in perm:
            return
        if name in temp:
            raise RuntimeError(f"Dependency cycle detected at '{name}'")
        if name not in REGISTRY:
            raise KeyError(f"Unknown protocol '{name}'")
        temp.add(name)
        for dep in REGISTRY[name].deps(args):
            visit(dep)
        temp.remove(name)
        perm.add(name)
        ordered.append(name)

    for s in selected:
        visit(s)

    return ordered

def apply_overrides(system, args: argparse.Namespace) -> None:
    # Keep this as the only place you mutate the System from CLI
    if getattr(args, "ncpu", None) is not None:
        budget = apply_cpu_budget(system, args.ncpu)
        print(f"[CONFIG] using CPU budget: {budget}")

    if getattr(args, "cpus_per_site", None) is not None:
        per_site = apply_cpus_per_site(system, args.cpus_per_site)
        print(f"[CONFIG] using cpus_per_site: {per_site}")

    if args.platform is not None:
        print(f"[CONFIG] overriding platform {system.platform } with {args.platform}")
        system.platform = args.platform
        
    if getattr(args, "no_map", False):
        # only do this if your System supports it
        system.density_map = None

    if getattr(args, "output", None) is not None:
        # only do this if your System supports it
        system.output = args.output
        
def build_pipeline(system, ordered_protocols: list[str]) -> None:
    for name in ordered_protocols:
        protocol_cls = REGISTRY[name].load_cls()
        system.add_protocol(protocol_cls(system))

def main() -> None:
    args = build_parser().parse_args()

    # Rewrite the deprecated --rescore-poses / --mapq-score spellings into --score.
    # Must happen before selected_protocols(), because --score's dependencies are
    # derived from args.score_with.
    from ChemEM.protocols.score.cli import apply_score_back_compat
    apply_score_back_compat(args)

    print(Messages.intro(ChemEM.__version__))

    if not os.path.exists(args.config):
        print("Config not found:", args.config)
        sys.exit(1)

    #try:
    if True:
        system = load_system(args.config, args)

        # Make args visible, TODO! migrate to typed config
        system.options = args

        apply_overrides(system, args)

        selected = selected_protocols(args)
        order = resolve_protocol_order(selected, args)

        build_pipeline(system, order)
        
        system.run()
        system.write_log()
        #TODO! write log file
        #TODO! decide the output

    #except Exception as err:
    #    print(Messages.fatal_exception("ChemEM CLI", err))
    #    traceback.print_exc()
    #    sys.exit(1)
if __name__ == "__main__":
    main()

