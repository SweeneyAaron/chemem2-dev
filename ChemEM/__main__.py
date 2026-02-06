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
from ChemEM.config import Config
from ChemEM.messages import Messages



def load_system(conf_file: str):
    cfg = Config()
    return cfg.load_config(conf_file)

def _flag_name(proto_key: str) -> str:
    # binding_site -> --binding-site
    return "--" + proto_key.replace("_", "-")

def build_parser() -> argparse.ArgumentParser:
    
    p = argparse.ArgumentParser(
        prog="chemem",
        description="ChemEM command-line interface",
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
    
    shared.add_argument("--ncpu", type=int, default=int(max(1, os.cpu_count() - 2)))
    
    shared.add_argument("--no-map", action="store_true",
                        help="Disable density map usage")
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
    if args.platform is not None:
        system.platform = args.platform

    if getattr(args, "no_map", False):
        # only do this if your System supports it
        system.density_map = None

    if getattr(args, "output", None) is not None:
        # only do this if your System supports it
        system.output = args.output
        
def build_pipeline(system, ordered_protocols: list[str]) -> None:
    for name in ordered_protocols:
        system.add_protocol(REGISTRY[name].cls(system))

def main() -> None:
    args = build_parser().parse_args()
    print(Messages.intro(ChemEM.__version__))

    if not os.path.exists(args.config):
        print("Config not found:", args.config)
        sys.exit(1)

    #try:
    if True:
        system = load_system(args.config)

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





