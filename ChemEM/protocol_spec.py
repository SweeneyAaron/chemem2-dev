#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>


#registry.py 

from dataclasses import dataclass
from typing import Callable, Optional

from ChemEM.protocols.docking import Docking
from ChemEM.protocols.binding_site import BindingSite

@dataclass(frozen=True)
class ProtocolSpec:
    name: str                 
    cls: type                 
    deps: Callable            
    add_args: Optional[Callable] = None
    help: str = ""
    
def binding_site_deps(args):
    return tuple()


def add_binding_site_args(p):
    
    p.add_argument("--probe-sphere-min", type=float, default=3.0)
    p.add_argument("--probe-sphere-max", type=float, default=6.0)
    p.add_argument("--first-pass-thr", type=float, default=1.73)
    p.add_argument("--fist-pass-cluster-size", type=int, default=35)
    p.add_argument("--second-pass-thr", type=float, default=4.5)
    p.add_argument("--binding-site-padding", type=float, default=6.0)
    p.add_argument("--binding-site-grid-spacing", type=float, default=0.5)
    p.add_argument("--third-pass-thr", type=float, default=2.5)
    p.add_argument("--n-overlaps", type=int, default=2)
    p.add_argument("--n-opening-voxels", type=int, default=10)
    p.add_argument("--voxel-buffer", type=float, default=1.5)
    p.add_argument("--fall-back_radius", type=float, default=15.0)
    p.add_argument("--lining_residue_distance", type=float,default=2.0)
    p.add_argument("--force-new-site", action="store_true")
    
def dock_deps(args):
    # Dock always needs binding_site first 
    return ("binding_site",)

def add_dock_args(p):
    p.add_argument("--rescore", action="store_true")
    p.add_argument("-fr", "--flexible-rings", action="store_true")
    p.add_argument("-ss", "--split-site", action="store_true")
    p.add_argument("-np", "--no-para", action="store_true")
    p.add_argument("--n-global-search", type=int, default=1000)
    p.add_argument("--n-local-search", type=int, default=10)
    p.add_argument("-br", "--bias-radius", type=float, default=12.0)
    p.add_argument("--cluster-docking", type=float, default=2.0)
    p.add_argument("--energy-cutoff", type=float, default=1.0)
    p.add_argument("--minimize-docking", action="store_true")
    p.add_argument("--aggregate-sites", action="store_true")
    p.add_argument("--water-refine", action="store_true")




SHORT_ALIASES = {
    "binding_site": "-b",
    "dock": "-d",
}


REGISTRY = {
    "binding_site": ProtocolSpec(
        name="binding_site",
        cls=BindingSite,
        deps=binding_site_deps,
        add_args=add_binding_site_args,
        help="Prepare/identify binding site",
    ),
    
    "dock": ProtocolSpec(
        name="dock",
        cls=Docking,
        deps=dock_deps,
        add_args=add_dock_args,
        help="Dock ligands into the binding site",
    ),
}