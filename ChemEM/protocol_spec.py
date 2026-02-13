#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>


"""
ChemEM2 protocol registry.

This file lists the protocols ChemEM2 knows about (e.g. binding_site, dock)
and the CLI options for each one.

How it works
------------
- Each protocol is described by a ProtocolSpec:
    * name: registry key / protocol name
    * cls:  protocol class to run
    * deps(args): returns names of protocols that must run first
    * add_args(parser): adds argparse options for this protocol (optional)
    * help: short description for --help output

Adding a new protocol
---------------------
1) Write the protocol class (ChemEM.protocols.<...>).
2) Write a deps() function (return () if none).
3) Write an add_args() function to add CLI options (optional).
4) Add a ProtocolSpec entry to REGISTRY.
5) (Optional) add a short alias in SHORT_ALIASES.
"""


from dataclasses import dataclass
from typing import Callable, Optional

from ChemEM.protocols.docking import Docking
from ChemEM.protocols.binding_site import BindingSite
from ChemEM.protocols.alpha_mask import AlphaMask
from ChemEM.protocols.confidence_map import ConfidenceMap

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
    g = p.add_argument_group("Binding site")
    
    g.add_argument("--probe-sphere-min", type=float, default=3.0)
    g.add_argument("--probe-sphere-max", type=float, default=6.0)
    g.add_argument("--first-pass-thr", type=float, default=1.73)
    g.add_argument("--fist-pass-cluster-size", type=int, default=35)
    g.add_argument("--second-pass-thr", type=float, default=4.5)
    g.add_argument("--binding-site-padding", type=float, default=6.0)
    g.add_argument("--binding-site-grid-spacing", type=float, default=0.5)
    g.add_argument("--third-pass-thr", type=float, default=2.5)
    g.add_argument("--n-overlaps", type=int, default=2)
    g.add_argument("--n-opening-voxels", type=int, default=10)
    g.add_argument("--voxel-buffer", type=float, default=1.5)
    g.add_argument("--fall-back_radius", type=float, default=15.0)
    g.add_argument("--lining_residue_distance", type=float,default=2.0)
    g.add_argument("--force-new-site", action="store_true")


def confidence_map_deps(args):
    return tuple() 

def add_confidence_map_args(p):
    return

def dock_deps(args):
    # Dock always needs binding_site first 
    return ("binding_site", "alpha_mask","confidence_map")

def add_dock_args(p):
    g = p.add_argument_group("Docking")
    
    g.add_argument("--rescore", action="store_true")
    g.add_argument("-fr", "--flexible-rings", action="store_true")
    g.add_argument("-ss", "--split-site", action="store_true")
    g.add_argument("-np", "--no-para", action="store_true")
    g.add_argument("--n-global-search", type=int, default=1000)
    g.add_argument("--n-local-search", type=int, default=10)
    g.add_argument("-br", "--bias-radius", type=float, default=12.0)
    g.add_argument("--cluster-docking", type=float, default=2.0)
    g.add_argument("--energy-cutoff", type=float, default=1.0)
    g.add_argument("--minimize-docking", action="store_true")
    g.add_argument("--aggregate-sites", action="store_true")
    g.add_argument("--water-refine", action="store_true")
    g.add_argument("--sci-weight", type=float, default=2.5,
                   help="scaling factor for the SCI score when docking with a density map")
    g.add_argument("--mi-weight", type=float, default=100.0,
                   help="scaling factor for the MIscore when docking with a density map")
    


def alpha_mask_deps(args):
    return ("binding_site","confidence_map")

def add_alpha_mask_args(p):
    # --- Alpha-sphere / SES geometry ---
    g = p.add_argument_group("Alpha mask: geometry")
    #g.add_argument("--probe-sphere-min", type=float, default=3.0,
    #               help="Minimum probe sphere radius used for masking")
    #g.add_argument("--probe-sphere-max", type=float, default=6.0,
    #               help="Maximum probe sphere radius used for masking")
    g.add_argument( "--alpha-pad", type=float, default=8.0,
                   help="How much to pad around the model when calculating density")
    
    g.add_argument("--ses-mask", action="store_true",
                   help="Mask using SES (solvent-excluded surface) instead of alpha-spheres")
    g.add_argument("--no-boundry", action="store_true",
                   help="Use large alpha-spheres to estimate the bulk-solvent boundary")
    #g.add_argument("--force-new-site", action="store_true",
    #               help="Force creation of a new site (ignore cached/previous site)")

    # --- SES post-processing ---
    g = p.add_argument_group("Alpha mask: SES mask post-processing")
    g.add_argument("--no-otsu-filter-ses-mask", action="store_true",
                   help="Disable Otsu-based smoothing/filtering of the SES mask")

    # --- Density segmentation ---
    g = p.add_argument_group("Alpha mask: Density segmentation")
    g.add_argument("--otsu-segment", action="store_true",
                   help="Use Otsu-based density segmentation")
    g.add_argument("--grad-thr", type=float, default=0.4,
                   help="Gradient threshold for density segmentation")
    g.add_argument("--sigma-coeff", type=float, default=0.356,
                   help="Sigma coefficient used when blurring simulated densities")
    g.add_argument("--segment-binding-sites", action="store_true",
                   help="Limit density segmentation to binding sites only")

    # --- Significant feature filters ---
    g = p.add_argument_group("Alpha mask: feature filters")
    g.add_argument("--sf-amp-frac", type=float, default=0.8,
                   help="Amplitude fraction threshold for feature inclusion")
    g.add_argument("--sf-volume-thr", type=float, default=15.0, metavar="Å^3",
                   help="Minimum feature volume for inclusion")
    g.add_argument("--sf-centroid-thr", type=float, default=2.1, metavar="Å",
                   help="Centroid distance threshold for feature inclusion")
    g.add_argument("--sf-sigma-thr", type=float, default=2.0, metavar="STD",
                   help="Sigma threshold for feature inclusion")

    

SHORT_ALIASES = {
    "binding_site": "-b",
    "dock": "-d",
    "alpha_mask" : "-am",
}


REGISTRY = {
    "binding_site": ProtocolSpec(
        name="binding_site",
        cls=BindingSite,
        deps=binding_site_deps,
        add_args=add_binding_site_args,
        help="Prepare/identify binding site",
    ),
    
    "confidence_map": ProtocolSpec(
        name="confidence_map",
        cls=ConfidenceMap,
        deps=confidence_map_deps,
        add_args=add_confidence_map_args,
        help="FDR Confidence map",
    ),
    
    "alpha_mask": ProtocolSpec(
        name="alpha_mask",
        cls=AlphaMask,
        deps=alpha_mask_deps,
        add_args=add_alpha_mask_args,
        help="Segment ligand density",
    ),
    
    "dock": ProtocolSpec(
        name="dock",
        cls=Docking,
        deps=dock_deps,
        add_args=add_dock_args,
        help="Dock ligands into the binding site",
    ),
}