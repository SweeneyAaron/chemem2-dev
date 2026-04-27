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
from importlib import import_module
from typing import Callable, Optional

@dataclass(frozen=True)
class ProtocolSpec:
    name: str                 
    class_path: str
    deps: Callable            
    add_args: Optional[Callable] = None
    help: str = ""

    def load_cls(self) -> type:
        module_name, _, class_name = self.class_path.partition(":")
        if not module_name or not class_name:
            raise ValueError(
                f"Invalid protocol class path for '{self.name}': {self.class_path}"
            )
        module = import_module(module_name)
        return getattr(module, class_name)
    
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
    return tuple()

def dock_deps(args):
    # Dock always needs binding_site first 
    return ("binding_site", "alpha_mask","confidence_map")

def add_dock_args(p):
    g = p.add_argument_group("Docking")
    
    g.add_argument("--rescore", action="store_true")
    g.add_argument("-fr", "--flexible-rings", action="store_true")
    g.add_argument("-ss", "--split-site", action="store_true")
    g.add_argument("-np", "--no-para", action="store_true")
    g.add_argument("--n-global-search", type=int, default=2000) #8000
    g.add_argument("--n-local-search", type=int, default=20) #change here
    g.add_argument("-br", "--bias-radius", type=float, default=12.0)
    g.add_argument("--cluster-docking", type=float, default=1.0)
    g.add_argument("--energy-cutoff", type=float, default=1.0)
    g.add_argument("--minimize-docking", action="store_true")
    g.add_argument("--refine-to-diff-map",action="store_true")
    g.add_argument("--aggregate-sites", action="store_true")
    g.add_argument("--water-refine", action="store_true")
    g.add_argument("--sci-weight", type=float, default=2.5,
                   help="scaling factor for the SCI score when docking with a density map")
    g.add_argument("--mi-weight", type=float, default=100.0,
                   help="scaling factor for the MIscore when docking with a density map")
    g.add_argument("--repulsion-cap-0", type=float, default=2.0) #from 2 #new 1
    g.add_argument("--repulsion-cap-1", type=float, default=5.0)#from 5
    g.add_argument("--repulsion-cap-nm", type=float, default = 10.0) #from 10 #new is 30.0
    g.add_argument("--repulsion-cap-polish", type=float, default=15.0) #from 15 #new is 30.0
    g.add_argument("--return-n", type=int, default=20)
    g.add_argument("--max-iterations", type=int, default=0)
    g.add_argument("--do-biased-md", action="store_true")
    g.add_argument("--inner-map-score", type=int, default=1)
    g.add_argument("--outer-map-score", type=int, default=0)
    


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
    
    g.add_argument("--sep-features", action="store_true")
    g.add_argument("--sep-features-dist", type=float, default=4.0)
    g.add_argument("--sepf-features-mode", type=str, default="voxels", help="options: voxels | com ")
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


def refine_deps(args):
    # Dock always needs binding_site first 
    #don't use con map here the user can specify 
    #return ("confidence_map",)
    return tuple()

def add_refine_args(p):
    g = p.add_argument_group("Density Refinement")

    g.add_argument('--pp-local-sites', type=str, default = 'global')
    g.add_argument('--md-restraints', type=str, default = 'sse')
    g.add_argument("--restrain-sidechains", action="store_true",
                   help="")
    
    g.add_argument("--global-k", type=float, default=75.0,
                   help="")
    
    g.add_argument('--local-refine', action="store_true")
    g.add_argument('--local-radius', type=float, default = 12.0)
    g.add_argument('--annealing', action="store_true")   
    g.add_argument('--pre-minimise', action="store_true")
    g.add_argument('--post-minimise', action="store_true")
    g.add_argument('--base-temp', type=float, default=50.0)
    g.add_argument('--heat-to-k', type=float, default=150.0)
    g.add_argument('--temp-step-k', type=float, default=5.0)
    g.add_argument('--steps-per-temp', type=int, default=25)
    g.add_argument('--high-hold-ps', type=float, default=0.0)
    g.add_argument('--cycles', type=int, default=1)
    g.add_argument('--seed', type=int, default=1)
    g.add_argument('--com-restraint', action="store_true")
    g.add_argument('--com-restraint-dist', type=float, default=2.0)
    g.add_argument('--com-restrain-kcal-per-mol', type=float, default=20.0)
    g.add_argument('--com_restrain_alpha_per_nm', type=float, default=80.0)

    
def search_refine_deps(args):
    return tuple()

def add_search_refine_args(p):
    g = p.add_argument_group("Search Refine (map-metric guided)")

    g.add_argument("--sr-map-source", type=str, default="confidence",
                   help="Map source policy: confidence | raw | difference")

    g.add_argument("--sr-scorer", type=str, default="sci",
                   choices=["sci", "ccc", "mi", "qscore"],
                   help="Goodness-of-fit metric driving both direction and scoring")
    g.add_argument("--sr-accept-strategy", type=str, default="greedy",
                   choices=["greedy", "metropolis", "basin_hopping"],
                   help="Proposal acceptance strategy")

    g.add_argument("--sr-max-outer-iter", type=int, default=50)
    g.add_argument("--sr-patience", type=int, default=4)
    g.add_argument("--sr-min-delta", type=float, default=1e-6)

    g.add_argument("--sr-proposals-per-iter", type=int, default=6)
    g.add_argument("--sr-md-steps-per-iter", type=int, default=250)
    g.add_argument("--sr-minimise-max-iters", type=int, default=200)
    g.add_argument("--sr-md-temp-k", type=float, default=150.0)
    g.add_argument("--sr-seed", type=int, default=1)

    g.add_argument("--sr-pocket-radius", dest="sr_pocket_radius", type=float, default=12.0,
                   help="Pocket-shell radius for local subset selection (Å)")
    g.add_argument("--sr-map-pad-a", type=float, default=3.0,
                   help="Padding for local map extraction around the local structure (Å)")

    g.add_argument("--sr-global-k", type=float, default=0.0,
                   help="OpenMM map potential weight for search_refine")
    g.add_argument("--sr-pin-k", type=float, default=5000.0,
                   help="Protein pin strength for local environment")

    g.add_argument("--sr-max-atom-delta-a", type=float, default=1.5,#from 0.5
                   help="Maximum per-proposal displacement cap per ligand heavy atom when building pull targets (Å)")
    g.add_argument("--sr-trust-k", type=float, default=250.0,
                   help="Harmonic pull strength toward per-atom target coordinates (kcal/mol/Å^2)")

    # Gradient estimation
    g.add_argument("--sr-fd-step-a", type=float, default=0.25, #from 0.1
                   help="Finite-difference step (Å) for metric-gradient estimation")
    g.add_argument("--sr-fd-mode", type=str, default="central",
                   choices=["central", "forward"],
                   help="Finite-difference scheme for metric gradients")

    # Acceptance (metropolis / basin-hopping)
    g.add_argument("--sr-accept-temp-start", type=float, default=0.05,
                   help="Metropolis temperature at outer-iter 1")
    g.add_argument("--sr-accept-temp-end", type=float, default=0.005,
                   help="Metropolis temperature at max_outer_iter")
    g.add_argument("--sr-basin-hop-stale", type=int, default=3,
                   help="Consecutive rejects before basin-hopping emits a perturb")
    g.add_argument("--sr-basin-hop-sigma-a", type=float, default=0.3,
                   help="Per-atom Gaussian kick sigma (Å) on a basin-hopping perturb")

    # MI-specific
    g.add_argument("--sr-mi-fd-step-a", type=float, default=0.1,
                   help="FD step (Å) for MI gradient (histogram MI is discontinuous)")
    g.add_argument("--sr-mi-nbins", type=int, default=20,
                   help="Number of histogram bins for MI")
    g.add_argument("--sr-mi-normalized", action="store_true", default=False,
                   help="Use normalized MI (NMI) as the MI score")

    # Q-score specific
    g.add_argument("--sr-qscore-sigma-ref", type=float, default=0.6,
                   help="Reference Gaussian sigma for Q-score")
    g.add_argument("--sr-qscore-radii", type=str, default=None,
                   help="Comma-separated radii (Å) for Q-score; default uses built-in shells")

    # CCC-specific
    g.add_argument("--sr-ccc-mask-mode", type=str, default="nonzero",
                   choices=["nonzero", "full"],
                   help="Voxel mask mode for truncated CC")

    # SCI-specific knobs (consumed only when --sr-scorer sci)
    g.add_argument("--sr-use-amp-eq", dest="sr_use_amp_eq", action="store_true", default=True,
                   help="Enable Fourier amplitude equalization before SCI channel scoring")
    g.add_argument("--sr-no-amp-eq", dest="sr_use_amp_eq", action="store_false",
                   help="Disable Fourier amplitude equalization before SCI channel scoring")
    g.add_argument("--sr-sci-sigma", type=float, default=1.0,
                   help="Gaussian derivative scale (pixels) used for SCI channels")
    g.add_argument("--sr-sigma-coeff", type=float, default=0.356,
                   help="Sigma coefficient for simulated ligand density blur")
    g.add_argument("--sr-sci-eps", type=float, default=1e-8,
                   help="Numerical epsilon for log-domain SCI fusion")
    g.add_argument("--sr-normalise-sim-map", dest="sr_normalise_sim_map", action="store_true", default=True,
                   help="Normalize simulated ligand map before SCI scoring")
    g.add_argument("--sr-no-normalise-sim-map", dest="sr_normalise_sim_map", action="store_false",
                   help="Disable normalization of simulated ligand map before SCI scoring")

    g.add_argument("--sr-w0", type=float, default=1.0,
                   help="SCI weight for 0th-derivative CC channel")
    g.add_argument("--sr-w1", type=float, default=1.0,
                   help="SCI weight for first-derivative CC channels")
    g.add_argument("--sr-w2", type=float, default=1.0,
                   help="SCI weight for second-derivative CC channels")
    g.add_argument("--sr-w-energy", type=float, default=0.0,
                   help="Hybrid score penalty weight for z-scored MM energy")
    g.add_argument("--sr-w-clash", type=float, default=0.0,
                   help="Hybrid score penalty weight for z-scored clash penalty")
    g.add_argument("--sr-clash-distance-a", type=float, default=1.6,
                   help="Soft clash distance threshold for clash penalty (Å)")

    g.add_argument("--sr-rmsd-dedupe", type=float, default=0.5,
                   help="RMSD threshold (Å) for deduplicating ranked refined poses")
    g.add_argument("--sr-return-n", type=int, default=1,
                   help="Maximum number of refined poses to keep per ligand. 1 (default) emits only the best pose. Larger values emit additional poses only if they are within --sr-return-score-margin of the best AND separated by more than --sr-rmsd-dedupe")
    g.add_argument("--sr-return-score-margin", type=float, default=0.01,
                   help="Maximum final-score gap from the best pose for an additional pose to be returned when --sr-return-n > 1")
    g.add_argument("--sr-stage", type=str, default="v2",
                   choices=["v2", "legacy"],
                   help="Refinement stage preset. 'v2' (default) enables analytical CCC gradient, diagnostic, dihedral / sub-region proposals, and directed basin-hopping kicks when their respective flags are set. 'legacy' forces the pre-refactor FD/tether/random-kick path bit-exact for A/B regression")
    g.add_argument("--sr-verbose", action="store_true", default=False,
                   help="Enable per-iteration/proposal debug output for search_refine")
    g.add_argument("--sr-log-every", type=int, default=1,
                   help="When --sr-verbose is enabled, print outer-loop updates every N iterations")
    g.add_argument("--sr-debug-relax", action="store_true", default=False,
                   help="With --sr-verbose, print per-stage OpenMM diagnostics (energy/forces/displacements) for each proposal")
    g.add_argument("--sr-diagnostic", action="store_true", default=False,
                   help="Per-iter per-atom fit diagnostic (Q-score + |∇CCC|, classified by Q thresholds). Requires --sr-verbose to be printed")
    g.add_argument("--sr-q-good-thresh", type=float, default=0.7,
                   help="Q-score threshold above which an atom is classified 'good' in the diagnostic")
    g.add_argument("--sr-q-bad-thresh", type=float, default=0.3,
                   help="Q-score threshold below which an atom is classified 'bad' in the diagnostic")
    g.add_argument("--sr-target-bad-only", action="store_true", default=False,
                   help="Apply gradient-driven tether displacement only to atoms classified 'bad' by the Q-score diagnostic; all other atoms are pinned at their current position. Implies the diagnostic is computed each iter")
    g.add_argument("--sr-dihedral-proposals-per-iter", type=int, default=0,
                   help="Per iter, replace up to N gradient proposals with dihedral-rotation proposals targeting bad atoms along their gradient direction. 0 disables. Implies the diagnostic is computed each iter")
    g.add_argument("--sr-subregion-proposals-per-iter", type=int, default=0,
                   help="Per iter, replace up to N proposals with sub-region rigid-body tweaks (closed-form Kabsch fit) applied to contiguous clusters of bad atoms. 0 disables. Implies the diagnostic is computed each iter")
    g.add_argument("--sr-subregion-min-size", type=int, default=3,
                   help="Minimum cluster size (in heavy atoms) for a sub-region rigid-body proposal to be considered")
    g.add_argument("--sr-subregion-max-size", type=int, default=8,
                   help="Maximum cluster size (in heavy atoms) for a sub-region rigid-body proposal; larger clusters are skipped to keep moves local")
    g.add_argument("--sr-directed-kick", action="store_true", default=False,
                   help="Replace the basin-hopping Gaussian kick with a directed dihedral delta on the worst-Q atom aligned with its CCC gradient. Falls back to Gaussian if no viable rotatable bond exists")
    g.add_argument("--sr-directed-kick-angle-deg", type=float, default=90.0,
                   help="Rotation magnitude (degrees) applied by the directed basin-hopping kick")

    
    
def mapq_score_deps(args):
    return tuple() 

def add_mapq_score_args(p):
    g = p.add_argument_group("MapQ Score")
    g.add_argument("--sigma-ref", type=float, default = 0.6)
    p.add_argument("--per-atom", action="store_true", help="Get per atom MapQ scores")


def ion_template_search_deps(args):
    return tuple()


def add_ion_template_search_args(p):
    g = p.add_argument_group("Ion Template Search")
    g.add_argument(
        "--its-auto-run-ion-fixer",
        action="store_true",
        default=False,
        help="After confident template mapping, execute IonFixer immediately in the same run.",
    )
    g.add_argument(
        "--its-confidence-thresh",
        type=float,
        default=0.45,
        help="Minimum confidence required to auto-populate IonFixer arguments.",
    )
    g.add_argument(
        "--its-max-entry-candidates",
        type=int,
        default=1000,
        help="Maximum RCSB search hits retained before detailed template evaluation.",
    )
    g.add_argument(
        "--its-homolog-identity-min",
        type=float,
        default=0.35,
        help="Minimum sequence identity for a PDB chain to qualify as a homolog of the target (RCSB sequence service identity_cutoff).",
    )
    g.add_argument(
        "--its-max-homolog-entries",
        type=int,
        default=1000,
        help="Maximum homolog polymer-entity hits to retain when pre-filtering chemical-search candidates.",
    )
    g.add_argument(
        "--its-max-templates",
        type=int,
        default=100,
        help="Maximum candidate templates to evaluate in depth.",
    )
    g.add_argument(
        "--its-seq-identity-min",
        type=float,
        default=0.35,
        help="Minimum template-target chain sequence identity for residue mapping.",
    )
    g.add_argument(
        "--its-local-chain-radius-a",
        type=float,
        default=12.0,
        help="Only target chains with residues within this radius of the fitted ligand are eligible for sequence mapping (Å).",
    )
    g.add_argument(
        "--its-ion-elements",
        type=str,
        default="",
        help="Comma-separated metal element allowlist, e.g. 'ZN,MG,CA'. Empty uses built-in defaults.",
    )
    g.add_argument(
        "--its-similarity-enabled",
        dest="its_similarity_enabled",
        action="store_true",
        default=True,
        help="Enable ligand similarity expansion when exact template hits are insufficient.",
    )
    g.add_argument(
        "--its-no-similarity",
        dest="its_similarity_enabled",
        action="store_false",
        help="Disable ligand similarity search and use exact matches only.",
    )


def ion_fixer_deps(args):
    return tuple() 

def add_ion_fixer_args(p):
    g = p.add_argument_group("Ion Fixer")
    g.add_argument("--ion-type", type=str)
    g.add_argument("--coordination-geometry", type=str, default='Octahedral', help="Coordination geometry : Octahedral |Square Planar | linear | Trigonal Bipyramidal | Triganal Planer | Square Pyrimidal | Tetrahedral | Pentagonal Bipyrimidal")
    
    g.add_argument(
        "--atom-spec",
        dest="atom_specs",
        action="append",
        default=[],
        help=(
            "Atom specification for a coordinating atom. "
            "Repeat this option multiple times. "
            "Format example: A:ASP:45:OD1 or LIG:0:O3"
        ))
    
    g.add_argument(
        "--exclude-spec",
        dest="exclude_specs",
        action="append",
        default=[],
        help=(
            "Atom specification for a coordinating atom. "
            "Repeat this option multiple times. "
            "Format example: A:ASP:45:OD1 or LIG:0:O3"
        ))
    
    g.add_argument(
        "--pin-spec",
        dest="pin_specs",
        action="append",
        default=[],
        help=(
            "Atom specification to pin during refinement/annealing. "
            "Repeat this option multiple times. "
            "Format example: A:ASP:45:OD1 or LIG:0:O3"
        ),
    )
    
    g.add_argument(
        "--distance-spec",
        dest="distance_specs",
        action="append",
        default=[],
        help=(
            "Distance restraint specification. Repeat this option multiple times. "
            "Format: <atom1-spec>;<atom2-spec>;<distance-in-A>. "
            "Example: A:ASP:45:OD1;LIG:0:O3;2.1"
        ),
    )
    
    g.add_argument("--ion-forcefield",type=str,default="amber14/tip3pfb.xml")
    g.add_argument("--k_ang", type=float, default=None)
    g.add_argument("--distance_fraction", type=float, default=0.9)
    g.add_argument("--n-cycles", type=int, default=60)
    
    
def lining_refine_deps(args):
    # Needs binding_site to obtain site.distance_map + lining_residues.
    return ("binding_site",)

def add_lining_refine_args(p):
    g = p.add_argument_group("Lining-residue refinement")

    # --- Detection (density-blob centroid method) ---
    g.add_argument("--lr-sigma-thr", type=float, default=1.5,
                   help="Map threshold in sigma (mean + sigma*std) for blob seeding")
    g.add_argument("--lr-crop-pad", type=float, default=2.0,
                   help="Padding around site bounding box when cropping the map (Å)")
    g.add_argument("--lr-blob-vol-min", type=float, default=20.0,
                   help="Minimum connected-component volume to be considered a ligand blob (Å³)")
    g.add_argument("--lr-centroid-depth-thr", type=float, default=2.0,
                   help="Minimum pocket depth at the blob centroid (Å)")
    g.add_argument("--lr-misfit-frac", type=float, default=0.5,
                   help="Fraction of a residue's heavy sidechain atoms that must be in-blob")
    g.add_argument("--lr-allow-backbone-bridge", action="store_true",
                   help="Do not reject blobs that touch backbone density")
    g.add_argument("--lr-backbone-bridge-dist", type=float, default=0.8,
                   help="Backbone-neighbour distance used by the bridge filter (Å)")

    # --- Refinement ---
    g.add_argument("--lr-neighborhood", type=float, default=10.0,
                   help="Radius around flagged atoms for the refinement subset (Å)")
    g.add_argument("--lr-global-k", type=float, default=150.0,
                   help="Density-map global k for the local refinement")
    g.add_argument("--lr-backbone-k", type=float, default=1000.0,
                   help="Soft positional pin k for non-flagged heavy atoms")
    g.add_argument("--lr-repel-k", type=float, default=500.0,
                   help="Strength of the pocket-repulsive force applied to flagged atoms")


def export_deps(args):
    return tuple()

def add_export_args(p):
    pass


def orchestrate_deps(args):
    return ("binding_site", "confidence_map", "alpha_mask")


def add_orchestrate_args(p):
    g = p.add_argument_group("Smart Orchestrator")
    g.add_argument("--orch-gate1-topk", type=int, default=5,
                   help="Per site, keep top-K candidates after Gate 1 (Q-score triage).")
    g.add_argument("--orch-gate2-topk", type=int, default=2,
                   help="Per site, keep top-K candidates after Gate 2 (Q-score + MMGBSA).")
    g.add_argument("--orch-w-qscore", type=float, default=1.0,
                   help="Weight on z(qscore) in Gate 2/3 composite (higher = better).")
    g.add_argument("--orch-w-mmgbsa", type=float, default=0.5,
                   help="Weight on -z(mmgbsa deltaG) in Gate 2/3 composite (lower deltaG = better).")
    g.add_argument("--orch-skip-mmgbsa", action="store_true",
                   help="Skip MMGBSA scoring at Gate 2/3 (use Q-score alone).")
    g.add_argument("--orch-skip-search-refine", action="store_true",
                   help="Skip the SearchRefine stage; pick final assembly straight from Gate 2.")


SHORT_ALIASES = {
    "binding_site": "-b",
    "dock": "-d",
    "alpha_mask" : "-am",
    "lining_refine": "-lr",
    "search_refine": "-sr",
    "ion_template_search": "-its",
    "orchestrate": "-o",
}


REGISTRY = {
    "binding_site": ProtocolSpec(
        name="binding_site",
        class_path="ChemEM.protocols.binding_site:BindingSite",
        deps=binding_site_deps,
        add_args=add_binding_site_args,
        help="Prepare/identify binding site",
    ),
    
    "confidence_map": ProtocolSpec(
        name="confidence_map",
        class_path="ChemEM.protocols.confidence_map:ConfidenceMap",
        deps=confidence_map_deps,
        add_args=add_confidence_map_args,
        help="FDR Confidence map",
    ),
    
    "alpha_mask": ProtocolSpec(
        name="alpha_mask",
        class_path="ChemEM.protocols.alpha_mask:AlphaMask",
        deps=alpha_mask_deps,
        add_args=add_alpha_mask_args,
        help="Segment ligand density",
    ),
    
    "dock": ProtocolSpec(
        name="dock",
        class_path="ChemEM.protocols._docking.docking:Docking",
        deps=dock_deps,
        add_args=add_dock_args,
        help="Dock ligands into the binding site",
    ),
    "refine": ProtocolSpec(
        name="refine",
        class_path="ChemEM.protocols.refine.minimize:Refine",
        deps=refine_deps,
        add_args=add_refine_args,
        help="MD-Refine ligand to density map"),
    "search_refine": ProtocolSpec(
        name="search_refine",
        class_path="ChemEM.protocols.refine.search_refine:SearchRefine",
        deps=search_refine_deps,
        add_args=add_search_refine_args,
        help="Iterative SCI-guided trust-region MD refinement from input conformer"),

    "ion_template_search": ProtocolSpec(
        name="ion_template_search",
        class_path="ChemEM.protocols.refine.ion_template_search:IonTemplateSearch",
        deps=ion_template_search_deps,
        add_args=add_ion_template_search_args,
        help="Mine PDB templates for metal coordination and prepare confidence-gated IonFixer inputs"),
    
    "ion_fixer": ProtocolSpec(
        name="ion_fixer",
        class_path="ChemEM.protocols.refine.ion_fixer:IonFixer",
        deps=ion_fixer_deps,
        add_args=add_ion_fixer_args,
        help="Refine Ion Cordination in cryoEM maps"),
    
    "mapq_score": ProtocolSpec(
        name="mapq_score",
        class_path="ChemEM.protocols.mapQ_score.mapQ_score:ScoreMapQ",
        deps=mapq_score_deps,
        add_args=add_mapq_score_args,
        help="Score Ligand MapQ",
    ),
    "lining_refine": ProtocolSpec(
        name="lining_refine",
        class_path="ChemEM.protocols.refine.lining_refine:LiningRefine",
        deps=lining_refine_deps,
        add_args=add_lining_refine_args,
        help="Refine pocket-lining sidechains out of ligand-density regions",
    ),
    "export" : ProtocolSpec(
        name="export",
        class_path="ChemEM.protocols.export_simulation.export_simulation:ExportSimulation",
        deps=export_deps,
        add_args=add_export_args,
        help="Export Simulation parameters",
    ),
    "orchestrate": ProtocolSpec(
        name="orchestrate",
        class_path="ChemEM.protocols.orchestrator:SmartOrchestrator",
        deps=orchestrate_deps,
        add_args=add_orchestrate_args,
        help="Smart 3-gate funnel: dock -> qscore-triage -> refine -> qscore+mmgbsa -> search_refine -> assemble",
    ),


}
