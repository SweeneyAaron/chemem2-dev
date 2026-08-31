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


import argparse
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

    g.add_argument("--protein-hydrogens", choices=("rdkit", "prep"), default="rdkit",
                   help="Where the protein hydrogen coordinates the ECHO H-bond term "
                        "reads come from. The site's RDKit mol is heavy-atom only, so "
                        "'rdkit' (default) rebuilds them with Chem.AddHs(addCoords=True) "
                        "and the prepared placement is lost: every freely rotatable donor "
                        "(Ser OG, Thr OG1, Tyr OH, Cys SG, Lys NZ) ends up at an "
                        "arbitrary torsion, and the H-bond term gates on the D-H...A "
                        "angle, so a real hydrogen bond pointing the wrong way scores as "
                        "a repulsive contact instead. 'prep' uses the hydrogens protein "
                        "preparation actually placed and minimised. Still ligand-blind, "
                        "but measured rather than guessed. Not the default because it "
                        "moves every ECHO score and the fitted weights were derived "
                        "against the RDKit placement.")

    # --- Manual (centroid-defined) binding site ---
    g.add_argument("--manual-site", action="store_true",
                   help="Build one binding site per config 'centroid =' with an explicitly "
                        "controlled extent, INSTEAD of alpha-shape pocket detection. The "
                        "requested volume minus solvent-excluded space becomes the search "
                        "region, so the site no longer depends on pocket detection or on "
                        "density segmentation splitting a pocket into several sites. "
                        "Alpha-mask may still attach density to these sites but may not "
                        "create new ones. Incompatible with --alpha-feature-sites and "
                        "--force-new-site, which both discard existing sites.")
    g.add_argument("--manual-site-radius", type=float, default=12.0, metavar="Å",
                   help="Radius of the --manual-site sphere. Default 12 Å. Ignored if "
                        "--manual-site-box is given.")
    g.add_argument("--manual-site-box", type=float, nargs="+", default=None, metavar="Å",
                   help="Extent of the --manual-site box in Å: one value for a cube, or "
                        "three for x y z. Takes precedence over --manual-site-radius. The "
                        "extent you ask for is the extent the search gets -- it is not "
                        "eroded by the probe radius.")


def confidence_map_deps(args):
    return tuple() 

def add_confidence_map_args(p):
    return tuple()

def dock_deps(args):
    # Dock always needs binding_site first 
    return ("binding_site", "alpha_mask","confidence_map")

def add_dock_args(p):
    g = p.add_argument_group("Docking")
    
    # Post-docking MM-GBSA of the poses --dock just produced. Renamed because bare
    # `--rescore` was routinely confused with the old `--rescore-poses` (ECHO), which
    # is a different thing entirely; `--rescore` is kept as a deprecated alias.
    g.add_argument("--dock-rescore-mmgbsa", "--rescore", dest="rescore",
                   action="store_true",
                   help="After docking, score the generated poses with a single-frame "
                        "MM-GBSA and write mmgbsa_rescore.txt. To score poses from a "
                        "file instead, use --score --score-with mmgbsa.")
    g.add_argument("--dock-seed", type=int, default=None,
                   help="Base seed for the ACO search RNG. Unset (default) draws a "
                        "fresh random seed each run, which is always logged as "
                        "'[dock] seed: N' so any run can be reproduced by passing it "
                        "back. The docking search is stochastic, and a fixed seed "
                        "makes one trajectory look like a converged answer -- rerun "
                        "with different seeds to see the real spread. The seed fully "
                        "determines the result: ants are seeded from (seed, iteration, "
                        "ant index), so it is independent of --ncpu and of ligand "
                        "order. One seed is drawn per run and shared by every site "
                        "and ligand in it.")
    g.add_argument("--echo-lattice-anchor", choices=("off", "global", "centroid"),
                   default="off",
                   help="Anchor the ECHO grid lattice so its phase does not depend on "
                        "the whole-protein bounding box. By default the grid origin is "
                        "min(all atoms) - padding, so an atom at the protein's edge -- "
                        "often a rebuilt, poorly-determined one tens of Angstrom from "
                        "the site -- slides the sampling lattice and changes "
                        "electro_attractive through trilinear interpolation. 'global' "
                        "snaps the origin to the absolute lattice {i*spacing}; "
                        "'centroid' makes the binding-site centroid an exact lattice "
                        "node, so the grid follows the site under rigid translation. "
                        "Both shift absolute ECHO scores by a small constant, which "
                        "would require refitting the ECHO weights -- hence off by "
                        "default. Prefer 'global' unless the config gives an explicit "
                        "centroid, since a derived centroid is segmentation-dependent.")
    g.add_argument("-fr", "--flexible-rings", action="store_true")
    g.add_argument("-ss", "--split-site", action="store_true")
    g.add_argument("-np", "--no-para", action="store_true")
    g.add_argument("--n-global-search", type=int, default=2000) #8000
    g.add_argument("--n-local-search", type=int, default=20) #change here
    g.add_argument("--local-minimiser", "--local-minimizer",
                   choices=("nelder-mead", "lbfgs"), default="nelder-mead",
                   help="Local minimiser used to refine poses, both in the "
                        "per-iteration refine of the top --n-local-search ants and "
                        "in the final polish. 'nelder-mead' (default) runs the staged "
                        "simplex: the 6 rigid-body dims are optimised to convergence, "
                        "frozen, and only then are the torsions optimised, so coupled "
                        "slide+torsion motions are unreachable. 'lbfgs' runs one joint "
                        "projected L-BFGS over all 6+nTors dims using central "
                        "finite-difference gradients. Try 'lbfgs' when a larger "
                        "--n-local-search finds the pose but is too slow: a gradient "
                        "costs 2*D evaluations, so it pays off only when it converges "
                        "in far fewer iterations than the simplex burns. Set "
                        "CHEMEM_DOCK_PROFILE=1 to print evals/refine for either.")
    g.add_argument("-br", "--bias-radius", type=float, default=12.0)
    g.add_argument("--cluster-docking", type=float, default=1.0)
    g.add_argument("--energy-cutoff", type=float, default=3.0)
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
    g.add_argument("--inner-map-score", type=int, default=1, choices=(0, 1),
                   help="Which map score drives the search (ant sampling + inner "
                        "Nelder-Mead refine): 0 = mutual information, 1 = SCI (default). "
                        "Scale with --mi-weight / --sci-weight respectively.")
    g.add_argument("--outer-map-score", type=int, default=0, choices=(0, 1),
                   help="Which map score drives the final polish, i.e. the score the "
                        "returned poses are RANKED by: 0 = mutual information (default), "
                        "1 = SCI. Set both --inner-map-score and --outer-map-score to the "
                        "same value to use one map term throughout.")
    g.add_argument("--dock-full-map", action="store_true",
                   help="Score the docking map term (MI/SCI) against the FDR confidence map "
                        "cropped to the binding-site box, instead of the alpha-masked / "
                        "blob-segmented site map. The crop is bit-exact on the same grid. "
                        "Also makes sites that segmentation dropped entirely dockable again. "
                        "NOTE the full map contains protein density and is not on the same "
                        "amplitude scale as the segmented map (which is multiplied by a "
                        "boundary EDT), so --mi-weight/--sci-weight need recalibrating and "
                        "absolute dock scores are NOT comparable across modes. Makes "
                        "--refine-to-diff-map a no-op.")
    g.add_argument("--echo-weights", type=str, default=None,
                   help="Path to a JSON of ECHO per-term weights (ECHOWeights field names) "
                        "used to drive the ACO search; overrides the compiled default_v1.")
    g.add_argument("--fast-sample", action="store_true",
                   help="(--dock2 only) score the ant-sampling stage with fast per-atom vdW "
                        "affinity grids (trilinear interpolation) so n-global-search can be cranked; "
                        "the final poses are still refined/ranked with the full ECHO score. Also "
                        "switches the R2 refine to the cheap analytic grid L-BFGS minimiser.")
    g.add_argument("--grid-min-steps", type=int, default=25,
                   help="(--dock2 --fast-sample) analytic grid L-BFGS iterations in the R2 refine.")
    g.add_argument("--polish-from-refined", action="store_true",
                   help="(--dock2) seed the final polish from the R2-REFINED conformer instead of "
                        "re-deriving the pose from the seed's discrete solution. Without this the "
                        "refinement is scored and ranked on, then discarded: refinePoseGrid never "
                        "updates discSol, and the FD path writes back only a 30-degree-quantised "
                        "approximation, so the polish restarts from a coarser pose than the one "
                        "that was measured. Off by default (previous behaviour).")
    g.add_argument("--hbond-geometric-gate", action="store_true",
                   help="(--dock2) classify an H-bond by GEOMETRY (atom-type mask, donor/"
                        "acceptor roles, D-H...A angle > 110 deg, distance) instead of by the "
                        "SIGN of the H-bond polynomial. A short charge-assisted contact such "
                        "as N...OD2 at 2.27 A currently fails the val<0 test, so its atom is "
                        "never marked satisfied and it collects unsat_polar and "
                        "hphob_enc_gt_7_only_hpil_unsat ON TOP of the repulsion -- a double "
                        "penalty on a real hydrogen bond. This removes only that double "
                        "count: nonbond_rep is provably unchanged and no weights move. "
                        "Off by default.")
    g.add_argument("--torsion-refine-steps", type=int, default=0, metavar="N",
                   help="(--dock2 --fast-sample) run N iterations of a TORSION-ONLY L-BFGS "
                        "after the grid refine, with the 6 rigid-body DOFs pinned. Uses the "
                        "analytic torsion gradient already computed by grid_pose_gradient, so "
                        "it adds no scorer evaluations beyond its own iterations. Targets the "
                        "measured failure mode on flexible ligands: the anchor fragment docks "
                        "correctly while the distal tail stays 3-5 A out, which a coupled "
                        "rigid+torsion step cannot escape. 0 (default) = off.")
    g.add_argument("--map-lookup-channel", type=int, default=0, choices=(0, 1, 2, 3, 4, 5),
                   help="(--dock2) which density field --map-lookup-weight samples: "
                        "0=ccc0 smoothed local-CC (default), 1=ccc1 gradient, 2=ccc2 "
                        "Laplacian, 3=RAW density, 4=ccc0+ccc1+ccc2 (SCI-like). Measured "
                        "native-vs-decoy separation on 9DMU/GAD: raw density 1.87x "
                        "(|rho| 0.79) vs 1.06-1.12x (|rho| 0.42-0.53) for every CCC "
                        "variant -- the CCC maps are clamped to [0,1] and nearly flat. "
                        "5 = raw normalised by its own max: same discrimination as 3 but "
                        "amplitude-independent, so a weight tuned on one map transfers. "
                        "Raw values are ~6x larger, so rescale the weight accordingly.")
    g.add_argument("--map-lookup-full-weight", type=float, default=0.0, metavar="W",
                   help="(--dock2) weight on the SAME density term inside the FULL score, "
                        "i.e. the objective the R2 refine and the final polish rank by. "
                        "The search objective ranks the deposited pose 1st (rho +0.52) "
                        "while the ranking objective (MI x100) ranks it 6th (+0.43), so "
                        "this lets the ranker use the term that discriminates. Separate "
                        "from --map-lookup-weight; 0 (default) leaves score() unchanged.")
    g.add_argument("--map-lookup-weight", type=float, default=0.0, metavar="W",
                   help="(--dock2 --fast-sample) weight on a density term in the FAST sampling "
                        "score: the summed local-CC map value under the ligand heavy atoms, "
                        "trilinearly interpolated. Without it the ant stage and the grid L-BFGS "
                        "are completely blind to the density and only the final polish sees the "
                        "map, so the geometry is chosen before the density is ever consulted. "
                        "0 (default) reproduces the previous behaviour exactly. Calibrate with "
                        "echo_score_fast_vs_full; --sci-weight is NOT transferable (it scales a "
                        "mean of three channels, this is a sum of one).")
    g.add_argument("--pose-min-rmsd", type=float, default=0.0,
                   help="(--dock2) finer RMSD for deduping the RETURNED modes (0 -> use the search "
                        "rms cutoff). Lower keeps more distinct near-native poses. Pair with a large "
                        "--return-n for recall.")
    g.add_argument("--energy-range", type=float, default=0.0,
                   help="(--dock2) return only modes within this score of the best (Vina energy_range; "
                        "0 -> disabled). Calibrate to ECHO units (~6-8).")
    g.add_argument("--reference-ligand", type=str, default=None,
                   help="(--dock2, TEMP diagnostic) native ligand SDF (same atom order as the docked "
                        "ligand); logs best heavy-atom RMSD-to-native at the sampled/refined/returned "
                        "stages so you can see where a near-native pose is lost.")
    g.add_argument("--no-write-site-files", dest="write_site_files", action="store_false",
                   default=True,
                   help="Suppress binding-site .mrc/.pdb writes (used by the redock loop to avoid "
                        "clutter and the output-dir-must-exist error).")



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
    g.add_argument("--hysteresis-segment", action="store_true",
                   help="Use hysteresis thresholding: keep voxels above a low threshold "
                        "only if connected to a high-threshold seed. Captures weak density "
                        "attached to strong density while rejecting unconnected noise. "
                        "Ignored if --otsu-segment is also set.")
    g.add_argument("--hyst-low-k", type=float, default=2.0, metavar="K",
                   help="Low threshold for --hysteresis-segment = bg_mean + K*bg_std "
                        "(background stats from confidence_map). Default 2.0.")
    g.add_argument("--hyst-low-sigma-fallback", type=float, default=1.0, metavar="STD",
                   help="Fallback low-threshold sigma multiplier on masked-density std "
                        "when background stats are unavailable. Default 1.0.")
    g.add_argument("--random-walker-segment", action="store_true",
                   help="Use random-walker segmentation (skimage). Probabilistic, "
                        "edge-aware: solves an anisotropic diffusion problem to get "
                        "per-voxel ligand probability. Best for weakly-resolved density "
                        "attached to strong density. Ignored if --otsu-segment is set; "
                        "takes precedence over --hysteresis-segment.")
    g.add_argument("--rw-beta", type=float, default=130.0, metavar="BETA",
                   help="Random-walker edge-weight penalty on density normalized to [0,1]: "
                        "edge_weight = exp(-BETA*(density_diff)^2). Higher BETA = sharper "
                        "boundaries. Default 130.")
    g.add_argument("--rw-prob-threshold", type=float, default=0.5, metavar="P",
                   help="Cutoff on the ligand-class posterior. Voxels with "
                        "P(ligand|seeds) >= P are kept. 0.5 = argmax; raise toward "
                        "0.7-0.9 for stricter segmentation. Default 0.5.")
    g.add_argument("--rw-bg-sigma-k", type=float, default=0.0, metavar="K",
                   help="Background-seed threshold: voxels with binding_density <= "
                        "bg_mean + K*bg_std are seeded as background. K=0 means voxels "
                        "at or below the noise mean are background seeds. Set negative "
                        "(e.g. -1) for stricter background seeding. Default 0.")
    g.add_argument("--rw-mode", type=str, default="cg_j",
                   choices=("cg_j", "cg", "cg_mg", "bf"),
                   help="Linear-system solver for random_walker. cg_j (default) = "
                        "conjugate gradient w/ Jacobi preconditioner; fast and robust "
                        "for 3D volumes.")
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

    # --- Alpha-feature-as-site mode ---
    g = p.add_argument_group("Alpha mask: alpha-feature-as-site mode")
    g.add_argument("--alpha-feature-sites", action="store_true",
                   help="Treat each alpha-shape ligand-density feature as its own binding site. "
                        "Replaces BindingSite-protocol sites entirely. Per-site density submap is "
                        "masked to the feature blob; per-site EDT confines ACO to the blob.")
    g.add_argument("--feature-site-radius", type=float, default=6.0, metavar="Å",
                   help="Auto-tightened bias_radius for alpha-feature sites AND padding when "
                        "cropping the cryo-EM bbox around the blob. Default 6 Å.")
    g.add_argument("--feature-residue-dilation", type=float, default=6.0, metavar="Å",
                   help="Lining-residue cutoff for alpha-feature sites: residues with at least "
                        "one heavy atom within this distance of any non-zero blob voxel are kept. "
                        "Default 6 Å. Independent of --feature-site-radius.")
    g.add_argument("--feature-aco-dilation", type=float, default=0.0, metavar="Å",
                   help="Isotropic dilation (Å) of the alpha-feature blob used to build the ACO "
                        "translation-point mask. Voxels within this distance of any blob voxel "
                        "are also valid ligand-centroid placements. Set to 0 for strict in-blob "
                        "search. Does NOT affect the score-side blob-masked density. Default 1.5 Å.")


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

    # --global-k lives in the shared group in __main__.py: it now drives every
    # minimiser (refine, slr2, ion_fixer, dock, lining_refine), not just this one.

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

    
    
# --score: pose scoring. Both the dependency function and the option registration
# live in the protocol's own package, because the dependencies depend on which
# scorers --score-with selected. `ChemEM.protocols.score.cli` is stdlib-only for
# exactly this reason -- it is imported while the parser is being built.
def score_deps(args):
    from ChemEM.protocols.score.cli import score_deps as _score_deps
    return _score_deps(args)


def add_score_args(p):
    from ChemEM.protocols.score.cli import add_score_args as _add_score_args
    return _add_score_args(p)


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
    g.add_argument("--ion-type", type=str, help="Ion to place, e.g. MG or ZN. Optional when --ion-spec is given: the type is then inferred from the supplied ion.")
    g.add_argument("--coordination-geometry", type=str, default='Octahedral', help="Coordination geometry : Octahedral |Square Planar | linear | Trigonal Bipyramidal | Triganal Planer | Square Pyrimidal | Tetrahedral | Pentagonal Bipyrimidal")

    g.add_argument(
        "--ion-spec",
        dest="ion_spec",
        type=str,
        default=None,
        help=(
            "Atom specification for an ion ALREADY PRESENT in the input structure. "
            "When given, no new ion is placed; the coordination distances and angles "
            "are refined around this ion instead. "
            "Format example: A:ZN:301:ZN"
        ))

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
    g.add_argument("--lr-global-k", type=float, default=None,
                   help="Density-map global k for the local refinement. Overrides the "
                        "shared --global-k; both unset means 150.0.")
    g.add_argument("--lr-backbone-k", type=float, default=1000.0,
                   help="Soft positional pin k for non-flagged heavy atoms")
    g.add_argument("--lr-repel-k", type=float, default=500.0,
                   help="Strength of the pocket-repulsive force applied to flagged atoms")

def smart_ligand_refine2_deps(args):
    return tuple()

def add_smart_ligand_refine2_args(p):
    g = p.add_argument_group("Smart Ligand Refinement 2")
    g.add_argument(
        "--sr2-optimisation-score",
        "--sr2-optimization-score",
        dest="sr2_optimisation_score",
        type=str,
        nargs="+",
        default="qscore",
        metavar="SCORES",
        help="Comma-separated SmartRefine2 fit score(s): qscore, ccc, mi, sci",
    )
    g.add_argument(
        "--sr2-optimisation-weights",
        "--sr2-optimization-weights",
        dest="sr2_optimisation_weights",
        type=str,
        nargs="+",
        default=None,
        metavar="WEIGHTS",
        help="Comma-separated weights for --sr2-optimisation-score",
    )
    g.add_argument(
        "--sr2-acceptance-score",
        type=str,
        nargs="+",
        default="qscore",
        metavar="SCORES",
        help="Comma-separated SmartRefine2 acceptance score(s): qscore, ccc, mi, sci",
    )
    g.add_argument(
        "--sr2-acceptance-weights",
        type=str,
        nargs="+",
        default=None,
        metavar="WEIGHTS",
        help="Comma-separated weights for --sr2-acceptance-score",
    )
    g.add_argument(
        "--sr2-final-minimise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "(Default ON.) Run a single OpenMM map-biased polish after "
            "the SmartRefine2 loop finishes. Passing --no-sr2-final-minimise "
            "disables only the post-loop polish; --sr2-no-polish disables "
            "both this and the on-stall polish."
        ),
    )
    g.add_argument(
        "--sr2-polish-on-stall",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "(Default ON.) OpenMM polish runs when the SmartRefine2 "
            "search stalls (patience trip OR walker returns empty), "
            "before the score-driven kick fires. Passing "
            "--no-sr2-polish-on-stall disables only this trigger; "
            "--sr2-no-polish disables both this and the final polish."
        ),
    )
    g.add_argument(
        "--sr2-no-polish",
        action="store_true",
        default=False,
        help=(
            "Force-disable ALL OpenMM polish in SmartRefine2 — both the "
            "on-stall recovery polish and the post-loop final polish. "
            "With round-13 defaults both polishes are ON; this is the "
            "simplest way to turn the entire polish system off without "
            "touching the individual --(no-)sr2-polish-* flags."
        ),
    )
    g.add_argument(
        "--sr2-minimiser",
        choices=["standard", "fragment"],
        default="standard",
        help=(
            "SmartRefine2 OpenMM polish strategy (applies to pre/stall/final "
            "polish). 'standard' (default) = single density-biased local-refine "
            "minimisation. 'fragment' = iterative fragment-pinned minimiser: "
            "minimise, score per-semantic-block Q vs the input pose, accept if "
            "overall Q holds and no block drops more than --sr2-fragmin-block-tol; "
            "else pin the dropped blocks at their input positions, restart from "
            "input and re-minimise; revert to input after --sr2-fragmin-max-iters."
        ),
    )
    g.add_argument(
        "--sr2-fragmin-block-tol",
        type=float,
        default=0.1,
        help=(
            "Per-block Q-score drop tolerance for --sr2-minimiser fragment: a "
            "block dropping by up to this is acceptable if the overall Q holds. "
            "Default 0.1."
        ),
    )
    g.add_argument(
        "--sr2-fragmin-max-iters",
        type=int,
        default=4,
        help=(
            "Maximum fragment-pin iterations for --sr2-minimiser fragment before "
            "reverting to the input pose. Default 4."
        ),
    )
    g.add_argument(
        "--sr2-robust",
        action="store_true",
        default=False,
        help=(
            "Convenience bundle for the robust 'minimise then torsion-polish' "
            "pipeline: enables --sr2-pre-minimise and asserts the robust "
            "profile (selection=greedy, centroid-trust on r=5.0 k=0.4, "
            "envelope-gate on slack=0.15, freeze-block-qscore=0.7). Note the "
            "robust centroid radius (5.0) is looser than the standalone default "
            "(1.0). --sr2-robust additionally turns pre-minimisation ON, and "
            "overrides the individual guard flags."
        ),
    )
    g.add_argument(
        "--sr2-selection",
        type=str,
        choices=["greedy", "branches"],
        default="greedy",
        help=(
            "SmartRefine2 branch-candidate selection. 'greedy' (default) "
            "includes the current pose in the pool with a raw-score floor so "
            "an iteration cannot regress; 'branches' picks the best re-fit "
            "branch only and can regress. The end-of-loop no-regression gate "
            "applies regardless. Default greedy."
        ),
    )


    g.add_argument(
        "--sr2-tail-aware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "(Default ON.) Tail-aware bundle of SmartRefine2 search "
            "defaults: lookahead=0.0, max_keep=8, kick_tries=3, "
            "patience=6, root_tabu_size=4, clash_tradeoff_lambda=0.05, "
            "selection=greedy. Ligands whose longest semantic walk "
            "exceeds --sr2-tail-aware-rotor-threshold dihedrals (default "
            "12) additionally get coarse_keep_fraction=0.40 and "
            "angular_diversity_sectors=6. Any individual --sr2-branch-* "
            "flag passed explicitly overrides the bundle and bypasses "
            "the adaptive logic for that field. Pass --no-sr2-tail-aware "
            "to fall back to the pre-tuning strict-gate defaults."
        ),
    )
    g.add_argument(
        "--sr2-tail-aware-rotor-threshold",
        type=int,
        default=10,
        help=(
            "When --sr2-tail-aware is on, ligands whose longest semantic "
            "walk depth exceeds this many dihedrals receive a wider "
            "walker search (coarse_keep_fraction=0.40, "
            "angular_diversity_sectors=6). Shorter walks keep the tighter "
            "defaults (0.20 and 0) that work better for compact ligands. "
            "Pass --sr2-branch-coarse-keep-fraction or "
            "--sr2-branch-angular-diversity-sectors explicitly to bypass "
            "the adaptive logic for those fields."
        ),
    )

    # --- Performance / tuning knobs (defaults preserve current behaviour, except
    # --- qscore-candidate-dirs which drops 256 -> 128 for ~2x Q-score speedup) ---
    g.add_argument(
        "--sr2-qscore-candidate-dirs",
        type=int,
        default=128,
        help=(
            "Number of Fibonacci-sphere candidate directions used when sampling "
            "per-atom Q-scores during SmartRefine2. Lower = faster (~linear), "
            "higher = smoother estimate. Published Q-score uses 128."
        ),
    )
    g.add_argument(
        "--sr2-freeze-block-qscore",
        type=float,
        default=0.7,
        help=(
            "Freeze SmartRefine2 semantic blocks whose mean per-atom Qscore is "
            ">= this threshold: they stay part of the rigid frame and are never "
            "torsion-searched, so the walker only perturbs poorly-fit regions "
            "(e.g. an ambiguous phosphate tail) and cannot disturb an "
            "already-well-fit core. Set 0 to disable. Default 0.7."
        ),
    )
    g.add_argument(
        "--sr2-fit-in-map-max-steps",
        type=int,
        default=64,
        help="Maximum finite-difference gradient steps per fit_in_map call.",
    )
    g.add_argument(
        "--sr2-fit-in-map-early-stop-tol",
        type=float,
        default=None,
        help=(
            "If set, stop fit_in_map early when the mean relative objective "
            "improvement over the last 3 accepted steps falls below this tol. "
            "Default: disabled (preserves current behaviour)."
        ),
    )
    g.add_argument(
        "--sr2-branch-coarse-step-deg",
        type=float,
        default=15.0,
        help="branch_walker coarse dihedral sweep step size in degrees.",
    ) 
    g.add_argument(
        "--sr2-branch-max-keep-per-step",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "branch_walker beam width (candidates kept per torsion step). "
            "(default: 3; with --sr2-tail-aware: 8)"
        ),
    )

    # --- Opt-in search-quality knobs (all defaults reproduce current behaviour).
    g.add_argument(
        "--sr2-branch-coarse-keep-fraction",
        type=float,
        default=argparse.SUPPRESS,
        help=(
            "Fraction of best score used as the threshold to keep coarse "
            "dihedral candidates: keep iff score >= best - X * |best|. "
            "Lower X = stricter; 0.35-0.50 retains more candidates for "
            "tail-heavy ligands. "
            "(default: 0.20; with --sr2-tail-aware: 0.40)"
        ),
    )
    g.add_argument(
        "--sr2-branch-downstream-lookahead-weight",
        type=float,
        default=argparse.SUPPRESS,
        help=(
            "Add alpha * mean(Qscore of moved-but-not-frontier atoms) to each "
            "candidate score so early-walk decisions are biased toward "
            "dihedrals that swing the downstream chain toward density. "
            "0.0 = frontier-only behaviour; 0.5-1.0 helps long tails. "
            "(default: 0.0; with --sr2-tail-aware: 0.75)"
        ),
    )
    g.add_argument(
        "--sr2-branch-angular-diversity-sectors",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "If > 0, _select_beam buckets candidates into N angular sectors "
            "(by the last dihedral) and keeps at least one survivor per "
            "non-empty sector before filling remaining beam slots by score. "
            "0 = off; 6 is a reasonable starting value. "
            "(default: 0; with --sr2-tail-aware: 6)"
        ),
    )
    g.add_argument(
        "--sr2-branch-metropolis-temp",
        type=float,
        default=None,
        help=(
            "Optional Metropolis temperature for branch-walker beam selection. "
            "Default None = strict greedy. A positive value lets the beam "
            "swap in a worse candidate with prob exp(-(best - cand) / T_iter), "
            "where T_iter decays per outer iteration."
        ),
    )
    g.add_argument(
        "--sr2-branch-metropolis-decay",
        type=float,
        default=0.7,
        help=(
            "Temperature decay per outer iteration when "
            "--sr2-branch-metropolis-temp is set: T_iter = T0 * decay^iter."
        ),
    )

    # --- Score-driven kick (basin-hopping after the patience break).
    g.add_argument(
        "--sr2-kick-tries",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of basin-hopping kick attempts after the patience break. "
            "0 = off. A kick perturbs only torsions on the walks to blocks "
            "that are below the Q-score threshold AND not improving; "
            "well-fit regions are not touched. "
            "(default: 0; with --sr2-tail-aware: 3)"
        ),
    )
    g.add_argument(
        "--sr2-kick-qscore-threshold",
        type=float,
        default=0.5,
        help=(
            "Per-block Q-score below which a block is considered 'poor fit' "
            "and eligible for kick perturbation. Q-score ranges 0-1; ~0.5 is "
            "the conventional 'poor fit' cutoff."
        ),
    )
    g.add_argument(
        "--sr2-kick-stagnation-tol",
        type=float,
        default=1e-3,
        help=(
            "Maximum per-block Q-score improvement since the last iteration "
            "for a block to count as 'stagnating' and be kick-eligible. "
            "Default 1e-3 (essentially no improvement)."
        ),
    )
    g.add_argument(
        "--sr2-kick-jitter-deg",
        type=float,
        default=30.0,
        help=(
            "Uniform random jitter magnitude (degrees) applied to each "
            "torsion on a kick-eligible walk. Default 30 degrees."
        ),
    )
    g.add_argument(
        "--sr2-kick-seed",
        type=int,
        default=None,
        help=(
            "Optional RNG seed for the kick perturbations. Provide for "
            "reproducible benchmark runs; default None uses a fresh seed."
        ),
    )

    # --- Outer-loop controls.
    g.add_argument(
        "--sr2-patience",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of outer iterations without raw-score improvement before "
            "the refine loop terminates (or triggers a kick if "
            "--sr2-kick-tries > 0). "
            "(default: 3; with --sr2-tail-aware: 6)"
        ),
    )
    g.add_argument(
        "--sr2-root-tabu-size",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "Size of the rolling root-block tabu deque. 1 = exclude only the "
            "previous root. Larger values force the loop to visit other "
            "blocks as root before reusing one — useful when poorly-fit "
            "tail blocks have low Q-scores and would otherwise never be "
            "selected as root. "
            "(default: 1; with --sr2-tail-aware: 4)"
        ),
    )

    # --- Composite acceptance gate (round 2). Trades raw_score against
    # --- clash_penalty so the loop can accept poses with slightly lower
    # --- Qscore in exchange for much lower clash penalty, and vice versa.
    g.add_argument(
        "--sr2-clash-tradeoff-lambda",
        type=float,
        default=argparse.SUPPRESS,
        help=(
            "Outer-accepter trade-off coefficient. A pose is accepted iff "
            "(delta_raw - lambda * delta_clash_penalty) > "
            "min_score_improvement, so the algorithm can cross small clash "
            "barriers when the score signal is strong and conversely accept "
            "small raw regressions when clash penalty drops a lot. The "
            "picker uses the same composite to choose among branch "
            "candidates. "
            "(default: unset = strict two-ratchet gate; "
            "with --sr2-tail-aware: 0.05)"
        ),
    )
    g.add_argument(
        "--sr2-branch-clash-tradeoff-lambda",
        type=float,
        default=None,
        help=(
            "Optional in-beam trade-off coefficient for the branch walker's "
            "candidate ranking. Default None preserves the current "
            "(frontier_score, clash_count) ordering. When set, candidates "
            "are ranked by (frontier_score - lambda * clash_penalty), so "
            "the beam keeps poses that have slightly worse Qscore but "
            "substantially fewer clashes. Tuned independently of "
            "--sr2-clash-tradeoff-lambda because the walker uses Qscores "
            "(range 0-1) while the outer accepter uses raw_score with a "
            "potentially different scale. Suggested starting value ~0.1."
        ),
    )

    # --- Optional pre-minimisation: run one OpenMM local-refine minimisation on
    # --- each ligand BEFORE the search loop. Useful when the docked pose is
    # --- rough and the search would otherwise drift into neighbouring density.
    g.add_argument(
        "--sr2-pre-minimise",
        "--sr2-pre-minimize",
        dest="sr2_pre_minimise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run one OpenMM local-refine minimisation on each ligand BEFORE "
            "the SmartRefine2 search loop (same minimiser as "
            "--refine --local-refine). Helps when the docked pose is rough so "
            "the search starts from the correct density basin. Default OFF."
        ),
    )

    # --- Aliphatic-ring conformer sampling. The branch walker treats ring
    # --- systems as rigid 'core' blocks, so it cannot correct a wrong pucker on
    # --- a saturated ring. When enabled, poorly-fit aliphatic ring blocks get a
    # --- small library of alternative ring conformers injected as extra search
    # --- candidates (scaffold held fixed), scored and refit like torsion moves.
    g.add_argument(
        "--sr2-ring-flex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "(Default ON.) Sample alternative conformations of aliphatic "
            "(saturated, non-aromatic) ligand rings during SmartRefine2. Only "
            "fires when such a ring exists AND its block fits the density "
            "poorly, so it is a no-op for rigid/aromatic-only ligands. Pass "
            "--no-sr2-ring-flex to disable."
        ),
    )
    g.add_argument(
        "--sr2-ring-flex-confs",
        type=int,
        default=8,
        help=(
            "Number of alternative ring conformers generated per flexible "
            "aliphatic ring system when --sr2-ring-flex is on. Higher = more "
            "thorough but slower. Default 8."
        ),
    )
    g.add_argument(
        "--sr2-ring-flex-rmsd",
        type=float,
        default=0.3,
        help=(
            "Minimum ring-atom RMSD (Angstrom) between retained ring "
            "conformers; smaller keeps more (more similar) conformers. "
            "Default 0.3."
        ),
    )
    g.add_argument(
        "--sr2-exo-torsions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Exocyclic ring-branch torsions: swing a poorly-fit substituent "
            "attached to a ring (e.g. a phosphate tail off a ribose) about a "
            "ring bond. The normal torsion model cannot move the first branch "
            "atom attached to a ring (it lies on its only rotation axis), so "
            "without this the whole branch is stuck at a fixed attachment "
            "direction. Gated to fire only on branches that fit the density "
            "worse than the best block. Default ON; --no-sr2-exo-torsions to "
            "disable."
        ),
    )
    g.add_argument(
        "--sr2-exo-step-deg",
        type=float,
        default=20.0,
        help=(
            "Coarse angular step (degrees) for the exo-torsion sweep. Smaller = "
            "finer but ~linearly more Q-score evaluations. Default 20.0."
        ),
    )
    g.add_argument(
        "--sr2-exo-min-downstream",
        type=int,
        default=3,
        help=(
            "Minimum number of downstream heavy atoms an exocyclic substituent "
            "must have for an exo torsion to be created (skips trivial 1-2 atom "
            "caps). Default 3."
        ),
    )

    # --- Anti-drift guards (opt-in). Anchored to the starting ligand pose, they
    # --- stop the search gaining score by translating/flipping the ligand into a
    # --- neighbouring density blob. Both default OFF: current behaviour unchanged.
    g.add_argument(
        "--sr2-centroid-trust",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Down-trust SmartRefine2 candidates that move the ligand centroid "
            "far from the STARTING pose: a flat-bottom penalty (zero within "
            "--sr2-centroid-trust-radius, then --sr2-centroid-trust-k per "
            "Angstrom beyond) is folded into candidate selection and "
            "acceptance. Resists drift into neighbouring density (the main "
            "regression mode on flexible ligands). Default ON; pass "
            "--no-sr2-centroid-trust to disable."
        ),
    )
    g.add_argument(
        "--sr2-centroid-trust-radius",
        type=float,
        default=5.0,
        help=(
            "Flat-bottom radius (Angstrom) for --sr2-centroid-trust: centroid "
            "moves up to this distance from the start are unpenalised. Kept "
            "well inside the 2A correctness band so a correct pose is never "
            "penalised but drift toward the failure boundary is. Default 1.0."
        ),
    )
    g.add_argument(
        "--sr2-centroid-trust-k",
        type=float,
        default=0.4,
        help=(
            "Penalty (in raw QScore units) per Angstrom of ligand-centroid "
            "displacement beyond --sr2-centroid-trust-radius. At 0.4 a 3A "
            "drift costs ~0.8 QScore, larger than almost any achievable gain, "
            "so drift must be strongly density-justified. Default 0.4."
        ),
    )
    g.add_argument(
        "--sr2-envelope-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reject SmartRefine2 candidates that leave the starting density "
            "blob: a pose whose fraction of ligand atoms sitting in "
            "above-threshold density drops more than --sr2-envelope-slack "
            "below the start is rejected at acceptance and dropped from "
            "selection. Catastrophic-drift backstop; no-ops when no map is "
            "present. Default ON; pass --no-sr2-envelope-gate to disable."
        ),
    )
    g.add_argument(
        "--sr2-envelope-threshold-sigma",
        type=float,
        default=1.0,
        help=(
            "Density threshold for the --sr2-envelope-gate coverage measure, "
            "expressed in map standard deviations above the mean "
            "(threshold = map_mean + sigma * map_std). Default 1.0."
        ),
    )
    g.add_argument(
        "--sr2-envelope-slack",
        type=float,
        default=0.15,
        help=(
            "Allowed drop in density-coverage fraction (vs the starting pose) "
            "before --sr2-envelope-gate rejects a candidate. Slightly loose so "
            "legitimate tail re-fitting into weak density is not over-rejected. "
            "Default 0.15."
        ),
    )

def export_deps(args):
    return tuple()

def add_export_args(p):
    pass


def orchestrate_deps(args):
    return ("binding_site", "confidence_map", "alpha_mask")


def add_orchestrate_args(p):
    g = p.add_argument_group("Smart Orchestrator")
    g.add_argument("--orch-gate1-topk", type=int, default=5,
                   help="Per site, keep top-K candidates after Gate 1 ranking.")
    g.add_argument("--orch-gate2-topk", type=int, default=2,
                   help="Per site, keep top-K candidates after Gate 2 ranking.")
    g.add_argument("--orch-gate3-topk", type=int, default=1,
                   help="Per site, keep top-K final assignments at Gate 3 (default 1 = single "
                        "winner; >1 returns ranked alternative solutions per site).")
    g.add_argument("--orch-audit-mode",
                   choices=["full", "scores", "selected", "off"],
                   default="full",
                   help="Persist orchestrator audit outputs: full, scores, selected, or off.")
    g.add_argument("--orch-score-mode",
                   choices=["absolute", "coverage", "qscore", "evidence"],
                   default="evidence",
                   help="Pose ranking mode: absolute map fit, legacy coverage z-score, Q-score, or "
                        "'evidence' (benchmark-backed: within-case Q-score gating, energy only for the "
                        "which-ligand tie-break). See the evidence-mode knobs below.")
    g.add_argument("--orch-w-qscore", type=float, default=0.5,
                   help="Weight on Q-score in map-fit ranking.")
    g.add_argument("--orch-w-qtail", type=float, default=0.25,
                   help="Weight on low-tail Q-score in map-fit ranking.")
    g.add_argument("--orch-w-density-coverage", type=float, default=5.0,
                   help="Weight on site-density coverage in map-fit ranking.")
    g.add_argument("--orch-w-density-overlap", type=float, default=1.0,
                   help="Weight on density overlap in absolute map-fit ranking.")
    g.add_argument("--orch-w-density-precision", type=float, default=0.5,
                   help="Weight on ligand-density precision in legacy coverage ranking.")
    g.add_argument("--orch-w-density-ccc", type=float, default=1.0,
                   help="Weight on local density correlation in map-fit ranking.")
    g.add_argument("--orch-w-mmgbsa", type=float, default=0.5,
                   help="Weight on -z(mmgbsa deltaG) in legacy Gate 2/3 composite.")
    g.add_argument("--orch-density-threshold-frac", type=float, default=0.05,
                   help="Fraction of site-map max density used for coverage/precision masks.")
    g.add_argument("--orch-density-sci-mode",
                   choices=["auto", "on", "off"],
                   default="auto",
                   help="Compute density SCI diagnostics: auto, on, or off.")
    g.add_argument("--orch-shape-metrics",
                   choices=["off", "gate3", "all"],
                   default="gate3",
                   help="Compute diagnostic density-shape metrics: off, Gate 3 only, or all density-scored stages.")
    g.add_argument("--orch-mmgbsa-pose-window", type=float, default=0.15,
                   help="Map-fit score window where MMGBSA may choose between poses of the same ligand.")
    g.add_argument("--orch-min-assignment-score", type=float, default=3.25,
                   help="Reject final sites below this absolute assignment score.")
    g.add_argument("--orch-min-density-coverage", type=float, default=0.30,
                   help="Reject final sites below this density coverage.")
    g.add_argument("--orch-min-assignment-margin", type=float, default=0.15,
                   help="Reject final sites when the best ligand margin is below this value.")
    # --- evidence-mode gating (only used when --orch-score-mode evidence) ---
    g.add_argument("--orch-qscore-floor", type=float, default=0.25,
                   help="[evidence] Reject a site whose best-candidate Q-score is below this absolute "
                        "floor (keeps ~95%% of real sites).")
    g.add_argument("--orch-qscore-strong", type=float, default=0.5,
                   help="[evidence] A best-candidate Q-score at/above this is 'clearly real' and is "
                        "accepted regardless of the within-structure relative gate.")
    g.add_argument("--orch-qscore-accept-z", type=float, default=-0.5,
                   help="[evidence] Relative decoy gate: a MODERATE site (Q-score below --orch-qscore-"
                        "strong) is rejected if its Q-score z-scored across the structure's sites is "
                        "below this. Bypassed for structures with <3 sites or no spread. Raise toward "
                        "+0.5 for higher precision, lower toward -0.5 for higher recall.")
    g.add_argument("--orch-wl-energy", choices=["on", "off"], default="on",
                   help="[evidence] Use the MM-GBSA+overlap blend to disambiguate competing ligands at "
                        "a site (density_overlap fallback when MM-GBSA fails).")
    g.add_argument("--orch-wl-margin", type=float, default=0.05,
                   help="[evidence] Q-score gap within which competing ligands are 'ambiguous' and the "
                        "energy blend decides the winner.")
    g.add_argument("--orch-wl-energy-w", type=float, default=0.6,
                   help="[evidence] Weight on MM-GBSA vs density_overlap in the which-ligand blend "
                        "(0.6 = 0.6*MM-GBSA + 0.4*overlap, within-site rank).")
    g.add_argument("--orch-energy-reject", type=float, default=None,
                   help="[evidence] Optional: reject a site as a fittable decoy if its accepted "
                        "winner's MM-GBSA deltaG exceeds this (kcal/mol); unset = off (energy targeted "
                        "to the which-ligand tie-break only).")
    g.add_argument("--orch-expected-assignments", type=str, default=None,
                   help="Optional labelled assignments for audit evaluation, e.g. '7:1,19:1'.")
    g.add_argument("--orch-compute-density-sci", action="store_true",
                   help="Compatibility alias for --orch-density-sci-mode on.")
    g.add_argument("--orch-skip-mmgbsa", action="store_true",
                   help="Skip MMGBSA scoring at Gate 2/3.")
    g.add_argument(
        "--orch-final-refiner",
        choices=["smart_refine_2", "search_refine", "none"],
        default="smart_refine_2",
        help="Final post-Gate-2 refinement stage before Gate 3 selection.",
    )
    g.add_argument(
        "--orch-skip-final-refine",
        "--orch-skip-search-refine",
        dest="orch_final_refiner",
        action="store_const",
        const="none",
        help="Skip the final refinement stage; pick final assembly straight from Gate 2.",
    )


SHORT_ALIASES = {
    "binding_site": "-b",
    "dock": "-d",
    "dock2": "-d2",
    "alpha_mask" : "-am",
    "lining_refine": "-lr",
    "ion_template_search": "-its",
    "orchestrate": "-o",
    "smart_ligand_refine2" : "-slr2",
    "score": "-sc",
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
    # Experimental sampling-efficient engine, fully isolated from `dock` (separate
    # docking_v2 .so). Shares the Docking CLI flags (add_args=None → no duplicate
    # argparse options; the flags are registered once by the `dock` spec above).
    "dock2": ProtocolSpec(
        name="dock2",
        class_path="ChemEM.protocols._docking.docking_v2:DockingV2",
        deps=dock_deps,
        add_args=None,
        help="Dock ligands with the experimental v2 engine (L-BFGS + cluster-refine)",
    ),
    "refine": ProtocolSpec(
        name="refine",
        class_path="ChemEM.protocols.refine.minimize:Refine",
        deps=refine_deps,
        add_args=add_refine_args,
        help="MD-Refine ligand to density map"),
    
    "smart_ligand_refine2": ProtocolSpec(
        name="smart_ligand_refine2",
        class_path="ChemEM.protocols.smart_refine_2.smart_refine:SmartRefine2",
        deps=smart_ligand_refine2_deps,
        add_args=add_smart_ligand_refine2_args,
        help="Q-score/CCC-guided near-fit ligand repair with OpenMM geometry filtering"),

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
    
    # Replaces the old `mapq_score` and `rescore_poses` protocols and the two
    # unregistered score_poses/echo_terms scripts. Its deps() is the only one in the
    # registry that varies with the CLI flags: scoring with Q-score alone needs no
    # binding site and no segmentation, scoring with ECHO needs both.
    "score": ProtocolSpec(
        name="score",
        class_path="ChemEM.protocols.score.score_poses:ScorePoses",
        deps=score_deps,
        add_args=add_score_args,
        help="Score docked poses with ECHO, MM-GBSA and/or the map metrics "
             "(--score-with echo,mmgbsa,qscore,density)",
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
        help="Smart 3-gate funnel: dock -> qscore-triage -> refine -> qscore+mmgbsa -> smart_refine_2 -> assemble",
    ),


}
