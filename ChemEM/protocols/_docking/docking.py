# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

import itertools
import json
import os
import secrets
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from rdkit import Chem

# NOTE: the compiled `docking` extension is imported lazily via Docking._echo_ext()
# (not at module load) so the isolated experimental engine `docking_v2` — which
# subclasses Docking — can be imported in the same process without a pybind11
# double-registration clash on shared type names (e.g. ECHOWeights). --dock behaviour
# is unchanged: the same module still backs every call, just loaded on first use.
from ChemEM.messages import Messages
from ChemEM.parsers.models import Ligand #stay

from ChemEM.tools.precomputed_data import PreCompDataLigand, PreCompDataProtein #move 
from ChemEM.tools.docking import energy_cutoff, write_results, dock_worker#move
from ChemEM.tools.ligand import  mol_with_positions #stay
from ChemEM.tools.geometry import rmsd_cluster #stay
from ChemEM.tools.resources import (
    resolve_cpu_budget,
    resolve_cpus_per_site,
    thread_limit_env,
)
#refactored
from .mmgbsa_score import mmgbsa_score_docked_poses, write_mmgbsa_scores
from ChemEM.protocols.refine.pose_minimiser import BatchPoseMinimizer, ChemEMSimulationSetup
from ChemEM.protocols.core.simulation import resolve_global_k, resolve_implicit_solvent
from openmm import app

#----refactor notes
#going to move mmgbsa rescoreing out of this file 
#need to stop the finla assign sites
# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DockResult:
    score: float
    position: np.ndarray  # (N_atoms × 3)


@dataclass
class SiteResult:
    site_id: str
    ligand_idx: int
    ligand: Ligand
    poses: List[tuple[float, np.ndarray]]  # list of (score, coords)


class Docking:
    # Which compiled engine this protocol drives. Surfaced in the docking summary so a
    # log.out can be attributed to --dock or --dock2 after the fact; without it the two
    # produce byte-identical headers and an A/B comparison cannot be reconstructed.
    ENGINE_NAME = "docking"

    def __init__(self, system):
        self.system = system
        self._site_results: List[SiteResult] = []
        self.lig_probs = []
        self.debug_prints = []
        
        self._run_started = None
        self._run_runtime_s = None
        self._site_runtimes_s: Dict[str, float] = {}
        self._poses_count: Dict[tuple[str, int], int] = {}
        self._dock_seed = None
    
    def log(self) -> str:
        """
        Write a human-readable summary of what this Docking instance did to self.system.log.

        Supports:
          - self.system.log as a list (append)
          - self.system.log as a file path (str/PathLike) (append to file)
          - self.system.log as a file-like object with .write()
        Returns the summary string.
        """
        opts = getattr(self.system, "options", None)

        n_sites = len({sr.site_id for sr in self._site_results})
        n_ligs = len(getattr(self.system, "ligand", []) or [])
        n_pairs = len(self._site_results)

        # Best scores (lower is better) per site and overall
        best_overall = None
        best_by_site: Dict[str, float] = {}
        for sr in self._site_results:
            if sr.poses:
                site_best = float(min(p[0] for p in sr.poses))
                best_by_site[str(sr.site_id)] = min(best_by_site.get(str(sr.site_id), site_best), site_best)
                best_overall = site_best if best_overall is None else min(best_overall, site_best)

        lines = []
        lines.append("=== ChemEM Docking Summary ===\n")
        lines.append(f"Engine: {self.ENGINE_NAME}\n")
        if self._dock_seed is not None:
            lines.append(f"Search seed: {self._dock_seed}  (reproduce with --dock-seed {self._dock_seed})\n")
        lines.append(f"Output dir: {getattr(self.system, 'output', None)}\n")
        lines.append(f"Sites docked: {n_sites}\n")
        lines.append(f"Ligands provided: {n_ligs}\n")
        lines.append(f"Site×Ligand runs: {n_pairs}\n")

        if self._run_runtime_s is not None:
            lines.append(f"Total runtime (s): {self._run_runtime_s:.3f}\n")

        # Options snapshot (only if present)
        if opts is not None:
            lines.append("Options:\n")
            for k in (
                "split_site",
                "no_para",
                "minimize_docking",
                "rescore",
                "cluster_docking",
                "energy_cutoff",
                "aggregate_sites",
                "bias_radius",
                "flexible_rings",
            ):
                if hasattr(opts, k):
                    lines.append(f"  - {k}: {getattr(opts, k)}\n")

        # Per-site runtimes
        if self._site_runtimes_s:
            lines.append("Per-site runtimes (s):\n")
            for site_id, rt in sorted(self._site_runtimes_s.items(), key=lambda x: x[0]):
                lines.append(f"  - {site_id}: {rt:.3f}\n")

        # Pose counts after post-processing
        if self._poses_count:
            lines.append("Pose counts after post-processing (per site×ligand):\n")
            for (site_id, lig_idx), nposes in sorted(self._poses_count.items(), key=lambda x: (x[0][0], x[0][1])):
                lines.append(f"  - site {site_id} / ligand {lig_idx}: {nposes}\n")

        # Best scores
        if best_overall is not None:
            lines.append(f"Best overall score: {best_overall:.6f}\n")
        if best_by_site:
            lines.append("Best score per site:\n")
            for site_id, sc in sorted(best_by_site.items(), key=lambda x: x[0]):
                lines.append(f"  - {site_id}: {sc:.6f}\n")

        summary = "".join(lines)

        self.system.log(summary)

    def run(self):
        
        t0 = time.perf_counter()
        
        self._run_started = time.time()
        
        self.system.log(Messages.create_centered_box("Molecular Docking"))
        self.system.log(f"[dock] engine: {self.ENGINE_NAME}")
        # Resolve the search seed BEFORE any PreCompDataProtein is built, and log it
        # immediately -- a run that crashes later must still record how to repeat it.
        self._resolve_seed()
        self._warn_unsupported_engine_features()

        ligands = self._precomupt_ligand_objects()
        
        
        for site_id, binding_site in self._iter_sites():

            # Alpha-feature sites override bias_radius with --feature-site-radius so the
            # ACO translation-point sphere and the bias penalty match the per-site mask.
            if getattr(binding_site, "is_alpha_feature_site", False):
                effective_bias_radius = float(self.system.options.feature_site_radius)
            else:
                effective_bias_radius = float(self.system.options.bias_radius)

            precomp_site = PreCompDataProtein(
                binding_site,
                self.system,
                bias_radius=effective_bias_radius,
                split_site=self.system.options.split_site,
            )
            
            t1 = time.perf_counter()
            self._dock_site(site_id, precomp_site, ligands)
            
            
            rt = time.perf_counter() - t1
            self._site_runtimes_s[str(site_id)] = float(rt)
            self.system.log(f"Docking site {site_id} runtime {rt}.")

        system_output = self._aggregate_sites(mode=1, write=self.system.options.aggregate_sites)
        self.system.docked = True

        for lig_idx, docking_results in system_output.items():
            docked = [DockResult(score, position) for score, position in docking_results]
            self.system.ligand[lig_idx].docked = docked
        
        
        if self.system.options.rescore:
            
            mmgbsa_scores = self.score_mmgbsa()
            
            write_mmgbsa_scores(mmgbsa_scores, os.path.join(self.system.output, 'mmgbsa_rescore.txt'))
            

        self._run_runtime_s = float(time.perf_counter() - t0)
        self.system.log(f"Docking total runtime {self._run_runtime_s}.")

        # write a summary into self.system.log
        self.log()
    
    
    
    def _resolve_seed(self):
        """Fix the ACO search seed for this run and write it back onto options.

        The seed is drawn ONCE here and stored on `system.options.dock_seed`, which
        is where `PreCompDataProtein` reads it, so every site and every ligand in the
        run shares one seed and the run reproduces as a whole.

        Unset means random. The search is stochastic, and a fixed seed hides that:
        rerunning replays the identical trajectory, so a lucky pose looks converged.
        Reproducibility is preserved by logging the seed rather than by freezing it.
        """
        seed = getattr(self.system.options, "dock_seed", None)
        if seed is None:
            # 63-bit: stays positive through the int64 -> uint64 cast into C++.
            seed = secrets.randbits(63)
            source = "random"
        else:
            seed = int(seed)
            source = "--dock-seed"
        if seed == 0:
            # 0 is a legal mt19937 seed, but reserving it keeps "unset" unambiguous
            # everywhere the value is passed around as a falsy-checked int.
            seed = 1
        self.system.options.dock_seed = seed
        self._dock_seed = seed
        self.system.log(
            f"[dock] seed: {seed} ({source}) -- reproduce this run with --dock-seed {seed}"
        )
        minimiser = str(getattr(self.system.options, "local_minimiser", "nelder-mead"))
        if minimiser != "nelder-mead":
            self.system.log(
                f"[dock] local minimiser: {minimiser} (inner refine + final polish)"
            )
        return seed

    def score_mmgbsa(self):
        all_scores = []
        
        for ligand in self.system.ligand:
            all_scores +=  mmgbsa_score_docked_poses(ligand, self.system.protein)
            
        return all_scores

    def score_echo(self, precompute_data, mol):
        block = self._molblock_with_fallback(mol, precompute_data)
        score = self._echo_ext().run_echo_score(precompute_data, block, weights=self._weights())

        return score

    def _get_output_objects(self):
        pass

    def _aggregate_sites(self, mode=1, write=True):
        """
        TODO: refactor (currently does too much)
        mode 1 == by ligand
        mode 2 == by site
        """
        pose_dic: Dict[int, List[tuple[float, np.ndarray]]] = {}

        if mode == 1:
            ligand_idxs = {i.ligand_idx for i in self._site_results}

            for idx in ligand_idxs:
                ligand = self.system.ligand[idx]

                poses = [i.poses for i in self._site_results if i.ligand_idx == idx]
                poses = list(itertools.chain.from_iterable(poses))
                poses = sorted(poses, key=lambda x: x[0])

                #poses = self._post_process(ligand, poses)

                if write:
                    outdir = os.path.join(self.system.output, "aggregate_sites", f"Ligand_{idx}")
                    write_results(ligand, poses, outdir)

                pose_dic[idx] = poses

        elif mode == 0:
            # TODO: aggregate by site
            pass

        return pose_dic

    def _precomupt_ligand_objects(self):
        
        
        return [
            (mol, PreCompDataLigand(
                mol,
                self.system.platform,
                flexible_rings=self.system.options.flexible_rings,
                resource_owner=self.system,
            ))
            for mol in self.system.ligand
        ]

    def _iter_sites(self):
        if not self.system.options.no_map  and (self.system.density_map is not None):

            # --dock-full-map scores against the cropped confidence map, which is built
            # on demand from the BindingSite geometry, so a site that segmentation
            # dropped is still dockable and must not be filtered out here. --manual-site
            # is the same situation: the user defined the volume, so it gets docked
            # whether or not a density blob happened to land in it.
            if (getattr(self.system.options, "dock_full_map", False)
                    or getattr(self.system.options, "manual_site", False)):
                return [(key, binding_site)
                        for key, binding_site in self.system.binding_sites.items()]

            sites = [
                (key, binding_site)
                for key, binding_site in self.system.binding_sites.items()
                if (binding_site.key in self.system.binding_site_maps.keys())
            ]
        else:
            sites = [(key, binding_site) for key, binding_site in self.system.binding_sites.items()]

        return sites

    def _dock_site(self, site_id: str, precomp_site, ligands):
        self.debug_prints = []

        for lig_idx, (ligand, precomp_lig) in enumerate(ligands):
            print(f'dock ligand {lig_idx}')
            combined = precomp_site + precomp_lig

            # Covalent docking: tether the warhead to the protein anchor. Gated on
            # ligand.covalent_link, so non-covalent ligands are completely unaffected.
            if getattr(ligand, "covalent_link", None) is not None:
                self._apply_covalent_anchor(ligand, precomp_site, combined)

            self._attach_reference_ligand(ligand, combined)

            block = self._molblock_with_fallback(ligand, combined)
            
            if self.system.options.split_site:
                poses = self._dock_split_site(block, combined)
            else:
                poses = self.dock(block, combined)
                

            poses = self._post_process(ligand, poses)

            # track how many poses survived post-processing
            self._poses_count[(str(site_id), int(lig_idx))] = int(len(poses))

            outdir = os.path.join(
                self.system.output, "docking", f"binding_site_{site_id}_Ligand_{lig_idx}"
            )
            write_results(ligand, poses, outdir)
            
            
            if self.system.options.minimize_docking:
                
                poses = self._minimize_and_rescore(site_id, combined, ligand, poses)
                
                refined_out = os.path.join(
                    self.system.output, "refined", f"binding_site_{site_id}_Ligand_{lig_idx}"
                )
                
                write_results(ligand, poses, refined_out)

            self._site_results.append(
                SiteResult(site_id=site_id, ligand_idx=lig_idx, ligand=ligand, poses=poses)
            )
            print(f'finish dock ligand {lig_idx}')

    
    def _apply_covalent_anchor(self, ligand, precomp_site, combined):
        """Attach the covalent-anchor restraints (ligand warhead -> fixed protein
        atom, one per covalent bond) to the merged precompute object so the C++
        search tethers every warhead to its covalent residue. Only called for
        ligands carrying a covalent link; for every other ligand the object has no
        covalent_* attributes and the C++ scorer's covalent term stays disabled
        (existing docking unchanged).

        The attributes are lists — a crosslinker bonded to two residues is held at
        both ends. Convergence relies on the covalent penalties together with the
        binding site being centred on the covalent residues (see BindingSite
        covalent handling); the protein is rigid during the search so the anchors
        are fixed points.
        """
        from ChemEM.tools.precomputed_data import resolve_covalent_anchors
        resolved = resolve_covalent_anchors(
            ligand, precomp_site, self.system.protein.complex_structure
        )
        if not resolved:
            return

        combined.covalent_warhead_idx = [int(r[0]) for r in resolved]
        combined.covalent_anchor_prot_idx = [
            (None if r[1] is None else int(r[1])) for r in resolved
        ]
        combined.covalent_anchor_xyz = [np.asarray(r[2], dtype=float) for r in resolved]
        combined.covalent_r0 = [float(r[3]) for r in resolved]
        combined.covalent_k = [float(r[4]) for r in resolved]

        for warhead_idx, anchor_prot_idx, _xyz, r0, k in resolved:
            self.system.log(
                f"[covalent] docking anchor: warhead heavy#{warhead_idx} -> "
                + (f"prot#{anchor_prot_idx}" if anchor_prot_idx is not None else "xyz")
                + f" r0={r0:.3f} A k={k:.1f}"
            )

    def _minimize_and_rescore(self, site_id, precomp_site, mol, poses):
        if not self.system.options.minimize_docking:
            return poses

        if getattr(mol, "covalent_link", None) is not None:
            # Covalent ligands: the docked poses already have the warhead tethered
            # to the residue by the search. The OpenMM covalent bond is formed and
            # enforced in the dedicated refine / smart-refine step (which is
            # covalent-aware). Doing it here would require reconciling the full
            # docked pose with the leaving-group-stripped covalent topology, so we
            # skip the docking-internal minimise for covalent ligands.
            self.system.log(
                "[covalent] skipping docking-internal minimise; covalent bond is "
                "formed/enforced in refine/smart-refine."
            )
            return poses

        binding_site = self.system.binding_sites[site_id]
        binding_site_map = self.system.binding_sites[site_id]
        conf_map = getattr(self.system, "confidence_map", None)
        densmap = None
        if conf_map is not None and precomp_site.binding_site_density_map_grid is not None:

            # Under --dock-full-map the docking grid IS the cropped confidence map, so
            # reuse it here rather than re-cropping: the minimiser then provably sees the
            # same density the score used. (This makes --refine-to-diff-map a no-op.)
            use_dock_grid = (self.system.options.refine_to_diff_map
                             or getattr(self.system.options, "dock_full_map", False))

            if use_dock_grid:

                from ChemEM.parsers.EMMap import EMMap
                densmap = EMMap(precomp_site.binding_site_density_map_origin,
                                precomp_site.binding_site_density_map_apix,
                                precomp_site.binding_site_density_map_grid,
                                0.0)
            else:
                densmap = conf_map.submap(origin=precomp_site.binding_site_density_map_origin,
                                          box_size=precomp_site.binding_site_density_map_grid.shape)
            
        

        env = ChemEMSimulationSetup(
            protein_structure=self.system.protein.complex_structure,
            ligand_structure=[mol.complex_structure],
            residues=binding_site.residues,  # The setup class will automatically cut the subset!
            density_map=densmap,
            platform_name=getattr(self.system, 'platform', 'CPU'),
            protein_restraint='protein',
            pin_k=5000.0,
            localise=False,
            global_k=resolve_global_k(self.system.options, 150.0),
            solvent=resolve_implicit_solvent(self.system.options, app.GBn2),
            resource_owner=self.system,
        )


        minimizer = BatchPoseMinimizer(env)
        all_pos = np.array([i[1] for i in poses])
        min_pos = minimizer.run(
            ligand_poses_angstrom=all_pos,
            do_biased_md=getattr(self.system.options, 'do_biased_md', False),
            md_ps=getattr(self.system.options, 'md_ps', 5.0),
            max_iters=0 # Or getattr(self.system.options, 'max_iters', 0)
        )


        '''
        pm = PoseMinimiser(
            protein_structure=self.system.protein.complex_structure,
            ligand_structure=[mol.complex_structure],
            residues=binding_site.residues,
            density_map=densmap,
            platform_name=getattr(self.system, 'platform', 'CPU'),
            protein_restraint='protein',
            pin_k=5000.0,
            localise=False,
            do_biased_md=self.system.options.do_biased_md
        )
        
        
        all_pos = np.array([i[1] for i in poses])
        min_pos = pm.minimize_pose_list(all_pos)
        
        '''
        min_pos_scored = []
        for pos in min_pos:
            
            new_mol = mol_with_positions(mol.mol, pos)
            block = Chem.MolToMolBlock(new_mol, includeStereo=True, confId=0)
            score = self._echo_ext().run_echo_score(precomp_site, block, rep_max=15.0)
            min_pos_scored.append((score, pos))
        
        #min_pos_scored = sorted(min_pos_scored, key=lambda x: x[0])
        #return self._post_process(mol, min_pos_scored)
        return min_pos_scored
    
   
    def _molblock_with_fallback(self, ligand, combined):
        try:
            # Covalent ligands: never ring-fragment. The ring-opened mol
            # (combined.break_bonds) can reorder atoms, which would invalidate the
            # covalent warhead heavy-atom index resolved against ligand.mol, and the
            # warhead-bearing ring must stay rigid at the junction anyway.
            if getattr(ligand, "covalent_link", None) is not None:
                return Chem.MolToMolBlock(ligand.mol, includeStereo=True)
            if combined.break_bonds:
                self.system.log("Using flexible heterocyclic rings.")
                return Chem.MolToMolBlock(combined.break_bonds, includeStereo=True, confId=0)
        except Exception as e:
            self.system.log(f"Ring fragmentation failed: {e}")
            # TODO: reset site?
        #NEW generate diverse subset.
        
        #ensemble_mol = generate_systematic_ensemble(ligand.mol, max_output=300, rmsd_cutoff=1.0)
        #sdf_string = multi_conf_to_sdf_string(ensemble_mol)
        #sdf_string = multi_conf_to_sdf_string(ligand.mol)
        
        #return sdf_string
        #import pdb 
        #pdb.set_trace()
        
        return Chem.MolToMolBlock(ligand.mol, includeStereo=True)

    def _dock_split_site(self, block, combined):
        """Handle split-site docking with optional parallelism."""
        jobs = list(enumerate(combined.split_site_translation_centroid_raidus))
        n_jobs = len(jobs)

        total_cpus = resolve_cpu_budget(self.system)
        cpus_per_site = resolve_cpus_per_site(self.system, total_cpus=total_cpus)
        max_jobs = max(1, total_cpus // cpus_per_site)
        max_jobs = min(max_jobs, n_jobs)

        if max_jobs > 1 and not self.system.options.no_para:
            self.system.log(
                f"Running {max_jobs} split-site jobs in parallel "
                f"({cpus_per_site} CPU(s) per job, {total_cpus} total)."
            )
            with ProcessPoolExecutor(max_workers=max_jobs) as pool:
                futs = [
                    pool.submit(
                        dock_worker,
                        block,
                        combined,
                        centroid,
                        radius,
                        cpus_per_site,
                    )
                    for _, (centroid, radius) in jobs
                ]
                results = []
                for fut in as_completed(futs):
                    results.extend(fut.result())
        else:
            results = []
            for idx, (centroid, radius) in jobs:
                self.system.log(f"Docking split-site {idx} serially.")
                local = combined.copy()
                local.add_multi_site_bias(centroid, radius)
                results.extend(self.dock(block, local, ncpu=cpus_per_site))

        return sorted(results, key=lambda r: r[0])

    def _echo_ext(self):
        """The compiled ECHO docking/scoring extension backing this engine.

        Baseline `Docking` returns the `docking` module (imported lazily — see the
        module-header note). The experimental `DockingV2` overrides this to return the
        isolated `docking_v2` module, so every ext call (search, rescore, weights) is
        routed to the selected engine without touching the other.
        """
        from ChemEM import docking
        return docking

    def _weights(self):
        """Cached ECHOWeights from the --echo-weights JSON (falls back to default_v1).

        The JSON is {ECHOWeights-field: value}; unknown keys are ignored. Loaded once per
        Docking run and reused for the ACO search and any run_echo_score rescoring so the
        search objective and the ranking objective stay identical.
        """
        w = getattr(self, "_echo_weights", None)
        if w is None:
            w = self._echo_ext().ECHOWeights.default_v1()
            path = getattr(self.system.options, "echo_weights", None)
            if path:
                with open(path) as fh:
                    for k, v in json.load(fh).items():
                        if hasattr(w, k):
                            setattr(w, k, float(v))
                self.system.log(f"[dock] loaded ECHO weights from {path}")

            # Density lookup for the --fast-sample objective. Applied after the JSON so an
            # explicit --map-lookup-weight always wins. 0.0 (the default_v1 value) leaves the
            # fast score bit-identical; the field only exists on the docking_v2 extension.
            mlf = float(getattr(self.system.options, "map_lookup_full_weight", 0.0) or 0.0)
            if mlf and hasattr(w, "map_lookup_full"):
                w.map_lookup_full = mlf
                self.system.log(f"[dock] full-score density lookup enabled, weight {mlf}")

            mlw = float(getattr(self.system.options, "map_lookup_weight", 0.0) or 0.0)
            if mlw:
                if not hasattr(w, "map_lookup"):
                    self.system.log(
                        "[dock] --map-lookup-weight ignored: this engine has no map_lookup "
                        "channel (it is a docking_v2 addition; use --dock2).")
                else:
                    w.map_lookup = mlw
                    self.system.log(
                        f"[dock] fast-sample density lookup enabled, weight {mlw}"
                        + ("" if getattr(self.system.options, "fast_sample", False)
                           else "  -- NOTE: only the --fast-sample path uses it"))

            self._echo_weights = w
        return w

    def _warn_unsupported_engine_features(self):
        """Say so when the selected engine ignores something the run asked for.

        Baseline `docking` supports everything, so this is a no-op here; DockingV2
        overrides it. Silence is the dangerous case -- a covalent run that quietly
        docks non-covalently still produces plausible-looking poses.
        """
        return

    def _attach_reference_ligand(self, ligand, combined):
        """Hand the engine the native heavy-atom coords **in the docked atom order**.

        The engine's recall diagnostic (`rmsd_to_ref`) compares by atom INDEX -- no
        superposition, no symmetry correction, no atom mapping. A reference SDF is
        written in whatever order the depositor used (element-sorted, typically), while
        the docked ligand is built from SMILES. Handing the raw SDF coordinates over
        therefore reports a large RMSD for a perfect pose: on 9DMU/GAD, a pose identical
        to the native measures **7.70 A**, which is inside the range of real answers and
        reads as a plausible result. So the coordinates are permuted here first.

        Raises rather than logging-and-continuing: a diagnostic that is silently wrong
        is worse than one that is absent.
        """
        ref = getattr(self.system.options, "reference_ligand", None)
        if not ref:
            return

        refmol = Chem.SDMolSupplier(ref, removeHs=True)[0]
        if refmol is None:
            raise ValueError(f"--reference-ligand: no readable molecule in {ref}")
        refmol = Chem.RemoveHs(refmol)

        # AddHs appends hydrogens, so heavy atoms keep indices 0..n_heavy-1 in
        # ligand.mol -- the same range rmsd_to_ref walks.
        probe = Chem.RemoveHs(Chem.Mol(ligand.mol))
        matches = probe.GetSubstructMatches(refmol, uniquify=False, maxMatches=50000)
        if not matches:
            raise ValueError(
                f"--reference-ligand: {ref} is not a substructure of the docked ligand, "
                "so its atoms cannot be mapped onto the docked order. The recall "
                "diagnostic would be meaningless; fix the reference or drop the flag.")

        ref_xyz = np.asarray(refmol.GetConformer().GetPositions(), dtype=float)
        mapped = np.zeros((probe.GetNumAtoms(), 3), dtype=float)
        for ref_idx, probe_idx in enumerate(matches[0]):
            mapped[probe_idx] = ref_xyz[ref_idx]

        combined.reference_heavy_coords = mapped

        note = ""
        if len(matches) > 1:
            # Automorphic mappings (phosphate oxygens, ring flips, ...). The engine gets
            # one fixed array and cannot take a min over them, so the reported RMSD is an
            # UPPER BOUND -- fine for localising where recall is lost, not a final number.
            note = (f" ({len(matches)} symmetry-equivalent mappings; reported RMSD is an "
                    "upper bound)")
        self.system.log(
            f"[dock] recall diagnostic on: {os.path.basename(ref)} mapped onto "
            f"{probe.GetNumAtoms()} docked heavy atoms{note}")

    def dock(self, block, precomp_echo, ncpu=None):
        budget = ncpu if ncpu is not None else resolve_cpu_budget(getattr(self, "system", None))
        with thread_limit_env(budget):
            docked_solutions = self._echo_ext().run_aco_docking(precomp_echo, block, weights=self._weights())
        return sorted(docked_solutions, key=lambda x: x[0])

    def _post_process(self, ligand, poses):
        if self.system.options.cluster_docking:
            poses = rmsd_cluster(
                poses,
                rmsd_threshold=self.system.options.cluster_docking,
                n_heavy=ligand.mol.GetNumHeavyAtoms(),
            )
        if self.system.options.energy_cutoff:
            poses = energy_cutoff(poses, delta=self.system.options.energy_cutoff)
        return poses
#new gereate ensemble  
import itertools
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.ML.Cluster import Butina

def generate_systematic_ensemble(mol, max_output=300, rmsd_cutoff=1.0, strain_cutoff=150.0):
    """
    Generates a diverse ensemble of low-strain conformers by systematically 
    rotating all single bonds to their trans/gauche energy minima.
    """
    print(f"--- Starting Systematic Conformer Generation ---")
    
    # 1. Ensure molecule has 3D coordinates to start
    work_mol = Chem.Mol(mol)
    work_mol = Chem.AddHs(work_mol)
    if work_mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(work_mol, AllChem.ETKDG())
    
    # 2. Find Rotatable Bonds and Extract Dihedral 4-Atom Indices
    rot_smarts = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
    rot_matches = work_mol.GetSubstructMatches(rot_smarts)
    
    dihedrals = []
    for match in rot_matches:
        idx2, idx3 = match[0], match[1]
        a2 = work_mol.GetAtomWithIdx(idx2)
        a3 = work_mol.GetAtomWithIdx(idx3)
        
        # Get neighbors to define the A-B-C-D dihedral angle
        neighbors2 = [n.GetIdx() for n in a2.GetNeighbors() if n.GetIdx() != idx3]
        neighbors3 = [n.GetIdx() for n in a3.GetNeighbors() if n.GetIdx() != idx2]
        
        if neighbors2 and neighbors3:
            dihedrals.append((neighbors2[0], idx2, idx3, neighbors3[0]))

    num_bonds = len(dihedrals)
    print(f"Found {num_bonds} rotatable bonds.")
    
    # Safety Check: Combinatorial explosion
    if num_bonds > 10:
        print(f"WARNING: Too many bonds ({num_bonds}). Limiting systematic search to first 10 to prevent memory crash.")
        dihedrals = dihedrals[:10]
        num_bonds = 10

    # 3. Generate all combinations of Trans (180), Gauche+ (60), and Gauche- (300)
    angles = [60.0, 180.0, 300.0]
    combinations = list(itertools.product(angles, repeat=num_bonds))
    print(f"Testing {len(combinations)} systematic dihedral combinations...")

    # 4. Apply Angles and Filter by Internal Strain (Clashes)
    valid_confs = []
    conf = work_mol.GetConformer(0)
    
    # Pre-compute MMFF properties for fast scoring
    mp = AllChem.MMFFGetMoleculeProperties(work_mol)

    for combo in combinations:
        # Spin the bonds
        for d_idx, angle in zip(dihedrals, combo):
            rdMolTransforms.SetDihedralDeg(conf, d_idx[0], d_idx[1], d_idx[2], d_idx[3], angle)
            
        # Score the internal strain
        ff = AllChem.MMFFGetMoleculeForceField(work_mol, mp, confId=0)
        if ff is None: 
            continue
            
        energy = ff.CalcEnergy()
        
        # Prune massive clashes!
        if energy < strain_cutoff:
            # Extract coordinates to save this state
            pos = [conf.GetAtomPosition(i) for i in range(work_mol.GetNumAtoms())]
            valid_confs.append((energy, pos))

    print(f"Combinations surviving internal clash filter: {len(valid_confs)}")
    if not valid_confs:
        raise ValueError("All conformers clashed! Increase strain_cutoff.")

    # Sort surviving poses by internal energy (best to worst)
    valid_confs.sort(key=lambda x: x[0])
    # =========================================================================
    # THE FAST CLUSTERING FIX
    # =========================================================================
    
    # NEW: 1. Energy Cap (Never cluster more than 2,000 to avoid O(N^2) death)
    MAX_TO_CLUSTER = 2000
    if len(valid_confs) > MAX_TO_CLUSTER:
        print(f"Truncating from {len(valid_confs)} down to the top {MAX_TO_CLUSTER} lowest-energy conformers...")
        valid_confs = valid_confs[:MAX_TO_CLUSTER]

    # 2. Pack survivors into the Multi-Conformer RDKit Mol
    multi_mol = Chem.Mol(work_mol)
    multi_mol.RemoveAllConformers()
    
    for idx, (en, pos) in enumerate(valid_confs):
        new_conf = Chem.Conformer(multi_mol.GetNumAtoms())
        for i, p in enumerate(pos):
            new_conf.SetAtomPosition(i, p)
        multi_mol.AddConformer(new_conf, assignId=True)

    num_confs = multi_mol.GetNumConformers()
    print(f"Pre-aligning {num_confs} conformers...")
    
    # NEW: 3. Pre-align all conformers to the first one (Conformer 0)
    # This aligns their centers of mass instantly, eliminating the need to 
    # individually align every pair during the matrix calculation.
    AllChem.AlignMolConformers(multi_mol)

    print("Calculating C++ Optimized RMSD matrix...")
    # NEW: 4. Use the C++ backend to generate the matrix. 
    # Because we pre-aligned them, we set prealigned=True. This takes milliseconds!
    dists = AllChem.GetConformerRMSMatrix(multi_mol, prealigned=True)

    print("Running Butina Clustering...")
    # 5. Cluster using Butina
    clusters = Butina.ClusterData(dists, num_confs, rmsd_cutoff, isDistData=True)
    
    # 6. Extract the top N diverse cluster centroids
    final_mol = Chem.Mol(multi_mol)
    final_mol.RemoveAllConformers()
    
    cluster_centroids = [c[0] for c in clusters]
    keep_ids = cluster_centroids[:max_output]
    
    for k_id in keep_ids:
        final_mol.AddConformer(multi_mol.GetConformer(k_id), assignId=True)
        
    print(f"--- Ensemble Generation Complete: {final_mol.GetNumConformers()} diverse conformers packed ---")
    
    return final_mol
    '''
    # 5. Pack survivors into a single Multi-Conformer RDKit Mol
    multi_mol = Chem.Mol(work_mol)
    multi_mol.RemoveAllConformers()
    
    for idx, (en, pos) in enumerate(valid_confs):
        new_conf = Chem.Conformer(multi_mol.GetNumAtoms())
        for i, p in enumerate(pos):
            new_conf.SetAtomPosition(i, p)
        multi_mol.AddConformer(new_conf, assignId=True)

    # 6. Cluster by RMSD to remove identical shapes
    print("Clustering conformers by RMSD...")
    dists = []
    num_confs = multi_mol.GetNumConformers()
    for i in range(num_confs):
        for j in range(i):
            dists.append(AllChem.GetBestRMS(multi_mol, multi_mol, i, j))

    # Butina clustering algorithm
    clusters = Butina.ClusterData(dists, num_confs, rmsd_cutoff, isDistData=True)
    
    # 7. Extract the top N diverse cluster centroids
    final_mol = Chem.Mol(multi_mol)
    final_mol.RemoveAllConformers()
    
    cluster_centroids = [c[0] for c in clusters]
    keep_ids = cluster_centroids[:max_output]
    
    for k_id in keep_ids:
        final_mol.AddConformer(multi_mol.GetConformer(k_id), assignId=True)
        
    print(f"--- Ensemble Generation Complete: {final_mol.GetNumConformers()} diverse conformers packed ---")
    
    return final_mol
    '''

def multi_conf_to_sdf_string(mol):
    """Stitches all conformers of a molecule into a single SDF string."""
    sdf_blocks = []
    for conf in mol.GetConformers():
        # Export each conformer explicitly by ID
        block = Chem.MolToMolBlock(mol, confId=conf.GetId())
        sdf_blocks.append(block)
    
    # Join them with the standard SDF separator
    return "\n$$$$\n".join(sdf_blocks) + "\n$$$$\n"
