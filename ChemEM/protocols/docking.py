# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from rdkit import Chem

from ChemEM import docking
from ChemEM.messages import Messages
from ChemEM.parsers.models import Ligand
from ChemEM.tools.mmgbsa_score import MMGBSAScore
from ChemEM.tools.precomputed_data import PreCompDataLigand, PreCompDataProtein
from ChemEM.tools.docking import energy_cutoff, write_results, dock_worker
from ChemEM.tools.ligand import  mol_with_positions
from ChemEM.tools.geometry import rmsd_cluster
from ChemEM.tools.pose_minimiser import PoseMinimiser


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
    def __init__(self, system):
        self.system = system
        self._site_results: List[SiteResult] = []
        self.lig_probs = []
        self.debug_prints = []
        
        self._run_started = None
        self._run_runtime_s = None
        self._site_runtimes_s: Dict[str, float] = {}
        self._poses_count: Dict[tuple[str, int], int] = {}
    
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
        ligands = self._precomupt_ligand_objects()
        
        for site_id, binding_site in self._iter_sites():
            
            precomp_site = PreCompDataProtein(
                binding_site,
                self.system,
                bias_radius=self.system.options.bias_radius,
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
            mmgbsa_scorer = MMGBSAScore(
                self.system.output,
                self.system.protein,
                self.system.ligand,
                self.system.platform,
            )
            mmgbsa_scorer.run()

        self._run_runtime_s = float(time.perf_counter() - t0)
        self.system.log(f"Docking total runtime {self._run_runtime_s}.")

        # write a summary into self.system.log
        self.log()

    def score_echo(self, precompute_data, mol):
        block = self._molblock_with_fallback(mol, precompute_data)
        score = docking.run_echo_score(precompute_data, block)
        
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

                poses = self._post_process(ligand, poses)

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
            (mol, PreCompDataLigand(mol, flexible_rings=self.system.options.flexible_rings))
            for mol in self.system.ligand
        ]

    def _iter_sites(self):
        if not self.system.options.no_map  and (self.system.density_map is not None):
            
            
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
            combined = precomp_site + precomp_lig
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

    
    def _minimize_and_rescore(self, site_id, precomp_site, mol, poses):
        if not self.system.options.minimize_docking:
            return poses
        
        binding_site = self.system.binding_sites[site_id]
        conf_map = getattr(self.system, "confidence_map", None)
        pm = PoseMinimiser(
            protein_structure=self.system.protein.complex_structure,
            ligand_structure=[mol.complex_structure],
            residues=binding_site.residues,
            density_map=conf_map,
            platform_name=getattr(self.system, 'platform', 'OpenCL'),
            protein_restraint='protein',
            pin_k=5000.0,
            localise=True 
        )
       
        
        all_pos = np.array([i[1] for i in poses])
        min_pos = pm.minimize_pose_list(all_pos)
        
        min_pos_scored = []
        for pos in min_pos:
            
            new_mol = mol_with_positions(mol.mol, pos)
            block = Chem.MolToMolBlock(new_mol, includeStereo=True, confId=0)
            score = docking.run_echo_score(precomp_site, block, rep_max=15.0)
            min_pos_scored.append((score, pos))
        
        min_pos_scored = sorted(min_pos_scored, key=lambda x: x[0])
        return self._post_process(mol, min_pos_scored)
    
   
    def _molblock_with_fallback(self, ligand, combined):
        try:
            if combined.break_bonds:
                self.system.log("Using flexible heterocyclic rings.")
                return Chem.MolToMolBlock(combined.break_bonds, includeStereo=True, confId=0)
        except Exception as e:
            self.system.log(f"Ring fragmentation failed: {e}")
            # TODO: reset site?
        return Chem.MolToMolBlock(ligand.mol, includeStereo=True, confId=0)

    def _dock_split_site(self, block, combined):
        """Handle split-site docking with optional parallelism."""
        jobs = list(enumerate(combined.split_site_translation_centroid_raidus))
        n_jobs = len(jobs)

        max_jobs = max(1, (os.cpu_count() or 1) // self.system.CPUS_PER_SITE)
        max_jobs = min(max_jobs, n_jobs)

        if max_jobs > 1 and not self.system.options.no_para:
            self.system.log(f"Running {max_jobs} split-site jobs in parallel.")
            with ProcessPoolExecutor(max_workers=max_jobs) as pool:
                futs = [
                    pool.submit(
                        dock_worker,
                        block,
                        combined,
                        centroid,
                        radius,
                        self.system.CPUS_PER_SITE,
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
                results.extend(self.dock(block, local))

        return sorted(results, key=lambda r: r[0])

    def dock(self, block, precomp_echo):
        docked_solutions = docking.run_aco_docking(precomp_echo, block)
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
