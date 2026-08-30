// bindings.cpp
#include <memory>
#include <stdexcept>
#include <string>
#include <chrono>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <GraphMol/RWMol.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/FileParsers/FileParsers.h>  // RDKit::v1::MolBlockToMol

#include "PreComputedData.h"
#include "ScoringFunctions.h"
#include "SearchFunctions.h"

namespace py = pybind11;

// Keep only one conformer so AntColonyOptimizer (which calls getConformer() with default)
// uses the conformer you chose.
static RDKit::RWMol keep_only_conformer(const RDKit::RWMol& in, int confId) {
    RDKit::RWMol m(in);

    const int n = static_cast<int>(m.getNumConformers());
    if (n == 0) {
        throw std::runtime_error("Mol has no conformers");
    }
    if (confId < 0 || confId >= n) {
        throw std::runtime_error("confId out of range");
    }

    for (int i = n - 1; i >= 0; --i) {
        if (i != confId) m.removeConformer(i);
    }

    // Optional: normalize remaining conformer id to 0 (not strictly necessary)
    m.getConformer(0).setId(0);
    return m;
}

PYBIND11_MODULE(docking_v2, m) {
    m.doc() = "ChemEM experimental docking engine v2 (sampling-efficient: L-BFGS local "
              "search + cluster-then-refine). Isolated from the baseline `docking` module.";

    // -----------------------------
    // ECHOWeights binding
    // -----------------------------
    py::class_<ECHOWeights>(m, "ECHOWeights")
        .def(py::init<>())
        .def_static("default_v1", &ECHOWeights::default_v1)
        .def_readwrite("aromatic", &ECHOWeights::aromatic)
        .def_readwrite("nonbond", &ECHOWeights::nonbond)
        .def_readwrite("saltbridge_raw", &ECHOWeights::saltbridge_raw)
        .def_readwrite("hbond_raw", &ECHOWeights::hbond_raw)
        .def_readwrite("ligand_intra", &ECHOWeights::ligand_intra)
        .def_readwrite("ligand_torsion", &ECHOWeights::ligand_torsion)
        .def_readwrite("electro_attractive", &ECHOWeights::electro_attractive)
        .def_readwrite("electro_repulsive_clamp", &ECHOWeights::electro_repulsive_clamp)
        .def_readwrite("desolvation_penalty_scaled", &ECHOWeights::desolvation_penalty_scaled)
        .def_readwrite("hphobe_raw_hpho", &ECHOWeights::hphobe_raw_hpho)
        .def_readwrite("hphobe_raw_hpil", &ECHOWeights::hphobe_raw_hpil)
        .def_readwrite("hphob_enc_gt_7_only_hpho", &ECHOWeights::hphob_enc_gt_7_only_hpho)
        .def_readwrite("hphob_enc_gt_7_only_hpil_unsat", &ECHOWeights::hphob_enc_gt_7_only_hpil_unsat)
        .def_readwrite("unsat_polar", &ECHOWeights::unsat_polar)
        .def_readwrite("aromatic_attr", &ECHOWeights::aromatic_attr)
        .def_readwrite("aromatic_clash", &ECHOWeights::aromatic_clash)
        .def_readwrite("nonbond_attr", &ECHOWeights::nonbond_attr)
        .def_readwrite("nonbond_rep", &ECHOWeights::nonbond_rep)
        .def_readwrite("clash", &ECHOWeights::clash)
        // fast-sampling density lookup; 0.0 in default_v1() so --fast-sample is unchanged
        .def_readwrite("map_lookup", &ECHOWeights::map_lookup)
        // same term inside the FULL score(), for the polish/ranking objective
        .def_readwrite("map_lookup_full", &ECHOWeights::map_lookup_full);

    // -----------------------------
    // Score helpers (updated to accept weights + actually use rep_max)
    // -----------------------------
    m.def(
        "run_echo_score",
        [](py::object py_pc,
           const std::string& molblock,
           int confId,
           double interaction_cutoff,
           double rep_max,
           double electro_clamp,
           ECHOWeights weights) -> double {

            PreComputedData pre(py_pc);

            std::unique_ptr<RDKit::RWMol> mol(
                RDKit::v1::MolBlockToMol(
                    molblock,
                    true,   // sanitize
                    false,  // removeHs
                    true    // strictParsing
                )
            );
            if (!mol) throw std::runtime_error("MolBlock parse failed");

            if (confId < 0 || confId >= mol->getNumConformers()) {
                throw std::runtime_error("confId out of range");
            }
            const RDKit::Conformer& conf = mol->getConformer(confId);

            ECHOScore scorer{pre, weights};
            scorer.interaction_cutoff = interaction_cutoff;
            scorer.electro_clamp      = electro_clamp;

            return scorer.score(conf, rep_max);
        },
        py::arg("py_precomputed"),
        py::arg("molblock"),
        py::arg("confId") = 0,
        py::arg("interaction_cutoff") = 6.0,
        py::arg("rep_max") = 5.0,
        py::arg("electro_clamp") = 2.0,
        py::arg("weights") = ECHOWeights::default_v1(),
        "Score one RDKit MolBlock conformer with ECHOScore."
    );

    // Fidelity check for the fast grid score: given the ligand topology (molblock) and a batch of
    // pose coordinates (N, n_atoms, 3), build the precompute ONCE (py_pc must have fast_sample=True so
    // the vdW grids exist) and return (full_scores, fast_scores) for every pose so Python can compute
    // their rank correlation. Diagnostic only — not on the docking path.
    m.def(
        "echo_score_fast_vs_full",
        [](py::object py_pc,
           const std::string& molblock,
           py::array_t<double, py::array::c_style | py::array::forcecast> coords,
           double interaction_cutoff,
           double rep_max,
           double electro_clamp,
           ECHOWeights weights) -> py::tuple {

            PreComputedData pre(py_pc);
            std::unique_ptr<RDKit::RWMol> mol(
                RDKit::v1::MolBlockToMol(molblock, true, false, true));
            if (!mol) throw std::runtime_error("MolBlock parse failed");
            if (mol->getNumConformers() == 0) throw std::runtime_error("molblock has no conformer");

            auto c = coords.unchecked<3>();
            const int N = static_cast<int>(c.shape(0));
            const int natoms = static_cast<int>(c.shape(1));

            ECHOScore scorer{pre, weights};
            scorer.interaction_cutoff = interaction_cutoff;
            scorer.electro_clamp      = electro_clamp;

            RDKit::Conformer& conf = mol->getConformer();
            std::vector<double> full(N), fast(N);
            for (int k = 0; k < N; ++k) {
                for (int a = 0; a < natoms; ++a)
                    conf.setAtomPos(a, RDGeom::Point3D(c(k, a, 0), c(k, a, 1), c(k, a, 2)));
                full[k] = scorer.score(conf, rep_max);
                fast[k] = scorer.score_fast(conf, rep_max);
            }
            return py::make_tuple(full, fast);
        },
        py::arg("py_precomputed"),
        py::arg("molblock"),
        py::arg("coords"),
        py::arg("interaction_cutoff") = 6.0,
        py::arg("rep_max") = 5.0,
        py::arg("electro_clamp") = 2.0,
        py::arg("weights") = ECHOWeights::default_v1(),
        "(full_scores, fast_scores) per pose — fidelity check for the fast grid score."
    );

    // Verify the analytic gradient of the grid score against finite differences (must be ~1e-5).
    // py_pc must have fast_sample=True so the vdW grids exist. Diagnostic only.
    m.def(
        "verify_grid_gradient",
        [](py::object py_pc, const std::string& molblock, ECHOWeights weights) -> double {
            PreComputedData pre(py_pc);
            if (pre.vdw_grids().empty())
                throw std::runtime_error("vdw grids not built — set py_pc.fast_sample=True");
            std::unique_ptr<RDKit::RWMol> mol(
                RDKit::v1::MolBlockToMol(molblock, true, false, true));
            if (!mol) throw std::runtime_error("MolBlock parse failed");
            AntColonyOptimizer opt(pre, *mol, weights);
            return opt.verify_grid_gradient();
        },
        py::arg("py_precomputed"), py::arg("molblock"),
        py::arg("weights") = ECHOWeights::default_v1(),
        "Max relative error of the analytic grid-score pose gradient vs finite differences."
    );

    // Run the analytic grid minimiser on a conformer; return (fast_before, fast_after, full_before,
    // full_after) so we can confirm it lowers the grid score (and see the effect on the full score).
    m.def(
        "grid_minimise_test",
        [](py::object py_pc, const std::string& molblock, ECHOWeights weights, int max_iters) -> py::tuple {
            PreComputedData pre(py_pc);
            if (pre.vdw_grids().empty())
                throw std::runtime_error("vdw grids not built — set py_pc.fast_sample=True");
            std::unique_ptr<RDKit::RWMol> mol(
                RDKit::v1::MolBlockToMol(molblock, true, false, true));
            if (!mol) throw std::runtime_error("MolBlock parse failed");
            AntColonyOptimizer opt(pre, *mol, weights);
            ECHOScore scorer{pre, weights};
            scorer.interaction_cutoff = pre.config().interaction_cutoff;
            scorer.electro_clamp      = pre.config().electro_clamp;
            RDKit::Conformer& conf = mol->getConformer();
            const double fast_before = scorer.score_fast(conf);
            const double full_before = scorer.score(conf, 5.0);
            opt.grid_minimise(conf, scorer, max_iters);
            const double fast_after = scorer.score_fast(conf);
            const double full_after = scorer.score(conf, 5.0);
            return py::make_tuple(fast_before, fast_after, full_before, full_after);
        },
        py::arg("py_precomputed"), py::arg("molblock"),
        py::arg("weights") = ECHOWeights::default_v1(), py::arg("max_iters") = 25,
        "Run the analytic grid minimiser; returns (fast_before, fast_after, full_before, full_after)."
    );

    // -----------------------------
    // Raw per-term breakdown (for offline weight-fitting). Unweighted term values,
    // including the split attractive/repulsive/clash channels.
    // -----------------------------
    m.def(
        "run_echo_terms",
        [](py::object py_pc,
           const std::string& molblock,
           int confId,
           double interaction_cutoff,
           double rep_max,
           double electro_clamp) -> py::dict {

            PreComputedData pre(py_pc);

            std::unique_ptr<RDKit::RWMol> mol(
                RDKit::v1::MolBlockToMol(
                    molblock,
                    true,   // sanitize
                    false,  // removeHs
                    true    // strictParsing
                )
            );
            if (!mol) throw std::runtime_error("MolBlock parse failed");

            if (confId < 0 || confId >= mol->getNumConformers()) {
                throw std::runtime_error("confId out of range");
            }
            const RDKit::Conformer& conf = mol->getConformer(confId);

            ECHOScore scorer{pre, ECHOWeights::default_v1()};
            scorer.interaction_cutoff = interaction_cutoff;
            scorer.electro_clamp      = electro_clamp;

            const std::map<std::string, double> terms = scorer.score_terms(conf, rep_max);
            py::dict out;
            for (const auto& kv : terms) {
                out[py::str(kv.first)] = kv.second;
            }
            return out;
        },
        py::arg("py_precomputed"),
        py::arg("molblock"),
        py::arg("confId") = 0,
        py::arg("interaction_cutoff") = 6.0,
        py::arg("rep_max") = 5.0,
        py::arg("electro_clamp") = 2.0,
        "Return the raw (unweighted) ECHO term channels as {name: value}, "
        "including split aromatic/nonbond attractive/repulsive/clash sub-terms."
    );

    m.def(
        "test_echo_score_speed",
        [](py::object py_pc,
           const std::string& molblock,
           int confId,
           double interaction_cutoff,
           double rep_max,
           double electro_clamp,
           int n_iters,
           ECHOWeights weights) -> std::pair<double, double> {

            PreComputedData pre(py_pc);

            std::unique_ptr<RDKit::RWMol> mol(
                RDKit::v1::MolBlockToMol(
                    molblock,
                    true,
                    false,
                    true
                )
            );
            if (!mol) throw std::runtime_error("MolBlock parse failed");

            if (confId < 0 || confId >= mol->getNumConformers()) {
                throw std::runtime_error("confId out of range");
            }
            const RDKit::Conformer& conf = mol->getConformer(confId);

            ECHOScore scorer{pre, weights};
            scorer.interaction_cutoff = interaction_cutoff;
            scorer.electro_clamp      = electro_clamp;

            // warmup
            volatile double warm = scorer.score(conf, rep_max);
            (void)warm;

            const int N = (n_iters > 0 ? n_iters : 1000);

            volatile double acc = 0.0;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < N; ++i) {
                acc += scorer.score(conf, rep_max);
            }
            const auto t1 = std::chrono::steady_clock::now();

            const std::chrono::duration<double> dt = t1 - t0;
            const double avg_ms = (dt.count() * 1000.0) / static_cast<double>(N);

            const double last_score = scorer.score(conf, rep_max);
            return {last_score, avg_ms};
        },
        py::arg("py_precomputed"),
        py::arg("molblock"),
        py::arg("confId") = 0,
        py::arg("interaction_cutoff") = 6.0,
        py::arg("rep_max") = 5.0,
        py::arg("electro_clamp") = 2.0,
        py::arg("n_iters") = 1000,
        py::arg("weights") = ECHOWeights::default_v1(),
        "Warm up once, then score N times and return (score, avg_ms_per_call)."
    );

    // -----------------------------
    // ACO docking entrypoint
    // -----------------------------
    m.def(
        "run_aco_docking",
        [](py::object py_pc,
           const std::string& molblock,
           int confId,
           ECHOWeights weights) -> py::list {

            PreComputedData pre(py_pc);

            std::unique_ptr<RDKit::RWMol> mol_ptr(
                RDKit::v1::MolBlockToMol(
                    molblock,
                    true,   // sanitize
                    false,  // removeHs
                    true    // strictParsing
                )
            );
            if (!mol_ptr) throw std::runtime_error("MolBlock parse failed");

            RDKit::RWMol mol = keep_only_conformer(*mol_ptr, confId);

            AntColonyOptimizer opt(pre, mol, weights);
            return opt.optimize(); // returns list of (score, coords)
        },
        py::arg("py_precomputed"),
        py::arg("molblock"),
        py::arg("confId") = 0,
        py::arg("weights") = ECHOWeights::default_v1(),
        "Run ACO docking (optionally with custom ECHO weights) and return "
        "[(score, coords_np), ...] from AntColonyOptimizer::optimize()."
    );
}
