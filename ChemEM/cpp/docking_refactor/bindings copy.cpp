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

PYBIND11_MODULE(docking, m) {
    m.doc() = "ChemEM core scoring + docking bindings";

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
        .def_readwrite("unsat_polar", &ECHOWeights::unsat_polar);

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
           int confId) -> py::list {

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

            AntColonyOptimizer opt(pre, mol);
            return opt.optimize(); // returns list of (score, coords)
        },
        py::arg("py_precomputed"),
        py::arg("molblock"),
        py::arg("confId") = 0,
        "Run ACO docking and return [(score, coords_np), ...] from AntColonyOptimizer::optimize()."
    );
}
