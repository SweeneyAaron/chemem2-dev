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
#include <sstream>
#include <GraphMol/FileParsers/MolSupplier.h>
namespace py = pybind11;

// Keep only one conformer so AntColonyOptimizer (which calls getConformer() with default)
// uses the conformer you chose.
static RDKit::RWMol keep_only_conformer(const RDKit::RWMol& in, int confId) {
    RDKit::RWMol m(in);

    const int n = static_cast<int>(m.getNumConformers());
    if (n == 0) {
        throw std::runtime_error("Mol has no conformers");
    }
    std::cout << "Mol has " << n << " conformers." <<std::endl;
    /*
    if (confId < 0 || confId >= n) {
        throw std::runtime_error("confId out of range");
    }

    for (int i = n - 1; i >= 0; --i) {
        if (i != confId) m.removeConformer(i);
    }

    // Optional: normalize remaining conformer id to 0 (not strictly necessary)
    m.getConformer(0).setId(0);
    */
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
    // Score helpers 
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

            return scorer.score(conf, rep_max, 1);
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
            volatile double warm = scorer.score(conf, rep_max, 1);
            (void)warm;

            const int N = (n_iters > 0 ? n_iters : 1000);

            volatile double acc = 0.0;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < N; ++i) {
                acc += scorer.score(conf, rep_max, 1);
            }
            const auto t1 = std::chrono::steady_clock::now();

            const std::chrono::duration<double> dt = t1 - t0;
            const double avg_ms = (dt.count() * 1000.0) / static_cast<double>(N);

            const double last_score = scorer.score(conf, rep_max, 1);
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
    // Docking entrypoint (UPDATED)
    // -----------------------------
    /*
    m.def(
        "run_aco_docking",
        [](py::object py_pc,
           const std::string& molblock,
           int confId,
           double interaction_cutoff,
           double electro_clamp,
           uint64_t baseseed,
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

            // 1. Initialize the Base Scorer
            ECHOScore scorer{pre, weights};
            scorer.interaction_cutoff = interaction_cutoff;
            scorer.electro_clamp      = electro_clamp;

            // 2. Pass scorer and seed into the updated Optimizer
            AntColonyOptimizer opt(pre, scorer, mol, baseseed);
            
            return opt.optimize(); // returns list of (score, coords)
        },
        py::arg("py_precomputed"),
        py::arg("molblock"),
        py::arg("confId") = 0,
        py::arg("interaction_cutoff") = 6.0,
        py::arg("electro_clamp") = 2.0,
        py::arg("baseseed") = 42,  // Default seed for reproducibility
        py::arg("weights") = ECHOWeights::default_v1(),
        "Run Basin-Hopping/L-BFGS docking and return [(score, coords_np), ...] from optimize()."
    );*/
    // -----------------------------
    // Docking entrypoint (UPDATED FOR MULTI-CONFORMER SDF)
    // -----------------------------
    m.def(
        "run_aco_docking",
        [](py::object py_pc,
           const std::string& sdf_string, // Now expects the SDF string
           double interaction_cutoff,
           double electro_clamp,
           uint64_t baseseed,
           ECHOWeights weights) -> py::list {

            PreComputedData pre(py_pc);

            // 1. Parse the multi-molecule SDF string
            std::stringstream ss(sdf_string);
            RDKit::ForwardSDMolSupplier suppl(&ss, false, true, false); // sanitize=true, removeHs=false

            std::unique_ptr<RDKit::RWMol> mol_ptr;

            while (!suppl.atEnd()) {
                std::unique_ptr<RDKit::ROMol> m(suppl.next());
                if (!m) continue;

                if (!mol_ptr) {
                    // First molecule becomes our base template
                    mol_ptr.reset(new RDKit::RWMol(*m));
                } else {
                    // For subsequent molecules, steal their conformer and add it to our base template
                    RDKit::Conformer* new_conf = new RDKit::Conformer(m->getConformer());
                    new_conf->setId(mol_ptr->getNumConformers()); // Assign sequential ID (1, 2, 3...)
                    
                    // addConformer takes ownership of the pointer
                    mol_ptr->addConformer(new_conf, true); 
                }
            }

            if (!mol_ptr || mol_ptr->getNumConformers() == 0) {
                throw std::runtime_error("Failed to parse SDF string or no conformers found.");
            }

            // Notice we completely removed keep_only_conformer! 
            // We want the optimizer to have access to ALL of them.
            RDKit::RWMol mol = *mol_ptr;

            // 2. Initialize the Base Scorer
            ECHOScore scorer{pre, weights};
            scorer.interaction_cutoff = interaction_cutoff;
            scorer.electro_clamp      = electro_clamp;

            // 3. Pass scorer, seed, and the MULTI-CONFORMER mol into the Optimizer
            AntColonyOptimizer opt(pre, scorer, mol, baseseed);
            
            return opt.optimize(); // returns list of (score, coords)
        },
        py::arg("py_precomputed"),
        py::arg("sdf_string"),
        py::arg("interaction_cutoff") = 6.0,
        py::arg("electro_clamp") = 2.0,
        py::arg("baseseed") = 42,  
        py::arg("weights") = ECHOWeights::default_v1(),
        "Run Rigid-Body/L-BFGS docking on a multi-conformer SDF string."
    );
}