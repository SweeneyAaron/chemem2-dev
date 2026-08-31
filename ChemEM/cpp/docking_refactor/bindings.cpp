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

// (N,3) float64 view of a conformer. SearchFunctions.cpp has its own copy of this for
// optimize()'s return value, but that one is file-static, so run_local_refine needs its
// own. Both produce the same layout, so poses from either entry point are interchangeable.
static py::array_t<double> conformer_to_coords(const RDKit::Conformer& conf) {
    const py::ssize_t N = static_cast<py::ssize_t>(conf.getNumAtoms());
    py::array_t<double> arr({N, static_cast<py::ssize_t>(3)});
    auto buf = arr.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < N; ++i) {
        const auto p = conf.getAtomPos(static_cast<unsigned int>(i));
        buf(i, 0) = p.x;
        buf(i, 1) = p.y;
        buf(i, 2) = p.z;
    }
    return arr;
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
        .def_readwrite("unsat_polar", &ECHOWeights::unsat_polar)
        .def_readwrite("aromatic_attr", &ECHOWeights::aromatic_attr)
        .def_readwrite("aromatic_clash", &ECHOWeights::aromatic_clash)
        .def_readwrite("nonbond_attr", &ECHOWeights::nonbond_attr)
        .def_readwrite("nonbond_rep", &ECHOWeights::nonbond_rep)
        .def_readwrite("clash", &ECHOWeights::clash);

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

    // -----------------------------
    // Local refinement of ONE given pose (no ACO search in front of it)
    // -----------------------------
    m.def(
        "run_local_refine",
        [](py::object py_pc,
           const std::string& molblock,
           int confId,
           ECHOWeights weights,
           double rep_max,
           double map_score_function) -> py::dict {

            PreComputedData pre(py_pc);

            // rep_max < 0 means "whatever the polish uses", i.e. --repulsion-cap-polish.
            // Defaulting to the config rather than to a literal keeps this entry point
            // from drifting away from the cap --dock actually ranks its poses at.
            const double cap = (rep_max < 0.0) ? pre.config().repCap_final_nm : rep_max;

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

            // One scorer, built exactly as the refiner's own is: interaction_cutoff and
            // electro_clamp come from the config, NOT from pybind defaults. Those two are
            // never plumbed through py_pc, so run_echo_score's 6.0/2.0 signature defaults
            // do not match what the engine scores with -- reading the config is what keeps
            // the terms describing the surface the pose was actually minimised on.
            ECHOScore scorer{pre, weights};
            scorer.interaction_cutoff = pre.config().interaction_cutoff;
            scorer.electro_clamp      = pre.config().electro_clamp;

            auto dump = [&](const RDKit::Conformer& c) {
                py::dict d;
                for (const auto& kv : scorer.score_terms(c, cap)) {
                    d[py::str(kv.first)] = kv.second;
                }
                return d;
            };

            // Before: the pose as handed in. For the deposited pose this IS the native
            // reference the offline fit scores against, and taking it here rather than
            // from a separate run_echo_terms call means the two can never be dumped under
            // different constants.
            const double score_start = scorer.score(mol.getConformer(), cap,
                                                    map_score_function);
            py::dict terms_start = dump(mol.getConformer());

            AntColonyOptimizer opt(pre, mol, weights);
            auto [score, refined] = opt.refineCurrentPose(cap, map_score_function);
            const RDKit::Conformer& conf = refined.getConformer();

            py::dict out;
            out["score"]        = score;
            out["coords"]       = conformer_to_coords(conf);
            out["terms"]        = dump(conf);
            out["score_start"]  = score_start;
            out["terms_start"]  = terms_start;
            // The constants this call actually used, so a caller never has to guess them.
            out["rep_max"]            = cap;
            out["interaction_cutoff"] = scorer.interaction_cutoff;
            out["electro_clamp"]      = scorer.electro_clamp;
            out["local_minimiser"]    = pre.config().local_minimiser;
            return out;
        },
        py::arg("py_precomputed"),
        py::arg("molblock"),
        py::arg("confId") = 0,
        py::arg("weights") = ECHOWeights::default_v1(),
        py::arg("rep_max") = -1.0,
        py::arg("map_score_function") = 0.0,
        "Locally refine ONE pose under the given ECHO weights.\n\n"
        "This is the final Nelder-Mead/L-BFGS polish --dock ranks its poses by, seeded\n"
        "at the pose you pass in instead of at an ant's solution, so it answers 'where\n"
        "does this pose settle under these weights'.\n\n"
        "Returns a dict: score / coords / terms (after refinement), score_start /\n"
        "terms_start (the pose as handed in), and the rep_max, interaction_cutoff,\n"
        "electro_clamp and local_minimiser it used. Before and after are scored with ONE\n"
        "scorer, so they are directly comparable and an offline fit cannot mix constants.\n\n"
        "rep_max < 0 (the default) uses --repulsion-cap-polish from the precompute.\n"
        "--local-minimiser selects Nelder-Mead vs L-BFGS, as in a dock run.\n"
        "Note the refiner's translation box is +/-2 A around the input centroid, so a\n"
        "pose displaced further than that cannot be fully recovered whatever the weights."
    );
}
