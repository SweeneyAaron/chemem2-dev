#pragma once

#include <vector>
#include <random>
#include <mutex>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <GraphMol/ROMol.h>
#include <GraphMol/Conformer.h>

// Assuming these are the names of your other headers based on your includes
#include "ScoringFunctions.h"
#include "PreComputedData.h" 

namespace py = pybind11;

// =========================================================================
// Struct to hold the best poses found during the search
// =========================================================================
struct IterBest {
    double score;
    std::vector<double> discSol; // Kept for compatibility if needed elsewhere
    RDKit::ROMol mol;
};

// =========================================================================
// Main Optimizer Class
// (Retains the name 'AntColonyOptimizer' for pybind11 compatibility, 
//  but now implements Basin Hopping + L-BFGS under the hood).
// =========================================================================
class AntColonyOptimizer {
public:
    // Constructor: Takes the precomputed data, baseline scorer, and the original molecule
    AntColonyOptimizer(
        const PreComputedData &pre_,
        const ECHOScore &scorer_base_,
        const RDKit::ROMol &original_mol_,
        uint64_t baseseed_
    ) : pre(pre_), 
        m_scorer_base(scorer_base_), 
        m_original_mol(original_mol_), 
        m_baseseed(baseseed_) {}

    // Dummy implementation to satisfy legacy Python calls
    void set_max_iterations();

    // The main execution function called from Python
    py::list optimize();

private:
    // ---------------------------------------------------------------------
    // Core Data Members
    // ---------------------------------------------------------------------
    const PreComputedData &pre;
    const ECHOScore m_scorer_base;
    const RDKit::ROMol m_original_mol;
    uint64_t m_baseseed;

    // ---------------------------------------------------------------------
    // Note: All of the old ACO variables (pheromones, discrete grids) 
    // and Nelder-Mead functions have been entirely removed. The state 
    // is now handled dynamically inside the optimize() function using 
    // the BasinHopperState struct in the .cpp file.
    // ---------------------------------------------------------------------
};