#pragma once

#include <vector>
#include <deque>
#include <random>
#include <memory>
#include <Eigen/Dense>


#include <GraphMol/ROMol.h>
#include <GraphMol/Conformer.h>
#include <vector>
#include <deque>

#include "PreComputedData.h" 
#include "ScoringFunctions.h"

#include <vector>
#include <array>
#include <limits>
#include <cstddef>

#include <Eigen/Dense>

#include <GraphMol/ROMol.h>
#include <GraphMol/Conformer.h>
#include <Geometry/point.h>



struct AntData { double score; std::vector<double> sol; };
struct IterBest {
  double               score;
  std::vector<double>  discSol;
  RDKit::ROMol         mol;
};

class AntColonyOptimizer {
public:

    AntColonyOptimizer(const PreComputedData &precomputed_data,
                       const RDKit::ROMol &original_mol);
    
    const std::vector< std::array<unsigned int, 4> > dihedral_indices() const { return m_dihedral_indices ; }
    const uint64_t baseseed() const { return m_baseseed ; }
    
    
    
    pybind11::list optimize();
    
private:
    //data
    const PreComputedData &pre;
    //score
    ECHOScore m_scorer_base;
    RDKit::ROMol m_original_mol;
    
    uint64_t m_baseseed = 1234567ULL;
    std::vector< std::array<unsigned int, 4> > m_dihedral_indices;
    unsigned int m_max_iterations;
    unsigned int m_diversity_now;
    std::vector<std::vector<double>> m_pheromones, m_reset_array;
    std::vector<std::vector<unsigned int>> m_previous_deposits;

    double m_theta;
    double m_rho;
    double m_tmax; 
    double m_sigma;
    double m_evapRate;
    
    std::deque<double> m_last_scores;
    double m_best_since_smoothing; 
    std::vector<size_t> m_bj_indexes;
    std::vector<double> m_pbest_probs;
    std::vector<double> m_pbest; 
    std::vector<double> m_tmins;
    
    std::vector<AntData> m_ants_vec;
    
    const bool rankWeightedUpdate = true;
    void initialize_pheromones();
    
    std::vector<std::pair<double, std::size_t>> get_probability_array(const std::vector<double>& phe, double alpha) const;
    std::size_t roulette_wheel(const std::vector<std::pair<double, std::size_t>>& probArr, std::mt19937& rng) const;
    std::size_t make_choice(const std::vector<double>& phe, std::mt19937& rng, double alpha) const;
    void construct_solution(std::mt19937& rng, std::vector<double>& sol, double alpha) const;
    
    std::vector<IterBest> bestPerIter;
    double m_final_eps;
    double m_p_best;                
    int    m_flatWindowSize ;
    int    m_smoothingCount;
    void set_max_iterations();
    void updateBestSinceSmoothing(double score);
    void updateTauLimitsFromBest_PLANTS();   // sets m_tmax and m_tmins using p_best
    void updateBjIndexes(const std::vector<double> &sol);
    // ---- pheromone update bookkeeping ----
    void updatePreviousDeposits_lastOnly();  // stores only last deposit indices (no unbounded growth)
    void updatePheromones(double rawScoreWeighted);
    
    // ---- smoothing / stagnation ----
    bool isFlatWindow() const;
    void smoothPheromones();                // soft reset
    void updateSmoothing(double iterBestScore);
        
    struct SplitNmResult {
        double score = std::numeric_limits<double>::infinity();
        RDKit::ROMol mol;
        std::vector<double> discSol;
    };

    struct Refined {
        double realScore = std::numeric_limits<double>::infinity();
        std::vector<double> realNorm;
        std::vector<double> discSol;
        RDKit::ROMol realMol;
    };
    
    
    
    
    double get_alpha_now(unsigned int iter) const {
        double t = (m_max_iterations <= 1) ? 1.0 : static_cast<double>(iter) / (m_max_iterations - 1);
        if (t < 0.50)      return pre.config().a_lo;
        else if (t < 0.75) return pre.config().a_mid;
        else               return pre.config().a_hi;
    }
    // Apply a discrete (ACO) solution to a conformer (translation/rotation/torsions).
    void apply_ant_solution(RDKit::Conformer& conf,
                            const std::vector<double>& sol) const;
    
    // Convert a refined normalized vector back to a discrete solution (for pheromone update).
    std::vector<double> convertRealSpaceToDiscrete(const std::vector<double>& orig_disc,
                                                   const std::vector<double>& x_norm) const;
    
    // Refine a discrete solution in real space using NM, using an injected scorer + rep cap.
    SplitNmResult refinePoseSplitNmFromDiscrete(const std::vector<double>& discSol,
                                               const ECHOScore& scorer,
                                               double rep_max,
                                               double map_score_function
                                               ) const;
    
    std::pair<double, RDKit::ROMol> runLocalNelderMeadFromSeeds(
        const Eigen::RowVector3d &ini_trans_xyz,
        const Eigen::Vector3d    &ini_rot_deg,
        const std::vector<double> &ini_tors_deg,
        const ECHOScore &scorer,
        double rep_max,
        double map_score_function
    ) const;


};

