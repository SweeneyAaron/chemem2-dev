#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>
#include <stdexcept>
#include <array>
#include <limits>
#include <numeric>
#include <deque>
#include <mutex>
#include <omp.h>

#include <GraphMol/ROMol.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/MolTransforms/MolTransforms.h>
#include <Geometry/point.h>

#include <Eigen/Dense>
#include <Eigen/Geometry> 

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "SearchFunctions.h"
#include "GeometryUtils.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <LBFGSB.h> 

namespace py = pybind11;

// -------------------------------------------------------------------------
// Fast, thread-safe RNG Seeder
// -------------------------------------------------------------------------
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// -------------------------------------------------------------------------
// Missing Helper: Convert RDKit conformer to numpy array for Python
// -------------------------------------------------------------------------
static py::array_t<double> conformerToCoords(const RDKit::Conformer& conf) {
    auto n_atoms = conf.getNumAtoms();
    py::array_t<double> coords({n_atoms, 3u});
    auto ptr = coords.mutable_unchecked<2>();
    for (unsigned int i = 0; i < n_atoms; ++i) {
        const auto& pt = conf.getAtomPos(i);
        ptr(i, 0) = pt.x;
        ptr(i, 1) = pt.y;
        ptr(i, 2) = pt.z;
    }
    return coords;
}

// =========================================================================
// 1. Thread-Local State Manager 
// =========================================================================
struct BasinHopperState {
    std::vector<double> current_x;
    double current_score;
    
    std::vector<double> best_x;
    double best_score;
    
    size_t nTors;
    Eigen::RowVector3d base_trans_xyz; 

    BasinHopperState(size_t nTors, const Eigen::RowVector3d& grid_pt, const std::vector<double>& start_x, double start_score)
        : current_x(start_x), current_score(start_score), 
          best_x(start_x), best_score(start_score), 
          nTors(nTors), base_trans_xyz(grid_pt) {}

    void apply_to_conformer(
        const std::vector<double>& x_norm,
        RDKit::Conformer& conf,
        const std::vector<RDGeom::Point3D>& baseCoords,
        const PreComputedData& pre) const 
    {
        for (size_t i = 0; i < baseCoords.size(); ++i) {
            conf.setAtomPos(i, baseCoords[i]);
        }

        if (nTors > 0) {
            const auto& torsion_idxs = pre.ligand_score().ligand_torsion_idxs; 
            for (size_t t = 0; t < nTors; ++t) {
                double angle_deg = x_norm[6 + t] * 360.0;
                const auto& t_atoms = torsion_idxs[t]; 
                
                // FIXED: Removed RDGeom:: namespace here
                MolTransforms::setDihedralDeg(
                    conf, t_atoms[0], t_atoms[1], t_atoms[2], t_atoms[3], angle_deg
                );
            }
        }

        Eigen::Vector3d pivot(0.0, 0.0, 0.0);
        for (size_t i = 0; i < conf.getNumAtoms(); ++i) {
            RDGeom::Point3D pt = conf.getAtomPos(i);
            pivot.x() += pt.x;
            pivot.y() += pt.y;
            pivot.z() += pt.z;
        }
        pivot /= static_cast<double>(conf.getNumAtoms());

        double rx = x_norm[3] * 2.0 * M_PI; 
        double ry = x_norm[4] * 2.0 * M_PI; 
        double rz = x_norm[5] * 2.0 * M_PI; 
        
        Eigen::AngleAxisd rollAngle(rx, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(ry, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(rz, Eigen::Vector3d::UnitZ());
        Eigen::Matrix3d R = (yawAngle * pitchAngle * rollAngle).matrix();

        const double TRANS_WINDOW = 4.0; 
        double tx = base_trans_xyz.x() + (x_norm[0] - 0.5) * TRANS_WINDOW;
        double ty = base_trans_xyz.y() + (x_norm[1] - 0.5) * TRANS_WINDOW;
        double tz = base_trans_xyz.z() + (x_norm[2] - 0.5) * TRANS_WINDOW;
        Eigen::Vector3d T(tx, ty, tz);

        for (size_t i = 0; i < conf.getNumAtoms(); ++i) {
            RDGeom::Point3D pt = conf.getAtomPos(i);
            Eigen::Vector3d vec(pt.x, pt.y, pt.z);
            
            vec = R * (vec - pivot) + T; 
            
            conf.setAtomPos(i, RDGeom::Point3D(vec.x(), vec.y(), vec.z()));
        }
    }

    std::vector<double> generate_kick(std::mt19937& rng) const {
        std::vector<double> mut_x = current_x;
        std::uniform_int_distribution<int> type_dist(0, 2);
        int mut_type = type_dist(rng);

        if (mut_type == 0) { 
            std::uniform_real_distribution<double> t_dist(-0.15, 0.15);
            for(int i = 0; i < 3; ++i) {
                mut_x[i] = std::clamp(mut_x[i] + t_dist(rng), 0.0, 1.0);
            }
        } 
        else if (mut_type == 1) { 
            std::uniform_real_distribution<double> r_dist(-0.15, 0.15);
            for(int i = 3; i < 6; ++i) {
                mut_x[i] += r_dist(rng);
                mut_x[i] -= std::floor(mut_x[i]); 
            }
        } 
        else if (nTors > 0) { 
            std::uniform_int_distribution<int> tors_dist(6, 5 + nTors);
            std::uniform_real_distribution<double> angle_dist(-0.5, 0.5); 
            
            int t_idx = tors_dist(rng);
            mut_x[t_idx] += angle_dist(rng);
            mut_x[t_idx] -= std::floor(mut_x[t_idx]); 
        }
        return mut_x;
    }

    void attempt_step(const std::vector<double>& new_x, double new_score, double temperature, std::mt19937& rng) {
        bool accept = false;
        if (new_score < current_score) {
            accept = true; 
        } else {
            double p = std::exp(-(new_score - current_score) / temperature);
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            if (u01(rng) < p) accept = true;
        }

        if (accept) {
            current_x = new_x;
            current_score = new_score;
            if (current_score < best_score) {
                best_score = current_score;
                best_x = current_x;
            }
        }
    }
};

// =========================================================================
// 2. L-BFGS Minimizer Wrapper
// =========================================================================
static std::pair<double, std::vector<double>> runLocalLBFGS_Helper(
    const std::vector<double>& x_start,
    const BasinHopperState& state,
    const ECHOScore& scorer,
    const PreComputedData& pre,
    RDKit::Conformer& conf,
    const std::vector<RDGeom::Point3D>& baseCoords,
    double rep_max,
    int map_score_function) 
{
    const size_t D = x_start.size();

    auto eval_full = [&](const std::vector<double>& x_norm) -> double {
        state.apply_to_conformer(x_norm, conf, baseCoords, pre);
        return scorer.score(conf, rep_max, map_score_function);
    };

    auto lbfgs_objective = [&](const Eigen::VectorXd& x_eigen, Eigen::VectorXd& grad) -> double {
        std::vector<double> x(x_eigen.data(), x_eigen.data() + D);
        double f0 = eval_full(x);
        
        const double h = 5e-4; 

        for (size_t i = 0; i < D; ++i) {
            double orig = x[i];
            
            double x_fwd = std::min(1.0, orig + h);
            double x_bwd = std::max(0.0, orig - h);
            
            x[i] = x_fwd; double f_fwd = eval_full(x);
            x[i] = x_bwd; double f_bwd = eval_full(x);
            x[i] = orig; 
            
            double actual_h = x_fwd - x_bwd;
            grad[i] = (actual_h > 1e-8) ? (f_fwd - f_bwd) / actual_h : 0.0;
        }
        return f0;
    };

    LBFGSpp::LBFGSBParam<double> param;
    param.m = 24;  //12             
    param.epsilon = 1e-3;       
    param.max_iterations = 120;//60  
    param.ftol = 1e-4;          
    param.wolfe = 0.9;          

    LBFGSpp::LBFGSBSolver<double> solver(param);

    const double TRANS_NORM_WINDOW = 2.5 / 4.0;  //changed from 2.5
    const double ROT_NORM_WINDOW   = 20.0 / 360.0; 
    const double TORS_NORM_WINDOW  = 30.0 / 360.0; 

    Eigen::VectorXd lb(D);
    Eigen::VectorXd ub(D);
    Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(x_start.data(), D);

    for (size_t i = 0; i < D; ++i) {
        double current_val = x[i];
        double window = (i < 3) ? TRANS_NORM_WINDOW : ((i < 6) ? ROT_NORM_WINDOW : TORS_NORM_WINDOW);

        lb[i] = std::max(0.0, current_val - window);
        ub[i] = std::min(1.0, current_val + window);
    }
    
    double best_score = eval_full(x_start);
    try {
        solver.minimize(lbfgs_objective, x, best_score, lb, ub);
    } catch (...) {}

    std::vector<double> x_opt(x.data(), x.data() + D);
    return {best_score, x_opt};
}

// =========================================================================
// 3. Main Optimization Loop (Basin Hopping)
// =========================================================================


/*
void AntColonyOptimizer::set_max_iterations() {}

py::list AntColonyOptimizer::optimize() {
    // 1. Start the timer
    auto start_time = std::chrono::steady_clock::now();

    const auto &config = pre.config();
    const auto &site_data = pre.binding_site_grid();
    const int inner_map_score = config.inner_map_score;
    
    // --- PARAMETER MAPPING ---
    const int N_RANDOM_DROPS = static_cast<int>(config.n_global_search); // e.g., 2000 (Scattergun)
    const int N_RESTARTS     = static_cast<int>(config.n_local_search);  // e.g., 32 or 64 (Sniper)
    const int JUMPS_PER_RESTART = 15;  // Shallow kicks since we are pre-filtering
    const double TEMPERATURE = 1.2;    
    
    const size_t nTors = pre.ligand_score().ligand_torsion_idxs.size();
    const size_t D = 6 + nTors;

    omp_set_num_threads(config.n_cpu);
    const double rmsd_pock = 2.0;
    const double SEARCH_REPCAP = 1.0;
    // =========================================================================
    // PHASE 1: THE FAST PRE-FILTER (SCATTERGUN)
    // =========================================================================
    struct RandomGuess {
        double score;
        Eigen::RowVector3d trans;
        std::vector<double> x;
    };
    std::vector<RandomGuess> initial_guesses(N_RANDOM_DROPS);

    #pragma omp parallel 
    {
        std::mt19937 rng(splitmix64(m_baseseed ^ omp_get_thread_num()));
        ECHOScore scorer = m_scorer_base;
        RDKit::ROMol localMol(m_original_mol);
        RDKit::Conformer &conf = localMol.getConformer();
        
        std::vector<RDGeom::Point3D> baseCoords;
        baseCoords.reserve(conf.getNumAtoms());
        for (unsigned int i = 0; i < conf.getNumAtoms(); ++i) {
            baseCoords.push_back(conf.getAtomPos(i));
        }

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < N_RANDOM_DROPS; ++i) {
            
            const auto& tpts = config.translation_points;
            size_t tIdx = std::uniform_int_distribution<size_t>(0, tpts.size() - 1)(rng);
            const auto tp_zyx = tpts[tIdx];
            const auto origin_xyz = site_data.origin;  
            const auto apix_xyz   = site_data.apix;          
            
            Eigen::RowVector3d ini_trans_xyz{
                origin_xyz[0] + static_cast<double>(tp_zyx[2]) * apix_xyz[0],  
                origin_xyz[1] + static_cast<double>(tp_zyx[1]) * apix_xyz[1],  
                origin_xyz[2] + static_cast<double>(tp_zyx[0]) * apix_xyz[2]   
            };
            
            std::vector<double> x_start(D);
            std::uniform_real_distribution<double> U01(0.0, 1.0);
            for(size_t d=0; d<D; ++d) x_start[d] = U01(rng); 
            
            // Instantly score without L-BFGS
            BasinHopperState state(nTors, ini_trans_xyz, x_start, 0.0);
            state.apply_to_conformer(x_start, conf, baseCoords, pre);
            double raw_score = scorer.score(conf, SEARCH_REPCAP, inner_map_score);
            
            initial_guesses[i] = {raw_score, ini_trans_xyz, x_start};
        }
    }

    // Sort to find the best funnels
    std::sort(initial_guesses.begin(), initial_guesses.end(), 
              [](const RandomGuess& a, const RandomGuess& b) { return a.score < b.score; });


    // =========================================================================
    // PHASE 1.5: GREEDY DIVERSITY FILTERING
    // =========================================================================
    std::vector<RandomGuess> diverse_guesses;
    diverse_guesses.reserve(N_RESTARTS);
    
    // Create a temporary molecule just to calculate heavy-atom RMSD
    RDKit::ROMol tempMol(m_original_mol);
    RDKit::Conformer &tempConf = tempMol.getConformer();
    std::vector<RDGeom::Point3D> baseCoordsForFilter;
    for (unsigned int i = 0; i < tempConf.getNumAtoms(); ++i) {
        baseCoordsForFilter.push_back(tempConf.getAtomPos(i));
    }
    
    // We store the 3D geometry of accepted seeds to compare against
    std::vector<RDKit::ROMol> selected_seed_mols;
    BasinHopperState tempState(nTors, Eigen::RowVector3d(0,0,0), std::vector<double>(D, 0.0), 0.0);

    for (const auto& guess : initial_guesses) {
        if (diverse_guesses.size() >= static_cast<size_t>(N_RESTARTS)) break;
        
        // 1. Unpack the guess into 3D Cartesian coordinates
        tempState.base_trans_xyz = guess.trans;
        tempState.apply_to_conformer(guess.x, tempConf, baseCoordsForFilter, pre);
        
        // 2. Check RMSD against all previously selected diverse seeds
        bool too_close = false;
        for (const auto& sel_mol : selected_seed_mols) {
            if (GeometryUtils::heavy_atom_rmsd(tempMol, sel_mol) < rmsd_pock) { //changed from config.rms_cutoff
                too_close = true;
                break;
            }
        }
        
        // 3. If it's a unique part of the pocket, keep it!
        if (!too_close) {
            diverse_guesses.push_back(guess);
            selected_seed_mols.push_back(RDKit::ROMol(tempMol)); 
        }
    }
    
    int actual_restarts = diverse_guesses.size();


    // =========================================================================
    // PHASE 2: BASIN HOPPING ON THE TOP POSES (SNIPER)
    // =========================================================================
    std::vector<IterBest> global_candidates;
    std::mutex cand_mutex;
    
    std::atomic<int> completed_tasks{0};
    std::mutex print_mutex;
    const int bar_width = 50;

    #pragma omp parallel 
    {
        std::mt19937 rng(splitmix64(m_baseseed ^ omp_get_thread_num()));
        
        ECHOScore scorer = m_scorer_base;
        RDKit::ROMol localMol(m_original_mol);
        RDKit::Conformer &conf = localMol.getConformer();
        
        std::vector<RDGeom::Point3D> baseCoords;
        baseCoords.reserve(conf.getNumAtoms());
        for (unsigned int i = 0; i < conf.getNumAtoms(); ++i) {
            baseCoords.push_back(conf.getAtomPos(i));
        }

        #pragma omp for schedule(dynamic, 1)
        for (int r = 0; r < actual_restarts; ++r) {
            
            // Start from the pre-filtered DIVERSE guess!
            Eigen::RowVector3d ini_trans_xyz = diverse_guesses[r].trans;
            std::vector<double> x_start = diverse_guesses[r].x;
            
            BasinHopperState state(nTors, ini_trans_xyz, x_start, 0.0);

            // 1. Initial Funnel Slide
            auto [init_score, init_x] = runLocalLBFGS_Helper(
                x_start, state, scorer, pre, conf, baseCoords, config.repCap_inner_nm, inner_map_score);
            
            state.current_score = init_score;
            state.current_x = init_x;
            state.best_score = init_score;
            state.best_x = init_x;

            // 2. Basin Hopping Jumps
            for (int jump = 0; jump < JUMPS_PER_RESTART; ++jump) {
                std::vector<double> x_mut = state.generate_kick(rng);
                auto [new_score, x_new_opt] = runLocalLBFGS_Helper(
                    x_mut, state, scorer, pre, conf, baseCoords, config.repCap_inner_nm, inner_map_score);
                state.attempt_step(x_new_opt, new_score, TEMPERATURE, rng);
            }

            state.apply_to_conformer(state.best_x, conf, baseCoords, pre);
            
            IterBest result;
            result.score = state.best_score;
            result.mol = RDKit::ROMol(localMol);
            
            {
                std::lock_guard<std::mutex> lock(cand_mutex);
                global_candidates.push_back(std::move(result));
            }

            // 3. Progress Bar Update
            int completed = ++completed_tasks;
            {
                std::lock_guard<std::mutex> print_lock(print_mutex);
                float progress = static_cast<float>(completed) / actual_restarts;
                int pos = static_cast<int>(bar_width * progress);

                std::cout << "\rDocking |";
                for (int i = 0; i < bar_width; ++i) {
                    if (i < pos) std::cout << "█";
                    else if (i == pos) std::cout << "▌";
                    else std::cout << " ";
                }
                std::cout << "| " << std::setw(3) << static_cast<int>(progress * 100.0) << "% "
                          << "[" << completed << "/" << actual_restarts << " threads]" << std::flush;
            }
        }
    }
    
    std::cout << "\n";

    // =========================================================================
    // PHASE 3: REPORTING & CLEANUP
    // =========================================================================
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    
    // Calculate actual number of minimizations performed
    int total_minimizations = actual_restarts * (1 + JUMPS_PER_RESTART);

    std::cout << "========================================\n"
              << "Docking Complete!\n"
              << "Total time taken: " << std::fixed << std::setprecision(2) << elapsed_seconds.count() << " seconds\n"
              << "Throughput: " << total_minimizations / elapsed_seconds.count() << " L-BFGS minimizations/sec\n"
              << "========================================\n" << std::flush;

    std::sort(global_candidates.begin(), global_candidates.end(),
              [](const IterBest &A, const IterBest &B){ return A.score < B.score; });

    std::vector<IterBest> selected;
    for (const auto &cand : global_candidates) {
        if (selected.size() >= static_cast<std::size_t>(config.topN)) break;
        bool tooClose = false;
        for (const auto &sel : selected) {
            if (GeometryUtils::heavy_atom_rmsd(cand.mol, sel.mol) < config.rms_cutoff) { 
                tooClose = true; break; 
            }
        }
        if (!tooClose) selected.push_back(cand);
    }

    py::list out;
    for (const auto &cand : selected) {
      out.append(py::make_tuple(cand.score, conformerToCoords(cand.mol.getConformer())));
    }
    return out;
}
*/
void AntColonyOptimizer::set_max_iterations() {}

py::list AntColonyOptimizer::optimize() {
    auto start_time = std::chrono::steady_clock::now();

    const auto &config = pre.config();
    const auto &site_data = pre.binding_site_grid();
    const int inner_map_score = config.inner_map_score;
    
    // We want the final polished poses to be equal to n_local_search
    const int N_RESTARTS = static_cast<int>(config.n_local_search); 
    const int JUMPS_PER_RESTART = 10; // Shallow Basin Hopping polish
    const double TEMPERATURE = 1.2;    
    
    const size_t nTors = pre.ligand_score().ligand_torsion_idxs.size();
    const size_t D = 6 + nTors;

    omp_set_num_threads(config.n_cpu);

    struct VoxelGuess {
        double score;
        Eigen::RowVector3d trans;
        std::vector<double> x; // The full [0,1] state array
        int conf_id;
    };

    // We use a map to store exactly ONE best guess per 3.0A Voxel
    std::mutex map_mutex;
    std::map<std::tuple<int, int, int>, VoxelGuess> voxel_best_poses;

    // =========================================================================
    // PHASE 1: SYSTEMATIC RIGID-BODY EXHAUSTIVE SEARCH
    // =========================================================================
    // 1. Generate Systematic Rotations (e.g., 4 steps per axis = 64 rotations)
    std::vector<Eigen::Vector3d> systematic_rotations;
    for (double rx = 0; rx < 1.0; rx += 0.083) { //0.25
        for (double ry = 0; ry < 1.0; ry += 0.083) {//from 0.25
            for (double rz = 0; rz < 1.0; rz += 0.083) {//from 0.25
                systematic_rotations.push_back(Eigen::Vector3d(rx, ry, rz));
            }
        }
    }

    const double VOXEL_SIZE = 2.0; // 3.0 Angstrom bins

    // Loop over EVERY conformer passed from Python
    int num_conformers = m_original_mol.getNumConformers();
    
    #pragma omp parallel 
    {
        ECHOScore scorer = m_scorer_base;
        RDKit::ROMol localMol(m_original_mol);
        RDKit::Conformer &conf = localMol.getConformer();
        
        // Loop through all points, but we will group them by Voxel
        #pragma omp for schedule(dynamic, 1) collapse(2)
        for (size_t tIdx = 0; tIdx < config.translation_points.size(); ++tIdx) {
            for (int cid = 0; cid < num_conformers; ++cid) {
                
                // Get the base coordinates for THIS specific conformer
                const RDKit::Conformer& src_conf = m_original_mol.getConformer(cid);
                std::vector<RDGeom::Point3D> baseCoords;
                for (unsigned int i = 0; i < src_conf.getNumAtoms(); ++i) {
                    baseCoords.push_back(src_conf.getAtomPos(i));
                }

                // Calculate exact Cartesian translation
                const auto tp_zyx = config.translation_points[tIdx];
                Eigen::RowVector3d trans{
                    site_data.origin[0] + static_cast<double>(tp_zyx[2]) * site_data.apix[0],  
                    site_data.origin[1] + static_cast<double>(tp_zyx[1]) * site_data.apix[1],  
                    site_data.origin[2] + static_cast<double>(tp_zyx[0]) * site_data.apix[2]   
                };

                // Calculate Voxel ID
                std::tuple<int, int, int> voxel_id = {
                    static_cast<int>(std::floor(trans.x() / VOXEL_SIZE)),
                    static_cast<int>(std::floor(trans.y() / VOXEL_SIZE)),
                    static_cast<int>(std::floor(trans.z() / VOXEL_SIZE))
                };

                // Extract Dihedrals from this conformer to populate x[6...N]
                std::vector<double> base_x(D, 0.5); // Default center
                if (nTors > 0) {
                    const auto& t_idxs = pre.ligand_score().ligand_torsion_idxs;
                    for (size_t t = 0; t < nTors; ++t) {
                        double deg = MolTransforms::getDihedralDeg(
                            src_conf, t_idxs[t][0], t_idxs[t][1], t_idxs[t][2], t_idxs[t][3]);
                        if (deg < 0) deg += 360.0;
                        base_x[6 + t] = deg / 360.0;
                    }
                }

                // Test all systematic rotations
                BasinHopperState state(nTors, trans, base_x, 0.0);
                for (const auto& rot : systematic_rotations) {
                    std::vector<double> test_x = base_x;
                    test_x[3] = rot.x(); test_x[4] = rot.y(); test_x[5] = rot.z();
                    
                    state.apply_to_conformer(test_x, conf, baseCoords, pre);
                    
                    // Softcore score to ignore minor clashes!
                    double score = scorer.score(conf, 0.1, inner_map_score);

                    // Update the Voxel Map safely
                    std::lock_guard<std::mutex> lock(map_mutex);
                    if (voxel_best_poses.find(voxel_id) == voxel_best_poses.end() || 
                        score < voxel_best_poses[voxel_id].score) {
                        voxel_best_poses[voxel_id] = {score, trans, test_x, cid};
                    }
                }
            }
        }
    }

    // =========================================================================
    // PHASE 1.5: EXTRACT AND SORT THE VOXEL WINNERS
    // =========================================================================
    std::vector<VoxelGuess> diverse_guesses;
    for (auto const& [key, val] : voxel_best_poses) {
        diverse_guesses.push_back(val);
    }
    std::sort(diverse_guesses.begin(), diverse_guesses.end(), 
              [](const VoxelGuess& a, const VoxelGuess& b) { return a.score < b.score; });
              
    int actual_restarts = std::min(N_RESTARTS, static_cast<int>(diverse_guesses.size()));

    // =========================================================================
    // PHASE 2: BASIN HOPPING / L-BFGS REFINEMENT
    // =========================================================================
    // (This remains almost exactly the same as your previous code!)
    
    std::vector<IterBest> global_candidates;
    std::mutex cand_mutex;
    std::atomic<int> completed_tasks{0};
    std::mutex print_mutex;
    const int bar_width = 50;

    #pragma omp parallel 
    {
        std::mt19937 rng(splitmix64(m_baseseed ^ omp_get_thread_num()));
        ECHOScore scorer = m_scorer_base;
        RDKit::ROMol localMol(m_original_mol);
        RDKit::Conformer &conf = localMol.getConformer();

        #pragma omp for schedule(dynamic, 1)
        for (int r = 0; r < actual_restarts; ++r) {
            
            // Get the winning Conformer ID for this Voxel!
            int cid = diverse_guesses[r].conf_id;
            const RDKit::Conformer& src_conf = m_original_mol.getConformer(cid);
            
            std::vector<RDGeom::Point3D> baseCoords;
            for (unsigned int i = 0; i < src_conf.getNumAtoms(); ++i) {
                baseCoords.push_back(src_conf.getAtomPos(i));
            }

            Eigen::RowVector3d ini_trans_xyz = diverse_guesses[r].trans;
            std::vector<double> x_start = diverse_guesses[r].x;
            
            BasinHopperState state(nTors, ini_trans_xyz, x_start, 0.0);

            // 1. Initial Funnel Slide (Restoring tight clash caps)
            auto [init_score, init_x] = runLocalLBFGS_Helper(
                x_start, state, scorer, pre, conf, baseCoords, config.repCap_inner_nm, inner_map_score);
            
            state.current_score = init_score;
            state.current_x = init_x;
            state.best_score = init_score;
            state.best_x = init_x;

            // 2. Minor Basin Hopping Jumps
            for (int jump = 0; jump < JUMPS_PER_RESTART; ++jump) {
                std::vector<double> x_mut = state.generate_kick(rng);
                auto [new_score, x_new_opt] = runLocalLBFGS_Helper(
                    x_mut, state, scorer, pre, conf, baseCoords, config.repCap_inner_nm, inner_map_score);
                state.attempt_step(x_new_opt, new_score, TEMPERATURE, rng);
            }

            state.apply_to_conformer(state.best_x, conf, baseCoords, pre);
            
            IterBest result;
            result.score = state.best_score;
            result.mol = RDKit::ROMol(localMol);
            
            {
                std::lock_guard<std::mutex> lock(cand_mutex);
                global_candidates.push_back(std::move(result));
            }

            // Progress Bar Update
            int completed = ++completed_tasks;
            {
                std::lock_guard<std::mutex> print_lock(print_mutex);
                float progress = static_cast<float>(completed) / actual_restarts;
                int pos = static_cast<int>(bar_width * progress);

                std::cout << "\rRefining |";
                for (int i = 0; i < bar_width; ++i) {
                    if (i < pos) std::cout << "█";
                    else if (i == pos) std::cout << "▌";
                    else std::cout << " ";
                }
                std::cout << "| " << std::setw(3) << static_cast<int>(progress * 100.0) << "% "
                          << "[" << completed << "/" << actual_restarts << " funnels]" << std::flush;
            }
        }
    }
    
    std::cout << "\n";

    // =========================================================================
    // PHASE 3: REPORTING & CLEANUP
    // =========================================================================
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    
    std::cout << "========================================\n"
              << "Docking Complete!\n"
              << "Voxels Evaluated: " << voxel_best_poses.size() << "\n"
              << "Total time taken: " << std::fixed << std::setprecision(2) << elapsed_seconds.count() << " seconds\n"
              << "========================================\n" << std::flush;

    std::sort(global_candidates.begin(), global_candidates.end(),
              [](const IterBest &A, const IterBest &B){ return A.score < B.score; });

    std::vector<IterBest> selected;
    for (const auto &cand : global_candidates) {
        if (selected.size() >= static_cast<std::size_t>(config.topN)) break;
        bool tooClose = false;
        for (const auto &sel : selected) {
            if (GeometryUtils::heavy_atom_rmsd(cand.mol, sel.mol) < config.rms_cutoff) { 
                tooClose = true; break; 
            }
        }
        if (!tooClose) selected.push_back(cand);
    }

    py::list out;
    for (const auto &cand : selected) {
      out.append(py::make_tuple(cand.score, conformerToCoords(cand.mol.getConformer())));
    }
    return out;
}

