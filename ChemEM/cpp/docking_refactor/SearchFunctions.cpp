#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include <deque>

#include <GraphMol/ROMol.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/MolTransforms/MolTransforms.h>
#include <Geometry/point.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Geometry> 
#include "SearchFunctions.h"
#include "GeometryUtils.h"
#include "nealderMead.h"

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}


void AntColonyOptimizer::set_max_iterations() {
    // number of torsion dimensions
    if (pre.config().iterations == 0) {
        auto n_torsions = pre.ligand_score().ligand_torsion_idxs.size();
        // number of heavy atoms in the ligand
        unsigned int n_heavy = m_original_mol.getNumHeavyAtoms();
        
        double raw = m_theta * 10.0 / 20
                   * (100.0 + 50.0 * double(n_torsions)
                            +  5.0 * double(n_heavy));
                            
        m_max_iterations = static_cast<unsigned int>(std::round(raw));
    } else { 
        m_max_iterations =  pre.config().iterations;
    }
    std::cout << "Docking with " << m_max_iterations << " iterations...\n";
}

void AntColonyOptimizer::initialize_pheromones() {
    
    // clear everything
    m_pheromones.clear();
    m_reset_array.clear();
    m_previous_deposits.clear();
    m_tmins.clear();
    
    size_t n_trans = pre.config().translation_points.size();
    const auto &all = pre.config().all_arrays;
    size_t n_tors = all.size();

    // total dimensions = 1 (translation) + rotation + torsion dims
    size_t n_dims = 1 + n_tors;

    m_pheromones.resize(n_dims);
    m_reset_array.resize(n_dims);
    m_previous_deposits.resize(n_dims);
    m_tmins.resize(n_dims);

    // --- dimension 0: translation ---
    {
        size_t len = n_trans;
        m_pheromones[0].assign(len, m_tmax);
        m_reset_array[0].assign(len, m_tmax);
        m_previous_deposits[0].clear();
        m_tmins[0] = 0.0;
    }

    // --- dimensions 1..nTors: rotation & torsions ---
    for (size_t d = 0; d < n_tors; ++d) {
        size_t len = all[d].size();
        auto &phe = m_pheromones[d+1];
        auto &res = m_reset_array[d+1];
        auto &prev = m_previous_deposits[d+1];

        phe.assign(len, m_tmax);
        res.assign(len, m_tmax);
        prev.clear();
        m_tmins[d+1] = 0.0;
    }
    
    //grab from pre no point copying TODO! DWD
    //adj = pre.Config().adjacency; 
    m_bj_indexes.assign(m_pheromones.size(), 0);
    m_pbest_probs.assign(m_pheromones.size(), 0.0);
    m_pbest.assign(m_pheromones.size(), 0.0);
    m_tmins.assign(m_pheromones.size(), 0.0);
    m_last_scores.clear();
    m_best_since_smoothing = std::numeric_limits<double>::infinity();
}


//initiliser
AntColonyOptimizer::AntColonyOptimizer(const PreComputedData &precomputed_data,
                                            const RDKit::ROMol &original_mol):
    pre(precomputed_data),
    m_original_mol(original_mol),
    m_scorer_base(precomputed_data),
    m_baseseed(1234567ULL),
    m_theta(0.25),
    m_rho(0.15),
    m_tmax(0.0), 
    m_sigma(0.5),
    m_evapRate(0.1),
    m_p_best(0.5),
    m_smoothingCount(0),
    m_flatWindowSize(10),
    
    m_ants_vec(pre.config().n_global_search) ,
    m_diversity_now(3)
    
{

    
   
    m_scorer_base.interaction_cutoff = pre.config().interaction_cutoff; 
    m_scorer_base.electro_clamp = pre.config().electro_clamp;
    
    set_max_iterations();
    m_dihedral_indices.reserve(pre.ligand_score().ligand_torsion_idxs.size());
    for (const auto &quad : pre.ligand_score().ligand_torsion_idxs) {
        m_dihedral_indices.push_back({
            static_cast<unsigned int>(quad[0]),
            static_cast<unsigned int>(quad[1]),
            static_cast<unsigned int>(quad[2]),
            static_cast<unsigned int>(quad[3])
        });
    }
    
    initialize_pheromones();
    
    const std::size_t n_vars = m_pheromones.size();
    for (auto& ant : m_ants_vec) {
        ant.sol.resize(n_vars); 
        ant.score = 0.0;
    }
    
    bestPerIter.reserve(m_max_iterations * m_diversity_now);
    
    //havent set ligandHeavyAtomIdx !
    /*
    ligandHeavyAtomIdx.clear();
    for (unsigned i = 0; i < originalMol.getNumAtoms(); ++i) {
        const auto *atom = originalMol.getAtomWithIdx(i);
        if (atom->getAtomicNum() != 1) {
            ligandHeavyAtomIdx.push_back(static_cast<int>(i));
        }
    }
    */
    if (pre.ligand_score().ligand_torsion_idxs.size() <= 10){
        m_final_eps = 0.25;
    } else {
        m_final_eps = 0.1;
    }
    
}




std::vector<std::pair<double, std::size_t>>
AntColonyOptimizer::get_probability_array(const std::vector<double>& phe, double alpha) const
{
    const std::size_t n = phe.size();
    std::vector<double> w(n, 0.0);

    double sumw = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double tau = std::max(0.0, phe[i]);
        const double wi  = std::pow(tau, alpha);
        w[i] = wi;
        sumw += wi;
    }

    std::vector<std::pair<double, std::size_t>> out;
    out.reserve(n);

    if (sumw > 0.0) {
        for (std::size_t i = 0; i < n; ++i) {
            out.emplace_back(w[i] / sumw, i);
        }
    } else {
        
        const double p = (n > 0) ? (1.0 / static_cast<double>(n)) : 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            out.emplace_back(p, i);
        }
    }

    // TODO! Keep sort, check with the update functions it may be possible to remove it
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    return out;
}


std::size_t AntColonyOptimizer::roulette_wheel(
    const std::vector<std::pair<double, std::size_t>>& probArr,
    std::mt19937& rng) const
{
    if (probArr.empty()) return 0;

    
    double total = 0.0;
    for (const auto& pi : probArr) total += pi.first;

    if (total <= 0.0) {
        // Degenerate fallback: return last index
        return probArr.back().second;
    }

    std::uniform_real_distribution<double> U(0.0, total);
    const double r = U(rng);

    double run = 0.0;
    for (const auto& pi : probArr) {
        run += pi.first;
        if (r <= run) return pi.second;
    }
    return probArr.back().second;
}


std::size_t AntColonyOptimizer::make_choice(
    const std::vector<double>& phe,
    std::mt19937& rng,
    double alpha) const
{
    auto pa = get_probability_array(phe, alpha);
    return roulette_wheel(pa, rng);
}


void AntColonyOptimizer::construct_solution(
    std::mt19937& rng,
    std::vector<double>& sol,
    double alpha) const
{
    // translation (dimension 0):
    const std::size_t t_idx = make_choice(m_pheromones[0], rng, alpha);
    sol[0] = static_cast<double>(t_idx);

    // rotation / torsion dims: store the chosen actual value from all_arrays[d][c]
    const std::size_t D = pre.config().all_arrays.size();
    for (std::size_t d = 0; d < D; ++d) {
        const std::size_t c = make_choice(m_pheromones[d + 1], rng, alpha);
        sol[d + 1] = pre.config().all_arrays[d][c];
    }
}



void AntColonyOptimizer::apply_ant_solution(RDKit::Conformer &conf,
                                           const std::vector<double> &sol) const
{
    // Expect: [tIdx, roll, pitch, yaw, torsions...]
    if (sol.size() < 4) {
        throw std::runtime_error("[Error] apply_ant_solution: sol.size() < 4");
    }

    const auto& tors = pre.ligand_score().ligand_torsion_idxs;
    if (sol.size() != 4 + tors.size()) {
        throw std::runtime_error("[Error] apply_ant_solution: sol size != 4 + #torsions");
    }

    
    //-----Global rotation (deg)
    
    const double roll = sol[1] * M_PI / 180.0;
    const double pitch = sol[2] * M_PI / 180.0;
    const double yaw = sol[3] * M_PI / 180.0;

    const Eigen::AngleAxisd R_x(roll, Eigen::Vector3d::UnitX());
    const Eigen::AngleAxisd R_y(pitch,Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd R_z(yaw, Eigen::Vector3d::UnitZ());
    const Eigen::Matrix3d R = (R_z * R_y * R_x).toRotationMatrix();

    const auto centroid = MolTransforms::computeCentroid(conf);

    for (unsigned i = 0; i < conf.getNumAtoms(); ++i) {
        const auto p = conf.getAtomPos(i);
        const Eigen::Vector3d v{p.x - centroid.x, p.y - centroid.y, p.z - centroid.z};
        const Eigen::Vector3d v2 = R * v;
        conf.setAtomPos(i, RDGeom::Point3D(v2.x() + centroid.x,
                                          v2.y() + centroid.y,
                                          v2.z() + centroid.z));
    }

   
    //-----Dihedral rotations (deg)
    
    for (std::size_t k = 0; k < tors.size(); ++k) {
        const auto& inds = tors[k];
        MolTransforms::setDihedralDeg(conf,
                                            inds[0], inds[1], inds[2], inds[3],
                                            sol[4 + k]);
    }

   
    // ----- Translation: sol[0] is an index (z,y,x) 
    
    const auto& tpts = pre.config().translation_points; // or pre.config(), be consistent
    const std::size_t tIdx = static_cast<std::size_t>(std::llround(sol[0]));
    if (tIdx >= tpts.size()) {
        throw std::runtime_error("[Error] apply_ant_solution: translation index out of range");
    }

    const auto tp = tpts[tIdx]; 

    
    const Eigen::RowVector3d gridIdx{ double(tp[2]), double(tp[1]), double(tp[0]) };

    Eigen::RowVector3d shift3d =
        pre.binding_site_grid().origin
      + gridIdx.cwiseProduct(pre.binding_site_grid().apix);

    // Move molecule so its centroid is at shift3d
    const auto newCent = MolTransforms::computeCentroid(conf);
    const RDGeom::Point3D delta(shift3d.x() - newCent.x,
                               shift3d.y() - newCent.y,
                               shift3d.z() - newCent.z);

    for (unsigned i = 0; i < conf.getNumAtoms(); ++i) {
        conf.setAtomPos(i, conf.getAtomPos(i) + delta);
    }
}

// Refactor: NM refinement evaluates deltas relative to the *discrete baseline pose*
// (baseline = originalMol with apply_ant_solution(discSol) applied once).
//
// Key changes vs your old approach:
//   1) No more RDKit::ROMol tmp(originalMol) inside every objective call
//   2) We reset conformer coords to the baseline pose each evaluation
//   3) We apply *delta* translation/rotation and set torsions to (base + delta)
//   4) Scoring is injected via ECHOScore + rep_max



// -----------------------------
// Helpers: snapshot/restore coords
// -----------------------------
static inline void snapshot_coords(const RDKit::Conformer& conf,
                                   std::vector<RDGeom::Point3D>& base) {
    const unsigned N = conf.getNumAtoms();
    base.resize(N);
    for (unsigned i = 0; i < N; ++i) base[i] = conf.getAtomPos(i);
}

static inline void restore_coords(RDKit::Conformer& conf,
                                  const std::vector<RDGeom::Point3D>& base) {
    const unsigned N = conf.getNumAtoms();
    // assume base.size() == N
    for (unsigned i = 0; i < N; ++i) conf.setAtomPos(i, base[i]);
}

// -----------------------------
// “Unpack” discrete ACO solution into base values
// (ini_rot and ini_tors are the *base* angles from discSol)
// -----------------------------
inline void unpackDiscreteSolution(
    const PreComputedData& pre,
    const std::vector<double>& sol,
    Eigen::RowVector3d &ini_trans_xyz, // (x,y,z) grid idx, from (z,y,x)
    Eigen::Vector3d &ini_rot_deg,   
    std::vector<double> &ini_tors_deg
) {
    const size_t tIdx = static_cast<size_t>(std::llround(sol[0]));
    const auto& tp_zyx = pre.config().translation_points[tIdx]; // {z,y,x}

    // convert to {x,y,z} indexing
    ini_trans_xyz = { double(tp_zyx[2]), double(tp_zyx[1]), double(tp_zyx[0]) };
    ini_rot_deg = Eigen::Vector3d{ sol[1], sol[2], sol[3] };
    ini_tors_deg.assign(sol.begin() + 4, sol.end());
}

// -----------------------------
// Apply *normalized deltas* on top of a baseline pose
// Baseline pose must already have discSol applied (translation/rot/tors)
// -----------------------------
static inline void applyNormalizedDeltasOnBaseline(
    const PreComputedData &pre,
    RDKit::Conformer& conf,
    const std::vector<double>& x_norm,        // [0..1]
    const std::vector<double>& base_tors_deg  // from discSol, size = nTors
) {
    // x_norm layout: [0..2]=T, [3..5]=R, [6..]=tors
    // Translation delta in ±max_shift Å
    constexpr double max_shift = 2.0;
    Eigen::RowVector3d dT;
    for (int i = 0; i < 3; ++i) dT[i] = (x_norm[i] * 2.0 - 1.0) * max_shift;

    // Apply translation delta
    for (unsigned i = 0; i < conf.getNumAtoms(); ++i) {
        const auto p = conf.getAtomPos(i);
        conf.setAtomPos(i, RDGeom::Point3D(p.x + dT[0], p.y + dT[1], p.z + dT[2]));
    }

    // Target centroid after translation (used to correct drift after torsions)
    const auto targetCent = MolTransforms::computeCentroid(conf);

    // Rotation delta in [-180, 180] degrees
    Eigen::Vector3d dR_deg;
    for (int i = 0; i < 3; ++i) dR_deg[i] = x_norm[3 + i] * 360.0 - 180.0;

    const double rx = dR_deg[0] * M_PI / 180.0;
    const double ry = dR_deg[1] * M_PI / 180.0;
    const double rz = dR_deg[2] * M_PI / 180.0;

    const Eigen::AngleAxisd R_x(rx, Eigen::Vector3d::UnitX());
    const Eigen::AngleAxisd R_y(ry, Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd R_z(rz, Eigen::Vector3d::UnitZ());
    const Eigen::Matrix3d R = (R_z * R_y * R_x).toRotationMatrix();

    // Rotate about current centroid
    {
        const auto c = MolTransforms::computeCentroid(conf);
        for (unsigned i = 0; i < conf.getNumAtoms(); ++i) {
            const auto p = conf.getAtomPos(i);
            const Eigen::Vector3d v{p.x - c.x, p.y - c.y, p.z - c.z};
            const Eigen::Vector3d v2 = R * v;
            conf.setAtomPos(i, RDGeom::Point3D(v2.x() + c.x, v2.y() + c.y, v2.z() + c.z));
        }
    }

    // Torsion deltas: set dihedral to (base + delta) in degrees
    const size_t nTors = base_tors_deg.size();
    for (size_t t = 0; t < nTors; ++t) {
        const double d_ang = x_norm[6 + t] * 360.0 - 180.0;
        double angle = std::fmod(base_tors_deg[t] + d_ang, 360.0);
        if (angle < 0) angle += 360.0;

        const auto& inds = pre.ligand_score().ligand_torsion_idxs[t];
        MolTransforms::setDihedralDeg(conf, inds[0], inds[1], inds[2], inds[3], angle);
    }

    // Recenter to the translated target centroid to remove drift from torsion operations
    {
        const auto c1 = MolTransforms::computeCentroid(conf);
        const RDGeom::Point3D shift(targetCent.x - c1.x,
                                    targetCent.y - c1.y,
                                    targetCent.z - c1.z);
        for (unsigned i = 0; i < conf.getNumAtoms(); ++i) {
            conf.setAtomPos(i, conf.getAtomPos(i) + shift);
        }
    }
}



std::vector<double> AntColonyOptimizer::convertRealSpaceToDiscrete(
    const std::vector<double> &orig_disc,
    const std::vector<double> &x_norm
) const {
    using Vec3d = Eigen::RowVector3d;

    // orig_disc layout: [tIdx, rot0, rot1, rot2, tors...]
    const size_t nTors = (orig_disc.size() >= 4) ? (orig_disc.size() - 4) : 0;
    const size_t expected_x = 6 + nTors; // x_norm: [0..2]=T, [3..5]=R, [6..]=tors
    if (orig_disc.size() < 4) {
        throw std::runtime_error("convertRealSpaceToDiscrete: orig_disc too small");
    }
    if (x_norm.size() != expected_x) {
        throw std::runtime_error(
            "convertRealSpaceToDiscrete: expected x_norm size " +
            std::to_string(expected_x) + " got " + std::to_string(x_norm.size()));
    }

    // ---- Translation: orig_disc[0] is discrete translation index into translation_points ----
    const auto &pts = pre.config().translation_points;
    const size_t tIdx = static_cast<size_t>(std::llround(orig_disc[0]));
    if (tIdx >= pts.size()) {
        throw std::runtime_error("convertRealSpaceToDiscrete: translation index out of range");
    }

    const auto tp = pts[tIdx]; // expected {z,y,x}

    const Vec3d apix   = pre.binding_site_grid().apix;
    const Vec3d origin = pre.binding_site_grid().origin;

    // real-space location of the chosen voxel (without origin)
    const Vec3d grid_real{
        double(tp[2]) * apix[0], // x
        double(tp[1]) * apix[1], // y
        double(tp[0]) * apix[2]  // z
    };

    // translation delta in ±2 Å, same convention as your NM code
    constexpr double max_shift = 2.0;
    Vec3d real_delta;
    for (int i = 0; i < 3; ++i) {
        real_delta[i] = (x_norm[i] * 2.0 - 1.0) * max_shift;
    }

    // target centroid (Cartesian Å)
    const Vec3d centroid3d = origin + grid_real + real_delta;

    // convert back to fractional grid coords (x,y,z) then to integer (x,y,z)
    const Vec3d disc_d = (centroid3d - origin).cwiseQuotient(apix);
    const Eigen::RowVector3i disc_i = disc_d.array().round().cast<int>();

    // want as {z,y,x} to compare against translation_points
    const std::array<int, 3> want{ disc_i[2], disc_i[1], disc_i[0] };

    // find closest precomputed translation point
    size_t best_t = 0;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < pts.size(); ++i) {
        const auto &p = pts[i];
        const double dz = double(p[0] - want[0]);
        const double dy = double(p[1] - want[1]);
        const double dx = double(p[2] - want[2]);
        const double d2 = dz*dz + dy*dy + dx*dx;
        if (d2 < best_d2) {
            best_d2 = d2;
            best_t  = i;
        }
    }

    std::vector<double> out;
    out.reserve(orig_disc.size());

    // new discrete translation index
    out.push_back(double(best_t));

    // ---- Rotations: orig_disc[1..3] + delta from x_norm[3..5] ----
    {
        Vec3d orig_rot{ orig_disc[1], orig_disc[2], orig_disc[3] };
        Vec3d delta_rot;
        for (int i = 0; i < 3; ++i) delta_rot[i] = x_norm[3 + i] * 360.0 - 180.0;

        Vec3d new_rot = orig_rot + delta_rot;

        for (int i = 0; i < 3; ++i) {
            new_rot[i] = std::fmod(new_rot[i], 360.0);
            if (new_rot[i] < 0) new_rot[i] += 360.0;

            // rotations are all_arrays[0..2]
            const auto &alist = pre.config().all_arrays[i];
            auto it = std::min_element(alist.begin(), alist.end(),
                [&](double a, double b) {
                    return std::abs(a - new_rot[i]) < std::abs(b - new_rot[i]);
                });
            out.push_back(*it);
        }
    }

    // ---- Torsions: orig_disc[4+t] + delta from x_norm[6+t] ----
    for (size_t t = 0; t < nTors; ++t) {
        const double base  = orig_disc[4 + t];
        const double d_ang = x_norm[6 + t] * 360.0 - 180.0;

        double a = std::fmod(base + d_ang, 360.0);
        if (a < 0) a += 360.0;

        const size_t idx = 3 + t; // torsions start at all_arrays[3]
        if (idx >= pre.config().all_arrays.size()) {
            throw std::runtime_error("convertRealSpaceToDiscrete: torsion all_arrays index OOR");
        }

        const auto &tlist = pre.config().all_arrays[idx];
        auto it = std::min_element(tlist.begin(), tlist.end(),
            [&](double x, double y) {
                return std::abs(x - a) < std::abs(y - a);
            });

        out.push_back(*it);
    }

    return out;
}

// -----------------------------
// Refined result
// -----------------------------
struct Refined {
    double realScore = std::numeric_limits<double>::infinity();
    std::vector<double> realNorm;
    std::vector<double> discSol;
    RDKit::ROMol realMol;
};
/*
// -----------------------------
// NM refinement from a discrete champion, with injected scorer + rep_max
// Assumes you have:
//   - apply_ant_solution(pre, conf, discSol)  (discrete -> baseline pose)
//   - convertRealSpaceToDiscrete(discSol, x_norm)
//   - NelderMeadOptimizer with nm.optimize(obj, simplex, iters, ftol, xtol)
// -----------------------------
AntColonyOptimizer::SplitNmResult
AntColonyOptimizer::refinePoseSplitNmFromDiscrete(
    const std::vector<double> &discSol,
    const ECHOScore &scorer,
    double rep_max
) const {
    // --- base params derived from discSol ---
    Eigen::RowVector3d ini_trans_xyz;
    Eigen::Vector3d    ini_rot_deg;
    std::vector<double> base_tors_deg;

    unpackDiscreteSolution(pre, discSol, ini_trans_xyz, ini_rot_deg, base_tors_deg);
    
    const size_t nTors = base_tors_deg.size();
    const size_t D = 6 + nTors;

    // --- workMol baseline pose: originalMol + discrete solution (once) ---
    RDKit::ROMol workMol(m_original_mol);
    RDKit::Conformer& conf = workMol.getConformer();
    apply_ant_solution(conf, discSol);

    std::vector<RDGeom::Point3D> baseline_coords;
    snapshot_coords(conf, baseline_coords);

    // Baseline x_norm = 0.5 (zero deltas)
    std::vector<double> x_full(D, 0.05);

    auto eval_full = [&](const std::vector<double>& x_norm) -> double {
        restore_coords(conf, baseline_coords);
        applyNormalizedDeltasOnBaseline(pre, conf, x_norm, base_tors_deg);
        return scorer.score(conf, rep_max);
    };

    double bestScore = eval_full(x_full);

    // ------------------------------
    // TR-only Nelder–Mead (6 dims)
    // ------------------------------
    std::vector<double> x0_TR(6, 0.5);
    const double eps_TR = 0.05; //0.25;

    std::vector<std::vector<double>> simplexTR(6 + 1, x0_TR);
    for (size_t i = 0; i < 6; ++i) simplexTR[i + 1][i] = std::clamp(x0_TR[i] + eps_TR, 0.0, 1.0);

    std::vector<double> x_tmp(D, 0.5); // reused buffer to avoid reallocation in objectives

    auto objTR = [&](const std::vector<double>& x_tr) -> double {
        std::fill(x_tmp.begin(), x_tmp.end(), 0.5);
        for (int i = 0; i < 6; ++i) x_tmp[i] = x_tr[i];
        return eval_full(x_tmp);
    };

    std::vector<double> x_after_TR(D, 0.5);

    NelderMeadOptimizer nmTR(6);
    auto resTR = nmTR.optimize(objTR, simplexTR, 10000, 1e-6, 1e-6);

    if (resTR.best_value < bestScore) {
        bestScore = resTR.best_value;
        for (int i = 0; i < 6; ++i) x_after_TR[i] = resTR.best_point[i];
    }

    // ------------------------------
    // (2) Torsion-only Nelder–Mead (nTors dims)
    // ------------------------------
    std::vector<double> x_final = x_after_TR;

    if (nTors > 0) {
        std::vector<double> x0_tor(nTors, 0.5);
        const double eps_tor = 0.05;//0.25;

        std::vector<std::vector<double>> simplexTor(nTors + 1, x0_tor);
        for (size_t i = 0; i < nTors; ++i) simplexTor[i + 1][i] = std::clamp(x0_tor[i] + eps_tor, 0.0, 1.0);

        auto objTor = [&](const std::vector<double>& x_tor) -> double {
            // fixed TR
            std::copy(x_after_TR.begin(), x_after_TR.begin() + 6, x_tmp.begin());
            // torsions variable
            for (size_t t = 0; t < nTors; ++t) x_tmp[6 + t] = x_tor[t];
            return eval_full(x_tmp);
        };

        NelderMeadOptimizer nmTor(static_cast<int>(nTors));
        auto resTor = nmTor.optimize(objTor, simplexTor, 10000, 1e-6, 1e-6);

        if (resTor.best_value < bestScore) {
            bestScore = resTor.best_value;
            for (size_t t = 0; t < nTors; ++t) x_final[6 + t] = resTor.best_point[t];
        }
    }

    // ------------------------------
    // (3) Build final molecule + refined discrete vector
    // ------------------------------
    // Build refined discrete vector for ACO update
    std::vector<double> disc_refined = convertRealSpaceToDiscrete(discSol, x_final);

    // Produce finalMol by applying best deltas on baseline
    restore_coords(conf, baseline_coords);
    applyNormalizedDeltasOnBaseline(pre, conf, x_final, base_tors_deg);

    SplitNmResult out;
    out.score   = bestScore;
    out.mol     = std::move(workMol);     // now contains refined coordinates
    out.discSol = std::move(disc_refined);
    return out;
}
*/


//new v1
/*
AntColonyOptimizer::SplitNmResult
AntColonyOptimizer::refinePoseSplitNmFromDiscrete(
    const std::vector<double> &discSol,
    const ECHOScore &scorer,
    double rep_max
) const {
    // --- base params derived from discSol ---
    Eigen::RowVector3d ini_trans_xyz;
    Eigen::Vector3d    ini_rot_deg;
    std::vector<double> base_tors_deg;
    unpackDiscreteSolution(pre, discSol, ini_trans_xyz, ini_rot_deg, base_tors_deg);

    const size_t nTors = base_tors_deg.size();
    const size_t D     = 6 + nTors;

    // --- workMol baseline pose: originalMol + discrete solution (once) ---
    RDKit::ROMol workMol(m_original_mol);
    RDKit::Conformer& conf = workMol.getConformer();
    apply_ant_solution(conf, discSol);

    std::vector<RDGeom::Point3D> baseline_coords;
    snapshot_coords(conf, baseline_coords);

    auto clamp01_inplace = [](std::vector<double>& x) {
        for (double &v : x) v = std::clamp(v, 0.0, 1.0);
    };
    
    auto make_simplex = [&](const std::vector<double>& x0, double eps) {
        const size_t dim = x0.size();
        std::vector<std::vector<double>> simplex(dim + 1, x0);
        for (size_t i = 0; i < dim; ++i) {
            simplex[i + 1][i] = std::clamp(x0[i] + eps, 0.0, 1.0);
        }
        return simplex;
    };

    // Baseline x_norm = 0.5 (zero deltas)
    std::vector<double> x_tmp(D, 0.5);

    auto eval_full = [&](const std::vector<double>& x_norm_in) -> double {
        std::vector<double> x_norm = x_norm_in;
        clamp01_inplace(x_norm);

        restore_coords(conf, baseline_coords);
        applyNormalizedDeltasOnBaseline(pre, conf, x_norm, base_tors_deg);
        return scorer.score(conf, rep_max);
    };

    std::vector<double> x_best_full(D, 0.5);
    double bestScore = eval_full(x_best_full);

    // ------------------------------
    // deterministic RNG per discSol (safe under OpenMP)
    // ------------------------------
    uint64_t seed = splitmix64(m_baseseed ^ (0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(D)));
    auto mix_q = [&](double v, double scale) {
        const int64_t q = static_cast<int64_t>(std::llround(v * scale));
        seed = splitmix64(seed ^ static_cast<uint64_t>(q));
    };
    mix_q(ini_trans_xyz[0], 1000.0);
    mix_q(ini_trans_xyz[1], 1000.0);
    mix_q(ini_trans_xyz[2], 1000.0);
    mix_q(ini_rot_deg[0],   100.0);
    mix_q(ini_rot_deg[1],   100.0);
    mix_q(ini_rot_deg[2],   100.0);
    for (size_t i = 0; i < std::min<size_t>(nTors, 16); ++i) mix_q(base_tors_deg[i], 10.0);

    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<double> unif(-1.0, 1.0);

    // Helper: xtol is L2-radius of simplex about best in your NM:
    // max_dist_sq is max over vertices of sum_j (x_i[j]-best[j])^2
    // So we scale xtol ~ perCoordTol * sqrt(dim) to keep per-coordinate precision consistent.
    auto xtol_from_percoord = [](double perCoordTol, size_t dim) {
        return perCoordTol * std::sqrt(static_cast<double>(std::max<size_t>(1, dim)));
    };

    // ------------------------------
    // (1) TR-only staged Nelder–Mead (6 dims)
    // ------------------------------
    std::vector<double> x_tr_best(6, 0.5);

    auto objTR = [&](const std::vector<double>& x_tr_in) -> double {
        std::fill(x_tmp.begin(), x_tmp.end(), 0.5);
        for (int i = 0; i < 6; ++i) x_tmp[i] = std::clamp(x_tr_in[i], 0.0, 1.0);
        return eval_full(x_tmp);
    };

    // Tightened eps: 0.03 -> 0.015 -> 0.0075
    // In physical terms (from your mapping):
    //  eps=0.03 => ~0.12 Å and ~10.8° per coordinate
    //  eps=0.0075 => ~0.03 Å and ~2.7° per coordinate
    struct StageTR { double eps; int iters; double ftol; double xtol; int jitters; double jitterFrac; };
    const std::vector<StageTR> trStages = {
        {0.030,  900, 2e-4, xtol_from_percoord(8e-4, 6), 2, 0.60},
        {0.015,  700, 1e-4, xtol_from_percoord(5e-4, 6), 2, 0.50},
        {0.0075, 600, 5e-5, xtol_from_percoord(3e-4, 6), 1, 0.40}
    };

    NelderMeadOptimizer nmTR(6);

    for (const auto& st : trStages) {
        // main
        {
            auto simplex = make_simplex(x_tr_best, st.eps);
            auto res = nmTR.optimize(objTR, simplex, st.iters, st.ftol, st.xtol);
            if (res.best_value < bestScore) {
                bestScore = res.best_value;
                x_tr_best = res.best_point;
                clamp01_inplace(x_tr_best);
            }
        }
        // jitter restarts
        for (int r = 0; r < st.jitters; ++r) {
            std::vector<double> x0 = x_tr_best;
            for (int i = 0; i < 6; ++i) {
                const double j = st.jitterFrac * st.eps * unif(rng);
                x0[i] = std::clamp(x0[i] + j, 0.0, 1.0);
            }
            auto simplex = make_simplex(x0, st.eps);
            auto res = nmTR.optimize(objTR, simplex, std::max(250, st.iters / 2), st.ftol, st.xtol);
            if (res.best_value < bestScore) {
                bestScore = res.best_value;
                x_tr_best = res.best_point;
                clamp01_inplace(x_tr_best);
            }
        }
    }

    std::vector<double> x_after_TR(D, 0.5);
    for (int i = 0; i < 6; ++i) x_after_TR[i] = x_tr_best[i];

    // ------------------------------
    // (2) Torsion-only staged Nelder–Mead (nTors dims)
    // ------------------------------
    std::vector<double> x_final = x_after_TR;

    if (nTors > 0) {
        std::vector<double> x_tor_best(nTors, 0.5);

        auto objTor = [&](const std::vector<double>& x_tor_in) -> double {
            // fixed TR
            std::copy(x_after_TR.begin(), x_after_TR.begin() + 6, x_tmp.begin());
            // torsions
            for (size_t t = 0; t < nTors; ++t) x_tmp[6 + t] = std::clamp(x_tor_in[t], 0.0, 1.0);
            return eval_full(x_tmp);
        };

        const bool manyTors = (nTors > 10);

        // Tightened eps for torsions:
        //  - many tors: 0.02 -> 0.01 -> 0.005  (7.2° -> 3.6° -> 1.8°)
        //  - few tors : 0.025 -> 0.0125 -> 0.00625
        const double eps0 = manyTors ? 0.020  : 0.025;
        const double eps1 = manyTors ? 0.010  : 0.0125;
        const double eps2 = manyTors ? 0.005  : 0.00625;

        // Tighter per-coordinate xtol: ~0.36° * 0.001 = 0.36°
        // We go down to perCoordTol=3e-4 in final => ~0.11° per torsion coordinate.
        const double xt0 = xtol_from_percoord(8e-4, nTors);
        const double xt1 = xtol_from_percoord(5e-4, nTors);
        const double xt2 = xtol_from_percoord(3e-4, nTors);

        auto itersTor = [&]() -> int {
            // modest scaling with dimension
            const int base = 300;
            const int perD = 45;
            return std::min(base + perD * static_cast<int>(nTors), 1800);
        };

        struct StageTor { double eps; int iters; double ftol; double xtol; int jitters; double jitterFrac; };
        const std::vector<StageTor> torStages = {
            {eps0, itersTor(),     2e-4, xt0, manyTors ? 3 : 2, 0.60},
            {eps1, itersTor() / 2, 1e-4, xt1, manyTors ? 2 : 1, 0.50},
            {eps2, itersTor() / 2, 5e-5, xt2, manyTors ? 1 : 0, 0.40}
        };

        NelderMeadOptimizer nmTor(static_cast<int>(nTors));

        for (const auto& st : torStages) {
            // main
            {
                auto simplex = make_simplex(x_tor_best, st.eps);
                auto res = nmTor.optimize(objTor, simplex, st.iters, st.ftol, st.xtol);
                if (res.best_value < bestScore) {
                    bestScore = res.best_value;
                    x_tor_best = res.best_point;
                    clamp01_inplace(x_tor_best);

                    x_final = x_after_TR;
                    for (size_t t = 0; t < nTors; ++t) x_final[6 + t] = x_tor_best[t];
                }
            }
            // jitter restarts
            for (int r = 0; r < st.jitters; ++r) {
                std::vector<double> x0 = x_tor_best;
                for (size_t t = 0; t < nTors; ++t) {
                    const double j = st.jitterFrac * st.eps * unif(rng);
                    x0[t] = std::clamp(x0[t] + j, 0.0, 1.0);
                }
                auto simplex = make_simplex(x0, st.eps);
                auto res = nmTor.optimize(objTor, simplex, std::max(250, st.iters / 2), st.ftol, st.xtol);
                if (res.best_value < bestScore) {
                    bestScore = res.best_value;
                    x_tor_best = res.best_point;
                    clamp01_inplace(x_tor_best);

                    x_final = x_after_TR;
                    for (size_t t = 0; t < nTors; ++t) x_final[6 + t] = x_tor_best[t];
                }
            }
        }
    }

    // ------------------------------
    // (3) Build final molecule + refined discrete vector
    // ------------------------------
    std::vector<double> disc_refined = convertRealSpaceToDiscrete(discSol, x_final);

    restore_coords(conf, baseline_coords);
    applyNormalizedDeltasOnBaseline(pre, conf, x_final, base_tors_deg);

    SplitNmResult out;
    out.score   = bestScore;
    out.mol     = std::move(workMol);
    out.discSol = std::move(disc_refined);
    return out;
}
*/
AntColonyOptimizer::SplitNmResult
AntColonyOptimizer::refinePoseSplitNmFromDiscrete(
    const std::vector<double> &discSol,
    const ECHOScore &scorer,
    double rep_max,
    double map_score_function
) const {
    Eigen::RowVector3d ini_trans_xyz;
    Eigen::Vector3d    ini_rot_deg;
    std::vector<double> base_tors_deg;
    unpackDiscreteSolution(pre, discSol, ini_trans_xyz, ini_rot_deg, base_tors_deg);

    const size_t nTors = base_tors_deg.size();
    const size_t D     = 6 + nTors;
    const bool manyTors = (nTors > 10);

    RDKit::ROMol workMol(m_original_mol);
    RDKit::Conformer& conf = workMol.getConformer();
    apply_ant_solution(conf, discSol);

    std::vector<RDGeom::Point3D> baseline_coords;
    snapshot_coords(conf, baseline_coords);

    auto clamp01_inplace = [](std::vector<double>& x) {
        for (double &v : x) v = std::clamp(v, 0.0, 1.0);
    };

    // FIX: avoid degenerate simplex near bounds
    auto make_simplex = [&](const std::vector<double>& x0, double eps) {
        const size_t dim = x0.size();
        std::vector<std::vector<double>> simplex(dim + 1, x0);
        for (size_t i = 0; i < dim; ++i) {
            double v = x0[i] + eps;
            if (v > 1.0) v = x0[i] - eps;
            simplex[i + 1][i] = std::clamp(v, 0.0, 1.0);
        }
        return simplex;
    };

    std::vector<double> x_tmp(D, 0.5);

    auto eval_full = [&](const std::vector<double>& x_norm_in) -> double {
        std::vector<double> x_norm = x_norm_in;
        clamp01_inplace(x_norm);
        restore_coords(conf, baseline_coords);
        applyNormalizedDeltasOnBaseline(pre, conf, x_norm, base_tors_deg);
        return scorer.score(conf, rep_max, map_score_function);
    };

    std::vector<double> x_best_full(D, 0.5);
    double bestScore = eval_full(x_best_full);

    uint64_t seed = splitmix64(m_baseseed ^ (0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(D)));
    auto mix_q = [&](double v, double scale) {
        const int64_t q = static_cast<int64_t>(std::llround(v * scale));
        seed = splitmix64(seed ^ static_cast<uint64_t>(q));
    };
    mix_q(ini_trans_xyz[0], 1000.0);
    mix_q(ini_trans_xyz[1], 1000.0);
    mix_q(ini_trans_xyz[2], 1000.0);
    mix_q(ini_rot_deg[0],   100.0);
    mix_q(ini_rot_deg[1],   100.0);
    mix_q(ini_rot_deg[2],   100.0);
    for (size_t i = 0; i < std::min<size_t>(nTors, 16); ++i) mix_q(base_tors_deg[i], 10.0);

    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<double> unif(-1.0, 1.0);

    auto xtol_from_percoord = [](double perCoordTol, size_t dim) {
        return perCoordTol * std::sqrt(static_cast<double>(std::max<size_t>(1, dim)));
    };

    // ------------------------------
    // (1) TR-only staged NM (6 dims) — reduced budgets + early stop
    // ------------------------------
    std::vector<double> x_tr_best(6, 0.5);

    auto objTR = [&](const std::vector<double>& x_tr_in) -> double {
        std::fill(x_tmp.begin(), x_tmp.end(), 0.5);
        for (int i = 0; i < 6; ++i) x_tmp[i] = std::clamp(x_tr_in[i], 0.0, 1.0);
        return eval_full(x_tmp);
    };

    struct StageTR { double eps; int iters; double ftol; double xtol; int jitters; double jitterFrac; };
    const std::vector<StageTR> trStages = {
        {0.030, 600, 3e-4, xtol_from_percoord(9e-4, 6), 1, 0.55},
        {0.015, 450, 2e-4, xtol_from_percoord(6e-4, 6), 1, 0.45},
        {0.0075,350, 1e-4, xtol_from_percoord(4e-4, 6), 0, 0.30}
    };

    NelderMeadOptimizer nmTR(6);

    for (const auto& st : trStages) {
        const double before = bestScore;

        // main
        {
            auto simplex = make_simplex(x_tr_best, st.eps);
            auto res = nmTR.optimize(objTR, simplex, st.iters, st.ftol, st.xtol);
            if (res.best_value < bestScore) {
                bestScore = res.best_value;
                x_tr_best = res.best_point;
                clamp01_inplace(x_tr_best);
            }
        }

        const double improve = before - bestScore;
        if (improve <= 2.0 * st.ftol) {
            // already local; stop TR early
            if (st.eps <= 0.015) break;
            continue;
        }

        // jitter (only if main helped)
        for (int r = 0; r < st.jitters; ++r) {
            std::vector<double> x0 = x_tr_best;
            for (int i = 0; i < 6; ++i) {
                const double j = st.jitterFrac * st.eps * unif(rng);
                x0[i] = std::clamp(x0[i] + j, 0.0, 1.0);
            }
            auto simplex = make_simplex(x0, st.eps);
            auto res = nmTR.optimize(objTR, simplex, std::max(200, st.iters / 2), st.ftol, st.xtol);
            if (res.best_value < bestScore) {
                bestScore = res.best_value;
                x_tr_best = res.best_point;
                clamp01_inplace(x_tr_best);
            }
        }
    }

    std::vector<double> x_after_TR(D, 0.5);
    for (int i = 0; i < 6; ++i) x_after_TR[i] = x_tr_best[i];

    // ------------------------------
    //  Torsion-only staged NM — reduced budgets + early stop
    // ------------------------------
    std::vector<double> x_final = x_after_TR;

    if (nTors > 0) {
        std::vector<double> x_tor_best(nTors, 0.5);

        auto objTor = [&](const std::vector<double>& x_tor_in) -> double {
            std::copy(x_after_TR.begin(), x_after_TR.begin() + 6, x_tmp.begin());
            for (size_t t = 0; t < nTors; ++t) x_tmp[6 + t] = std::clamp(x_tor_in[t], 0.0, 1.0);
            return eval_full(x_tmp);
        };

        // Keep  eps ladder, but reduce restarts/iters and loosen late ftol slightly for &&.
        const double eps0 = manyTors ? 0.020  : 0.025;
        const double eps1 = manyTors ? 0.010  : 0.0125;
        const double eps2 = manyTors ? 0.005  : 0.00625;

        auto itersTor = [&]() -> int {
            const int base = 180;
            const int perD = 22;
            return std::min(base + perD * static_cast<int>(nTors), 900);
        };

        const double xt0 = xtol_from_percoord(9e-4, nTors);
        const double xt1 = xtol_from_percoord(6e-4, nTors);
        const double xt2 = xtol_from_percoord(4e-4, nTors);

        struct StageTor { double eps; int iters; double ftol; double xtol; int jitters; double jitterFrac; };
        const std::vector<StageTor> torStages = {
            {eps0, itersTor(),     3e-4, xt0, manyTors ? 2 : 1, 0.55},
            {eps1, itersTor() / 2, 2e-4, xt1, manyTors ? 1 : 0, 0.45},
            {eps2, itersTor() / 2, 1e-4, xt2, 0,               0.30}
        };

        NelderMeadOptimizer nmTor(static_cast<int>(nTors));

        for (const auto& st : torStages) {
            const double before = bestScore;

            // main
            {
                auto simplex = make_simplex(x_tor_best, st.eps);
                auto res = nmTor.optimize(objTor, simplex, st.iters, st.ftol, st.xtol);
                if (res.best_value < bestScore) {
                    bestScore = res.best_value;
                    x_tor_best = res.best_point;
                    clamp01_inplace(x_tor_best);

                    x_final = x_after_TR;
                    for (size_t t = 0; t < nTors; ++t) x_final[6 + t] = x_tor_best[t];
                }
            }

            const double improve = before - bestScore;
            if (improve <= 2.0 * st.ftol) {
                if (st.eps <= eps1) break; // already local; stop torsion early
                continue;
            }

            // jitter (only if main helped)
            for (int r = 0; r < st.jitters; ++r) {
                std::vector<double> x0 = x_tor_best;
                for (size_t t = 0; t < nTors; ++t) {
                    const double j = st.jitterFrac * st.eps * unif(rng);
                    x0[t] = std::clamp(x0[t] + j, 0.0, 1.0);
                }
                auto simplex = make_simplex(x0, st.eps);
                auto res = nmTor.optimize(objTor, simplex, std::max(200, st.iters / 2), st.ftol, st.xtol);
                if (res.best_value < bestScore) {
                    bestScore = res.best_value;
                    x_tor_best = res.best_point;
                    clamp01_inplace(x_tor_best);

                    x_final = x_after_TR;
                    for (size_t t = 0; t < nTors; ++t) x_final[6 + t] = x_tor_best[t];
                }
            }
        }
    }

    std::vector<double> disc_refined = convertRealSpaceToDiscrete(discSol, x_final);

    restore_coords(conf, baseline_coords);
    applyNormalizedDeltasOnBaseline(pre, conf, x_final, base_tors_deg);

    SplitNmResult out;
    out.score   = bestScore;
    out.mol     = std::move(workMol);
    out.discSol = std::move(disc_refined);
    return out;
}


// Helper: your score is "lower is better" and usually negative.
// Convert to a positive "quality" deposit amount.
static inline double deposit_quality_from_score(double score) {
    // good: -5 -> 5, mediocre: -1 -> 1, bad/positive -> 0
    return (score < 0.0) ? (-score) : 0.0;
}

// Put this helper in the same .cpp (e.g., near other small helpers)
static inline size_t find_nearest_index(const std::vector<double>& vals, double x) {
    if (vals.empty()) {
        throw std::runtime_error("updateBjIndexes: empty bin array");
    }
    size_t best_i = 0;
    double best_d = std::abs(vals[0] - x);
    for (size_t i = 1; i < vals.size(); ++i) {
        const double d = std::abs(vals[i] - x);
        if (d < best_d) { best_d = d; best_i = i; }
    }
    return best_i;
}

void AntColonyOptimizer::updateBjIndexes(const std::vector<double> &sol) {
    // torsion count (your file uses ligand_score().ligand_torsion_idxs)
    const size_t nTors = pre.ligand_score().ligand_torsion_idxs.size();
    const size_t expected = 4 + nTors; // [tIdx, r1, r2, r3, tors...]
    if (sol.size() != expected) {
        throw std::runtime_error(
            "updateBjIndexes: expected sol size " + std::to_string(expected) +
            " got " + std::to_string(sol.size()));
    }

    // all_arrays dims should be 3 + nTors in your code path
    const size_t D = pre.config().all_arrays.size();
    if (D + 1 != m_pheromones.size()) {
        throw std::runtime_error("updateBjIndexes: m_pheromones size != 1 + all_arrays size");
    }

    if (m_bj_indexes.size() != m_pheromones.size()) {
        m_bj_indexes.assign(m_pheromones.size(), 0);
    }

    // 0) Translation index already discrete
    const size_t tIdx = static_cast<size_t>(std::llround(sol[0]));
    if (tIdx >= m_pheromones[0].size()) {
        throw std::runtime_error("updateBjIndexes: translation index out of range");
    }
    m_bj_indexes[0] = tIdx;

    // 1) Rotations + torsions: sol[d+1] maps into pre.config().all_arrays[d]
    for (size_t d = 0; d < D; ++d) {
        const auto &vals = pre.config().all_arrays[d]; // vector<double>
        const size_t idx = find_nearest_index(vals, sol[d + 1]);

        if (idx >= m_pheromones[d + 1].size()) {
            throw std::runtime_error("updateBjIndexes: bin index out of range for dim " + std::to_string(d + 1));
        }
        m_bj_indexes[d + 1] = idx;
    }

    // Optional sanity (debug)
    // for (size_t d = 0; d < m_bj_indexes.size(); ++d) {
    //     if (m_bj_indexes[d] >= m_pheromones[d].size())
    //         throw std::runtime_error("updateBjIndexes: m_bj_indexes[" + std::to_string(d) + "] OOR post-check");
    // }
}

void AntColonyOptimizer::updateBestSinceSmoothing(double score) {
    // Track best (minimum) score since last (soft/hard) reset.
    if (score < m_best_since_smoothing) {
        m_best_since_smoothing = score;
    }
}

void AntColonyOptimizer::updateTauLimitsFromBest_PLANTS() {
    // tau_max: keep your steady-state rationale:
    // if the best bin gets deposit ~= Q each iteration:
    // tau -> (1-rho)tau + Q, steady state tau* = Q/rho.
    const double q_best = deposit_quality_from_score(m_best_since_smoothing);

    // Guard: if we haven't seen a finite best yet
    if (!std::isfinite(q_best) || q_best <= 0.0) {
        m_tmax = 1.0; // safe fallback
    } else {
        m_tmax = q_best / m_rho;
    }

    const size_t nDims = m_pheromones.size();
    if (m_tmins.size() != nDims) m_tmins.assign(nDims, 0.0);

    // MMAS/PLANTS idea: use fixed p_best to derive tau_min from tau_max.
    // A common form is:
    //   tau_min = tau_max * (1 - p^(1/n)) / ((n_i - 1) * p^(1/n))
    // where n is number of decisions, n_i is number of choices in that decision.
    //
    // (This is the classic MMAS "Equation 11" usage of p_best.)  :contentReference[oaicite:2]{index=2}
    const double p_term = std::pow(m_p_best, 1.0 / double(std::max<size_t>(1, nDims)));

    for (size_t d = 0; d < nDims; ++d) {
        const double nChoices = static_cast<double>(m_pheromones[d].size());
        if (nChoices <= 1.0 || p_term <= 0.0) {
            m_tmins[d] = m_tmax;
            continue;
        }

        double tmin = m_tmax * (1.0 - p_term) / ((nChoices - 1.0) * p_term);

        // Safety: clamp to [0, tmax]
        if (!std::isfinite(tmin) || tmin < 0.0) tmin = 0.0;
        if (tmin > m_tmax) tmin = m_tmax;

        m_tmins[d] = tmin;
    }
}

void AntColonyOptimizer::updatePreviousDeposits_lastOnly() {
    // IMPORTANT: avoid unbounded growth. Your old code kept appending forever.
    // We only need the last index per dimension for your neighbor-deposit logic.
    const size_t nDims = m_bj_indexes.size();
    if (m_previous_deposits.size() != nDims) m_previous_deposits.assign(nDims, {});

    for (size_t d = 0; d < nDims; ++d) {
        auto &prev = m_previous_deposits[d];
        prev.clear();
        prev.push_back(static_cast<unsigned>(m_bj_indexes[d]));
    }
}

void AntColonyOptimizer::updatePheromones(double rawScoreWeighted) {
    const double Q = deposit_quality_from_score(rawScoreWeighted);

    for (size_t d = 0; d < m_pheromones.size(); ++d) {
        auto &phe = m_pheromones[d];

        const double tmin = (d < m_tmins.size()) ? m_tmins[d] : 0.0;
        const double tmax = m_tmax;

        // Build deposit set around "last" deposited bin
        std::vector<size_t> depositSet;
        depositSet.reserve(8);

        if (d < m_previous_deposits.size() && !m_previous_deposits[d].empty()) {
            const size_t last = static_cast<size_t>(m_previous_deposits[d].back());
            if (last < phe.size()) {
                if (d == 0) {
                    // translation: include last + grid-neighbors
                    depositSet.push_back(last);
                    if (last < pre.config().adjacency.size()) {
                        for (size_t nb :  pre.config().adjacency[last]) {
                            if (nb < phe.size()) depositSet.push_back(nb);
                        }
                    }
                } else {
                    // rotations/torsions: +-2 wrap-around
                    const size_t L = phe.size();
                    for (int off : {-2, -1, 0, 1, 2}) {
                        const size_t idx = (last + L + size_t(off)) % L;
                        depositSet.push_back(idx);
                    }
                }
            }
        }

        // Evaporate + deposit
        for (size_t i = 0; i < phe.size(); ++i) {
            const bool inSet = (std::find(depositSet.begin(), depositSet.end(), i) != depositSet.end());
            const double delta = (inSet ? Q : 0.0);

            double tnew = (1.0 - m_rho) * phe[i] + delta;
            phe[i] = std::clamp(tnew, tmin, tmax);
        }
    }
}

bool AntColonyOptimizer::isFlatWindow() const {
    if (m_last_scores.size() < static_cast<size_t>(m_flatWindowSize)) return false;

    auto [itMin, itMax] = std::minmax_element(m_last_scores.begin(), m_last_scores.end());
    const double span = *itMax - *itMin;

    // Your heuristic: <= 2% of magnitude at low end
    const double denom = std::max(1e-12, std::abs(*itMin));
    return span <= 0.02 * denom;
}

void AntColonyOptimizer::smoothPheromones() {
    // Soft reset toward tau_max
    for (auto &dim : m_pheromones) {
        for (double &tau : dim) {
            tau += m_sigma * (m_tmax - tau);
        }
    }
}

void AntColonyOptimizer::updateSmoothing(double iterBestScore) {
    // Maintain rolling window
    if (m_last_scores.size() == static_cast<size_t>(m_flatWindowSize)) {
        m_last_scores.pop_front();
    }
    m_last_scores.push_back(iterBestScore);

    if (!isFlatWindow()) return;

    // Soft vs hard
    if (m_smoothingCount >= 3) {
        m_pheromones = m_reset_array; // hard reset
        m_smoothingCount = 0;
    } else {
        smoothPheromones();           // soft reset
        ++m_smoothingCount;
    }

    // restart monitoring + best tracking
    m_last_scores.clear();
    m_best_since_smoothing = std::numeric_limits<double>::infinity();
}

// Apply a normalized Nelder–Mead vector x_norm onto a pose defined by
// (ini_trans_xyz, ini_rot_deg, ini_tors_deg).
//
// Conventions used here:
//   - ini_trans_xyz is an (x,y,z) grid index (already flipped from tp_zyx if needed)
//   - pre.binding_site_apix is (ax, ay, az) in Å per grid step
//   - pre.grid_origin is (ox, oy, oz) in Å (Cartesian)
//   - x_norm layout: [0..2]=translation delta, [3..5]=rotation delta, [6..]=torsion deltas
//
// Effect:
//   - Translates molecule so its centroid lands at target centroid3d
//   - Applies global rotation about centroid
//   - Applies torsions (base + delta)
//   - Recenters again to remove drift caused by torsion ops
//
static inline void applyNormalizedSolution(
    const PreComputedData &pre,
    RDKit::Conformer          &conf,
    const std::vector<double> &x_norm,        // [0..1] NM vars
    const Eigen::RowVector3d  &ini_trans_xyz, // (x,y,z) grid index
    const Eigen::Vector3d     &ini_rot_deg,   // base rotation (deg)
    const std::vector<double> &ini_tors_deg   // base torsions (deg)
) {
    // 1) grid -> real Cartesian (Å) component of centroid
    const auto apix = pre.binding_site_grid().apix; // expects 3-vector (ax,ay,az)
    const Eigen::RowVector3d grid_real{
        ini_trans_xyz[0] * apix[0],
        ini_trans_xyz[1] * apix[1],
        ini_trans_xyz[2] * apix[2]
    };

    // 2) translation delta in ±max_shift Å from x_norm[0..2]
    constexpr double max_shift = 2.0;
    Eigen::RowVector3d real_delta;
    for (int i = 0; i < 3; ++i) {
        real_delta[i] = (x_norm[i] * 2.0 - 1.0) * max_shift;
    }

    // 3) target centroid in Cartesian Å
    const Eigen::RowVector3d centroid3d =
        pre.binding_site_grid().origin + grid_real + real_delta;

    // 4) translate molecule so centroid lands at centroid3d
    {
        const auto c0 = MolTransforms::computeCentroid(conf);
        const RDGeom::Point3D shift0{
            centroid3d.x() - c0.x,
            centroid3d.y() - c0.y,
            centroid3d.z() - c0.z
        };

        for (unsigned i = 0; i < conf.getNumAtoms(); ++i) {
            conf.setAtomPos(i, conf.getAtomPos(i) + shift0);
        }
    }

    // 5) apply global rotation:
    //    rot = ini_rot_deg + delta, where delta in [-180,180] degrees from x_norm[3..5]
    Eigen::Vector3d rot_delta_deg;
    for (int i = 0; i < 3; ++i) {
        rot_delta_deg[i] = x_norm[3 + i] * 360.0 - 180.0;
    }

    Eigen::Vector3d rot_deg = ini_rot_deg + rot_delta_deg;
    for (int i = 0; i < 3; ++i) {
        rot_deg[i] = std::fmod(rot_deg[i], 360.0);
        if (rot_deg[i] < 0.0) rot_deg[i] += 360.0;
    }

    {
        const double rx = rot_deg[0] * M_PI / 180.0;
        const double ry = rot_deg[1] * M_PI / 180.0;
        const double rz = rot_deg[2] * M_PI / 180.0;

        const Eigen::AngleAxisd R_x(rx, Eigen::Vector3d::UnitX());
        const Eigen::AngleAxisd R_y(ry, Eigen::Vector3d::UnitY());
        const Eigen::AngleAxisd R_z(rz, Eigen::Vector3d::UnitZ());
        const Eigen::Matrix3d   R = (R_z * R_y * R_x).toRotationMatrix();

        const auto center = MolTransforms::computeCentroid(conf);

        for (unsigned i = 0; i < conf.getNumAtoms(); ++i) {
            const auto p = conf.getAtomPos(i);
            const Eigen::Vector3d v{ p.x - center.x, p.y - center.y, p.z - center.z };
            const Eigen::Vector3d v2 = R * v;
            conf.setAtomPos(i, RDGeom::Point3D(v2.x() + center.x,
                                              v2.y() + center.y,
                                              v2.z() + center.z));
        }
    }

    // 6) torsions: angle = base + delta, delta in [-180,180] degrees from x_norm[6..]
    for (size_t t = 0; t < ini_tors_deg.size(); ++t) {
        const double d_ang = x_norm[6 + t] * 360.0 - 180.0;

        double angle = std::fmod(ini_tors_deg[t] + d_ang, 360.0);
        if (angle < 0.0) angle += 360.0;

        const auto &inds = pre.ligand_score().ligand_torsion_idxs[t];
        MolTransforms::setDihedralDeg(conf,
                                             inds[0], inds[1], inds[2], inds[3],
                                             angle);
    }

    // 7) recenter to centroid3d to remove numerical drift from torsion ops
    {
        const auto c1 = MolTransforms::computeCentroid(conf);
        const RDGeom::Point3D shift1{
            centroid3d.x() - c1.x,
            centroid3d.y() - c1.y,
            centroid3d.z() - c1.z
        };

        for (unsigned i = 0; i < conf.getNumAtoms(); ++i) {
            conf.setAtomPos(i, conf.getAtomPos(i) + shift1);
        }
    }
}

/*
std::pair<double, RDKit::ROMol>
AntColonyOptimizer::runLocalNelderMeadFromSeeds(
    const Eigen::RowVector3d &ini_trans_xyz,
    const Eigen::Vector3d    &ini_rot_deg,
    const std::vector<double> &ini_tors_deg,
    const ECHOScore &scorer,
    double rep_max
) const
{
    const size_t D = 6 + ini_tors_deg.size();

    RDKit::ROMol workMol(m_original_mol);
    RDKit::Conformer &conf = workMol.getConformer();

    // snapshot original coords once
    std::vector<RDGeom::Point3D> baseCoords;
    snapshot_coords(conf, baseCoords);

    std::vector<double> x0(D, 0.5);
    const double eps = 0.01; // from 0.25

    std::vector<std::vector<double>> simplex(D + 1, x0);
    for (size_t i = 0; i < D; ++i) {
        simplex[i + 1][i] = std::clamp(x0[i] + eps, 0.0, 1.0); //changes from eps
    }

    auto obj = [&](const std::vector<double> &x_norm) -> double {
        restore_coords(conf, baseCoords);

        applyNormalizedSolution(
            pre,                 // <-- your PrecomputedDataCPP2 (member)
            conf,
            x_norm,
            ini_trans_xyz,        // <-- MUST be xyz grid index
            ini_rot_deg,
            ini_tors_deg
        );

        return scorer.score(conf, rep_max);
    };

    NelderMeadOptimizer nm(static_cast<int>(D));
    auto result = nm.optimize(obj, simplex, 10000, 1e-8, 1e-8);

    // finalize best pose into workMol, then move it out
    restore_coords(conf, baseCoords);
    applyNormalizedSolution(pre, conf, result.best_point, ini_trans_xyz, ini_rot_deg, ini_tors_deg);

    return { result.best_value, std::move(workMol) };
}
*/


std::pair<double, RDKit::ROMol>
AntColonyOptimizer::runLocalNelderMeadFromSeeds(
    const Eigen::RowVector3d &ini_trans_xyz,
    const Eigen::Vector3d    &ini_rot_deg,
    const std::vector<double> &ini_tors_deg,
    const ECHOScore &scorer,
    double rep_max,
    double map_score_function
) const
{
    const size_t nTors = ini_tors_deg.size();
    const size_t D = 6 + nTors;

    RDKit::ROMol workMol(m_original_mol);
    RDKit::Conformer &conf = workMol.getConformer();

    std::vector<RDGeom::Point3D> baseCoords;
    snapshot_coords(conf, baseCoords);

    auto clamp01_inplace = [](std::vector<double>& x) {
        for (double &v : x) v = std::clamp(v, 0.0, 1.0);
    };

    // FIX: avoid degenerate simplex near bounds (try -eps if +eps would clamp)
    auto make_simplex = [&](const std::vector<double>& x0, double eps) {
        std::vector<std::vector<double>> simplex(D + 1, x0);
        for (size_t i = 0; i < D; ++i) {
            double v = x0[i] + eps;
            if (v > 1.0) v = x0[i] - eps;
            simplex[i + 1][i] = std::clamp(v, 0.0, 1.0);
        }
        return simplex;
    };

    auto obj = [&](const std::vector<double> &x_norm_in) -> double {
        std::vector<double> x_norm = x_norm_in;
        clamp01_inplace(x_norm);

        restore_coords(conf, baseCoords);
        applyNormalizedSolution(pre, conf, x_norm, ini_trans_xyz, ini_rot_deg, ini_tors_deg);
        return scorer.score(conf, rep_max, map_score_function);
    };

    // -----------------------------
    // Staged restart schedule (speed-tuned for && stopping)
    // -----------------------------
    const bool manyTors = (nTors > 10);

    const double eps_fine   = manyTors ? 0.005 : 0.010;
    const double eps_ultra  = 0.5 * eps_fine;   // keep, but we’ll exit early if stalled
    const double eps_mid    = 2.0 * eps_fine;
    const double eps_coarse = 4.0 * eps_fine;

    auto xtol_from_percoord = [&](double perCoordTol) {
        return perCoordTol * std::sqrt(static_cast<double>(std::max<size_t>(1, D)));
    };

    // Slightly looser than eps/10 so xtol is not the bottleneck under &&
    auto xtol_for_eps = [&](double eps) {
        const double per = std::max(3e-4, eps / 6.0);
        return xtol_from_percoord(per);
    };

    // With &&, ftol must also pass; keep it not-too-tight in late stages for speed.
    auto ftol_for_stage = [&](double eps) {
        if (eps >= eps_mid)   return 3e-4;
        if (eps >= eps_fine)  return 2e-4;
        return 1e-4; // was 5e-5 (often becomes the slow bottleneck under &&)
    };

    // Smaller iteration budgets: multiple stages + restarts already
    auto iters_for = [&](double /*eps*/) -> int {
        const int base = 160;
        const int perD = 22;
        const int it = base + perD * static_cast<int>(D);
        return std::min(it, 900);
    };

    struct Stage {
        double eps;
        int iters;
        double ftol;
        double xtol;
        int jitters;
        double jitterFrac;
    };

    // Fewer restarts (big speed win)
    const int jit_coarse = manyTors ? 2 : 1;
    const int jit_mid    = manyTors ? 1 : 0;
    const int jit_fine   = 0;
    const int jit_ultra  = 0;

    std::vector<Stage> stages = {
        { eps_coarse, iters_for(eps_coarse), ftol_for_stage(eps_coarse), xtol_for_eps(eps_coarse), jit_coarse, 0.70 },
        { eps_mid,    iters_for(eps_mid),    ftol_for_stage(eps_mid),    xtol_for_eps(eps_mid),    jit_mid,    0.55 },
        { eps_fine,   iters_for(eps_fine),   ftol_for_stage(eps_fine),   xtol_for_eps(eps_fine),   jit_fine,   0.35 },
        { eps_ultra,  std::max(250, iters_for(eps_ultra) / 2),
                      ftol_for_stage(eps_ultra), xtol_for_eps(eps_ultra), jit_ultra, 0.20 }
    };

    // deterministic RNG per-seed
    uint64_t seed = splitmix64(m_baseseed ^ (0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(D)));
    auto mix_q = [&](double v, double scale) {
        const int64_t q = static_cast<int64_t>(std::llround(v * scale));
        seed = splitmix64(seed ^ static_cast<uint64_t>(q));
    };
    mix_q(ini_trans_xyz[0], 1000.0);
    mix_q(ini_trans_xyz[1], 1000.0);
    mix_q(ini_trans_xyz[2], 1000.0);
    mix_q(ini_rot_deg[0],   100.0);
    mix_q(ini_rot_deg[1],   100.0);
    mix_q(ini_rot_deg[2],   100.0);
    for (size_t i = 0; i < std::min<size_t>(nTors, 16); ++i) mix_q(ini_tors_deg[i], 10.0);

    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<double> unif(-1.0, 1.0);

    NelderMeadOptimizer nm(static_cast<int>(D));

    std::vector<double> x_best(D, 0.5);
    double f_best = obj(x_best);

    for (const auto& st : stages) {
        const double f_before = f_best;

        // main run
        {
            auto simplex = make_simplex(x_best, st.eps);
            auto res = nm.optimize(obj, simplex, st.iters, st.ftol, st.xtol);
            if (res.best_value < f_best) {
                f_best = res.best_value;
                x_best = res.best_point;
                clamp01_inplace(x_best);
            }
        }

        const double improve_main = f_before - f_best;

        // SPEED: if main stage didn’t improve meaningfully, skip jitter and often stop early
        if (improve_main <= 2.0 * st.ftol) {
            // If we’re already at fine/ultra scales, bail out early
            if (st.eps <= eps_mid) break;
            continue;
        }

        // jitter restarts (shorter runs)
        for (int r = 0; r < st.jitters; ++r) {
            std::vector<double> x0 = x_best;
            for (size_t i = 0; i < D; ++i) {
                const double j = st.jitterFrac * st.eps * unif(rng);
                x0[i] = std::clamp(x0[i] + j, 0.0, 1.0);
            }

            auto simplex = make_simplex(x0, st.eps);
            const int it_short = std::max(200, st.iters / 2);

            auto res = nm.optimize(obj, simplex, it_short, st.ftol, st.xtol);
            if (res.best_value < f_best) {
                f_best = res.best_value;
                x_best = res.best_point;
                clamp01_inplace(x_best);
            }
        }
    }

    restore_coords(conf, baseCoords);
    applyNormalizedSolution(pre, conf, x_best, ini_trans_xyz, ini_rot_deg, ini_tors_deg);

    return { f_best, std::move(workMol) };
}

static py::array_t<double> conformerToCoords(const RDKit::Conformer &conf) {
    py::ssize_t N = conf.getNumAtoms();
    // shape = { N, 3 }
    // strides = { 3*sizeof(double), sizeof(double) } for C‐order layout
    std::initializer_list<py::ssize_t> shape   = { N, 3 };
    std::initializer_list<py::ssize_t> strides = {
        3 * sizeof(double),
        sizeof(double)
    };
    py::array_t<double> arr(shape, strides);
    auto buf = arr.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < N; ++i) {
        auto p = conf.getAtomPos(i);
        buf(i,0) = p.x;
        buf(i,1) = p.y;
        buf(i,2) = p.z;
    }
    return arr;
}


py::list AntColonyOptimizer::optimize() {
    
    const auto &config = pre.config();//TODO!! do you still need this?
    
    const int inner_map_score = config.inner_map_score;
    const int outer_map_score = config.outer_map_score;
    for (unsigned int iter = 0; iter < m_max_iterations; ++iter) {
        //set alpha value lower at the start to increase breath of search
        //need to pass this as its not being set anymore
        double alpha_now = get_alpha_now(iter);
        
        //cap on repulsion gets tighter as iteration increases
        double t = (m_max_iterations <= 1) ? 1.0 : static_cast<double>(iter) / (m_max_iterations - 1);
        double repCap_discrete = config.repCap0 + (config.repCap1 - config.repCap0) * (t * t);
        
        //set num threads
        if (t > 0.75) {
            m_diversity_now = 1;
        }
        omp_set_num_threads(config.n_cpu);
        
        #pragma omp parallel 
        {
            //make sure this works
            static thread_local std::mt19937 tl_rng;
            static thread_local bool tl_seeded = false;
            if (!tl_seeded) {
                int tid = omp_get_thread_num();
                uint64_t s = splitmix64(m_baseseed ^ uint64_t(tid));
                // mt19937 takes 32-bit seed fine; you can also seed_seq if you want
                tl_rng.seed(static_cast<uint32_t>(s));
                tl_seeded = true;
            }
            ECHOScore scorer = m_scorer_base;
            #pragma omp for schedule(static)
            for (int a = 0; a < static_cast<int>(config.n_global_search); ++a) {
                 auto& ant = m_ants_vec[a];
                 construct_solution(tl_rng, ant.sol, alpha_now);
                
                 RDKit::ROMol mol(m_original_mol); 
                 apply_ant_solution(mol.getConformer(), ant.sol);
                 ant.score = scorer.score(mol.getConformer(), repCap_discrete, inner_map_score);
                 
             }
        }//end omp para 
        
        
        //sort ants by best scoring
        const unsigned K = std::min<unsigned>(config.n_local_search, m_ants_vec.size());
        if (K==0) {
            continue;
        }
        std::partial_sort(
          m_ants_vec.begin(), m_ants_vec.begin() + K, m_ants_vec.end(),
          [](const auto &A,const auto &B){ return A.score < B.score; }
        );
        
        const int nThreads = std::min<int>(K, config.n_cpu);
        using RowVector3d = Eigen::Matrix<double,1,3,Eigen::RowMajor>;
        
        
        std::vector<Refined> refined(K);

        #pragma omp parallel num_threads(nThreads) 
        {
            ECHOScore scorer = m_scorer_base; 
        
            #pragma omp for schedule(dynamic,1)
            for (int k = 0; k < static_cast<int>(K); ++k) {
                const auto& champ = m_ants_vec[k];
                SplitNmResult res = refinePoseSplitNmFromDiscrete(champ.sol, scorer, config.repCap_inner_nm, inner_map_score);
        
                refined[k].realScore = res.score;
                refined[k].realMol   = std::move(res.mol);
                refined[k].discSol   = std::move(res.discSol);
                refined[k].realNorm.clear();
            }
        }
        
        // Rank refined solutions by score 
        std::vector<size_t> rorder(refined.size());
        std::iota(rorder.begin(), rorder.end(), 0);
        std::sort(rorder.begin(), rorder.end(),
                  [&](size_t i, size_t j){
                    return refined[i].realScore < refined[j].realScore;
                  });

        auto &best = refined[rorder[0]];
        //  Store top-K-per-iteration refined poses as seeds for the final return stage
        {
          const unsigned keepCount = std::min<unsigned>(m_diversity_now, (unsigned)rorder.size());
          for (unsigned i = 0; i < keepCount; ++i) {
            const auto &cand = refined[rorder[i]];
            bestPerIter.push_back(IterBest{ cand.realScore, cand.discSol, cand.realMol });
          }
        }
        
        std::vector<size_t> updateIdx;
        updateIdx.reserve(std::min<unsigned>(m_diversity_now, (unsigned)rorder.size()));
        
        if (m_diversity_now == 1) {
            updateIdx.push_back(rorder[0]);
        } else {
            for (size_t k = 0; k < rorder.size() && updateIdx.size() < m_diversity_now; ++k) {
                const size_t cand = rorder[k];
                //updateIdx.push_back(cand);
                //make with rmsd diversity
                bool tooClose = false;
                for (size_t chosen : updateIdx) {
                    double rmsd = GeometryUtils::heavy_atom_rmsd(refined[cand].realMol, refined[chosen].realMol);
                    if (rmsd < config.rms_cutoff) { tooClose = true; break; }
                }
                if (!tooClose) updateIdx.push_back(cand);
            }
        }
        
        
        updateBestSinceSmoothing(best.realScore);
        updateTauLimitsFromBest_PLANTS();

        {
            const size_t M = updateIdx.size();
            double wsum = 0.0;
            for (size_t r = 0; r < M; ++r) {
                wsum += rankWeightedUpdate ? double(M - r) : 1.0;
            }
            
            if (wsum <= 0.0) wsum = 1.0;
            for (size_t r = 0; r < M; ++r) {
                const size_t idx = updateIdx[r];
                const double w = (rankWeightedUpdate ? double(M - r) : 1.0) / wsum;
        
                // 1) select bins for this candidate
                updateBjIndexes(refined[idx].discSol);
        
                // 2) set "last deposit" indices (used by neighbor depositSet logic)
                updatePreviousDeposits_lastOnly();
        
                // 3) weighted deposit: score is negative => good => deposit positive
                // updatePheromones converts negative rawScore to +deposit internally
                updatePheromones(refined[idx].realScore * w);
            }
        }

        updateSmoothing(best.realScore);
        std::cout << "Iter " << iter << "  realScore= " << best.realScore << "\n";
        }//end iteration loop 
    

    //change rep_Cap here to repCap_final_nm 
    std::sort(
      bestPerIter.begin(), bestPerIter.end(),
      [](const IterBest &A, const IterBest &B){
        return A.score < B.score;
      }
    );
    
    py::list out;
    std::vector<IterBest> selected;
    selected.reserve(config.topN);
    
    // ---- Final polish with full MI score (no shared mutation) ----
    const unsigned extraSeeds = config.topN * 2;
    const unsigned seedCount =
        std::min<unsigned>(static_cast<unsigned>(bestPerIter.size()), config.topN + extraSeeds);
    
    // Collect all candidates produced by final polish (NM only)
    std::vector<IterBest> candidateList;
    candidateList.reserve(seedCount);
    
    const int nThreads = std::max(1, config.n_cpu);
    #pragma omp parallel num_threads(nThreads)
    {
        std::vector<IterBest> localCandidates;
        localCandidates.reserve(16);
        ECHOScore scorer = m_scorer_base;
        #pragma omp for schedule(dynamic,1)
        for (int i = 0; i < static_cast<int>(seedCount); ++i) {
            const auto &seed = bestPerIter[static_cast<std::size_t>(i)];
            Eigen::RowVector3d   ini_trans_xyz;
            Eigen::Vector3d      ini_rot_deg;
            std::vector<double>  ini_tors_deg;
            unpackDiscreteSolution(
                pre,
                seed.discSol,
                ini_trans_xyz,
                ini_rot_deg,
                ini_tors_deg
            );
            // Plain combined NM refinement around this seed
            auto [nmScore, nmMol] = runLocalNelderMeadFromSeeds(
                ini_trans_xyz, ini_rot_deg, ini_tors_deg,
                scorer,
                config.repCap_final_nm,
                outer_map_score
            );
            
            IterBest cand;
            cand.score   = nmScore;
            cand.discSol = seed.discSol;     // keep original discrete seed (or replace if you also map back)
            cand.mol     = std::move(nmMol);
    
            localCandidates.push_back(std::move(cand));
        }
        #pragma omp critical
        {
            candidateList.insert(
                candidateList.end(),
                std::make_move_iterator(localCandidates.begin()),
                std::make_move_iterator(localCandidates.end())
            );
        }
    }//end para
    // Sort candidates by score ascending (lower is better)
    std::sort(candidateList.begin(), candidateList.end(),
              [](const IterBest &A, const IterBest &B){ return A.score < B.score; });
    
    // Greedily select up to nReturn, ensuring uniqueness by RMSD
    for (const auto &cand : candidateList) {
        if (selected.size() >= static_cast<std::size_t>(config.topN)) break;
    
        bool tooClose = false;
        for (const auto &sel : selected) {
            const double rmsd = GeometryUtils::heavy_atom_rmsd(cand.mol, sel.mol);
            if (rmsd < config.rms_cutoff) { tooClose = true; break; }
        }
        if (!tooClose) selected.push_back(cand);
    }

    //need conformer to coords
    for (const auto &cand : selected) {
      auto coords = conformerToCoords(cand.mol.getConformer());
      out.append(py::make_tuple(cand.score, coords));
    }

    return out;
}
     

