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

#include <iostream>
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
#include "LbfgsRefine.h"   // R1: numerical-gradient bounded L-BFGS local optimizer (docking_v2)

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


// Manual dihedral setter equivalent to MolTransforms::setDihedralDeg, but with the moved-atom set
// (the l-side of the j-k bond) PRECOMPUTED — that is the graph BFS setDihedralDeg redoes every call.
// Uses RDKit's own getDihedralRad for the current angle (so the dihedral convention matches exactly),
// then rotates `moved` about the j->k axis through k by (target - current). Verified byte-identical
// to setDihedralDeg (ctor self-check, ~1e-13). `i` participates only via getDihedralRad.
static inline void fast_set_dihedral_deg(RDKit::Conformer& conf,
                                         unsigned i, unsigned j, unsigned k, unsigned l,
                                         double target_deg,
                                         const std::vector<unsigned int>& moved) {
    const double cur = MolTransforms::getDihedralRad(conf, i, j, k, l);
    const double delta = target_deg * M_PI / 180.0 - cur;
    const RDGeom::Point3D jp = conf.getAtomPos(j);
    const RDGeom::Point3D kp = conf.getAtomPos(k);
    Eigen::Vector3d axis(kp.x - jp.x, kp.y - jp.y, kp.z - jp.z);
    const double nrm = axis.norm();
    if (nrm < 1e-12) return;
    axis /= nrm;
    const Eigen::Matrix3d R = Eigen::AngleAxisd(delta, axis).toRotationMatrix();
    const Eigen::Vector3d kc(kp.x, kp.y, kp.z);
    for (unsigned a : moved) {
        const RDGeom::Point3D p = conf.getAtomPos(a);
        const Eigen::Vector3d w = R * Eigen::Vector3d(p.x - kc.x(), p.y - kc.y(), p.z - kc.z()) + kc;
        conf.setAtomPos(a, RDGeom::Point3D(w.x(), w.y(), w.z()));
    }
}

// forward decls (defined later in this TU) so the grid minimiser above the ctor can use them
static inline void snapshot_coords(const RDKit::Conformer& conf, std::vector<RDGeom::Point3D>& base);
static inline void restore_coords(RDKit::Conformer& conf, const std::vector<RDGeom::Point3D>& base);

// ---- Analytic pose-variable gradient + retraction for the grid minimiser -----------------------
// Pose variables: [dt(3, Å), dθ(3, rad about all-atom centroid), dφ(N, rad per torsion)].
// From per-atom forces f_i = d(score)/d(pos_i): translation grad = Σ f_i; rotation grad = Σ (p_i−c)×f_i
// (torque); torsion t grad = axis_t · Σ_{i∈moved[t]} (p_i−k_t)×f_i. axis_t = normalize(k−j).
// TEMP recall diagnostic: heavy-atom RMSD of a conformer to the native reference (same atom order,
// first ref.rows() atoms). Not symmetry-corrected — a good near/far signal, not an exact number.
static inline double rmsd_to_ref(const RDKit::Conformer& conf, const MatX3d& ref) {
    const int nh = static_cast<int>(ref.rows());
    if (nh == 0) return std::numeric_limits<double>::infinity();
    double s = 0.0;
    for (int i = 0; i < nh; ++i) {
        const auto p = conf.getAtomPos(i);
        const double dx = p.x - ref(i,0), dy = p.y - ref(i,1), dz = p.z - ref(i,2);
        s += dx*dx + dy*dy + dz*dz;
    }
    return std::sqrt(s / nh);
}

static inline Eigen::RowVector3d all_atom_centroid(const RDKit::Conformer& conf) {
    const int N = conf.getNumAtoms();
    Eigen::RowVector3d c(0,0,0);
    for (int i = 0; i < N; ++i) { const auto p = conf.getAtomPos(i); c += Eigen::RowVector3d(p.x,p.y,p.z); }
    return c / static_cast<double>(std::max(1, N));
}

static inline void grid_pose_gradient(const RDKit::Conformer& conf,
                                      const std::vector<Eigen::RowVector3d>& forces,
                                      const std::vector<std::array<unsigned int,4>>& tors_idx,
                                      const std::vector<std::vector<unsigned int>>& tors_moved,
                                      std::vector<double>& grad) {
    const int N = conf.getNumAtoms();
    const int T = static_cast<int>(tors_idx.size());
    grad.assign(6 + T, 0.0);
    const Eigen::RowVector3d c = all_atom_centroid(conf);
    Eigen::RowVector3d gt(0,0,0), gr(0,0,0);
    for (int i = 0; i < N; ++i) {
        const auto p = conf.getAtomPos(i);
        const Eigen::RowVector3d pi(p.x, p.y, p.z);
        gt += forces[i];
        gr += (pi - c).cross(forces[i]);
    }
    grad[0]=gt.x(); grad[1]=gt.y(); grad[2]=gt.z();
    grad[3]=gr.x(); grad[4]=gr.y(); grad[5]=gr.z();
    for (int t = 0; t < T; ++t) {
        const auto jp = conf.getAtomPos(tors_idx[t][1]);
        const auto kp = conf.getAtomPos(tors_idx[t][2]);
        Eigen::RowVector3d axis(kp.x-jp.x, kp.y-jp.y, kp.z-jp.z);
        const double n = axis.norm();
        if (n < 1e-12) { grad[6+t] = 0.0; continue; }
        axis /= n;
        const Eigen::RowVector3d kc(kp.x, kp.y, kp.z);
        Eigen::RowVector3d tau(0,0,0);
        for (unsigned i : tors_moved[t]) {
            const auto p = conf.getAtomPos(i);
            tau += (Eigen::RowVector3d(p.x,p.y,p.z) - kc).cross(forces[i]);
        }
        grad[6+t] = axis.dot(tau);
    }
}

// Apply a pose increment (retraction): torsions, then rigid rotation about centroid, then translation.
static inline void grid_retract(RDKit::Conformer& conf,
                                const std::vector<std::array<unsigned int,4>>& tors_idx,
                                const std::vector<std::vector<unsigned int>>& tors_moved,
                                const std::vector<double>& delta) {
    const int N = conf.getNumAtoms();
    const int T = static_cast<int>(tors_idx.size());
    for (int t = 0; t < T; ++t) {
        const double ang = delta[6 + t];
        if (std::fabs(ang) < 1e-15) continue;
        const auto jp = conf.getAtomPos(tors_idx[t][1]);
        const auto kp = conf.getAtomPos(tors_idx[t][2]);
        Eigen::Vector3d axis(kp.x-jp.x, kp.y-jp.y, kp.z-jp.z);
        const double n = axis.norm();
        if (n < 1e-12) continue;
        axis /= n;
        const Eigen::Matrix3d R = Eigen::AngleAxisd(ang, axis).toRotationMatrix();
        const Eigen::Vector3d kc(kp.x, kp.y, kp.z);
        for (unsigned i : tors_moved[t]) {
            const auto p = conf.getAtomPos(i);
            const Eigen::Vector3d w = R * Eigen::Vector3d(p.x-kc.x(), p.y-kc.y(), p.z-kc.z()) + kc;
            conf.setAtomPos(i, RDGeom::Point3D(w.x(), w.y(), w.z()));
        }
    }
    const Eigen::Vector3d dth(delta[3], delta[4], delta[5]);
    const double ath = dth.norm();
    if (ath > 1e-15) {
        const Eigen::RowVector3d cr = all_atom_centroid(conf);
        const Eigen::Vector3d c(cr.x(), cr.y(), cr.z());
        const Eigen::Matrix3d R = Eigen::AngleAxisd(ath, dth / ath).toRotationMatrix();
        for (int i = 0; i < N; ++i) {
            const auto p = conf.getAtomPos(i);
            const Eigen::Vector3d w = R * Eigen::Vector3d(p.x-c.x(), p.y-c.y(), p.z-c.z()) + c;
            conf.setAtomPos(i, RDGeom::Point3D(w.x(), w.y(), w.z()));
        }
    }
    const RDGeom::Point3D dt(delta[0], delta[1], delta[2]);
    if (delta[0] != 0.0 || delta[1] != 0.0 || delta[2] != 0.0)
        for (int i = 0; i < N; ++i) conf.setAtomPos(i, conf.getAtomPos(i) + dt);
}

// Compare the analytic pose gradient of score_fast_grad against central finite differences.
// Moves the ligand into the binding-site box first so the grids are active. Returns max rel error.
double AntColonyOptimizer::verify_grid_gradient() const {
    RDKit::ROMol base(m_original_mol);
    RDKit::Conformer& cbase = base.getConformer();
    {   // centre the ligand on the site so trilinear lookups are in-box
        const Eigen::RowVector3d c = all_atom_centroid(cbase);
        const auto& site = pre.config().binding_site_centroid;
        const RDGeom::Point3D sh(site.x()-c.x(), site.y()-c.y(), site.z()-c.z());
        for (unsigned i = 0; i < cbase.getNumAtoms(); ++i) cbase.setAtomPos(i, cbase.getAtomPos(i) + sh);
    }
    ECHOScore scorer = m_scorer_base;
    std::vector<Eigen::RowVector3d> forces;
    std::vector<double> g;
    scorer.score_fast_grad(cbase, forces);
    grid_pose_gradient(cbase, forces, m_dihedral_indices, m_torsion_moved, g);

    const int D = 6 + static_cast<int>(m_dihedral_indices.size());
    const double eps = 1e-5;
    double maxrel = 0.0;
    for (int j = 0; j < D; ++j) {
        std::vector<double> dp(D, 0.0), dm(D, 0.0);
        dp[j] = eps; dm[j] = -eps;
        RDKit::ROMol mp(base), mm(base);
        RDKit::Conformer& cp = mp.getConformer();
        RDKit::Conformer& cm = mm.getConformer();
        grid_retract(cp, m_dihedral_indices, m_torsion_moved, dp);
        grid_retract(cm, m_dihedral_indices, m_torsion_moved, dm);
        std::vector<Eigen::RowVector3d> ft;
        const double fp = scorer.score_fast_grad(cp, ft);
        const double fm = scorer.score_fast_grad(cm, ft);
        const double fd = (fp - fm) / (2.0 * eps);
        const double rel = std::fabs(g[j] - fd) / (std::fabs(fd) + 1e-6);
        maxrel = std::max(maxrel, rel);
        const char* grp = (j < 3) ? "trans" : (j < 6) ? "rot  " : "tors ";
        std::cerr << "[gridgrad] " << grp << " var " << j
                  << " analytic=" << g[j] << " fd=" << fd << " rel=" << rel << "\n";
    }
    std::cerr << "[gridgrad] D=" << D << " max rel err = " << maxrel << "\n";
    return maxrel;
}

// Analytic-gradient L-BFGS on the fast grid score, with retraction (Vina-style: optimise pose-var
// increments applied to the current pose). Cheap because the gradient is analytic (one score_fast_grad
// per accepted step) — no finite differences, no protein neighbour loop. Minimises conf in place.
double AntColonyOptimizer::grid_minimise(RDKit::Conformer& conf, const ECHOScore& scorer,
                                         int max_iters) const {
    const int T = static_cast<int>(m_dihedral_indices.size());
    const int D = 6 + T;
    std::vector<Eigen::RowVector3d> forces;
    std::vector<double> g(D), gnew(D), dir(D), delta(D, 0.0);
    std::vector<RDGeom::Point3D> snap;

    double f = scorer.score_fast_grad(conf, forces);
    grid_pose_gradient(conf, forces, m_dihedral_indices, m_torsion_moved, g);

    std::deque<std::vector<double>> S, Y;
    std::deque<double> RHO;
    const int history = 6;

    for (int it = 0; it < max_iters; ++it) {
        double gmax = 0.0; for (double v : g) gmax = std::max(gmax, std::fabs(v));
        if (gmax < 1e-6) break;

        // L-BFGS two-loop: dir = -H g
        for (int i = 0; i < D; ++i) dir[i] = -g[i];
        const int m = static_cast<int>(S.size());
        std::vector<double> alpha(m, 0.0);
        for (int k = m - 1; k >= 0; --k) {
            double sd = 0.0; for (int i = 0; i < D; ++i) sd += S[k][i] * dir[i];
            alpha[k] = RHO[k] * sd;
            for (int i = 0; i < D; ++i) dir[i] -= alpha[k] * Y[k][i];
        }
        if (m > 0) {
            double sy = 0.0, yy = 0.0;
            for (int i = 0; i < D; ++i) { sy += S[m-1][i]*Y[m-1][i]; yy += Y[m-1][i]*Y[m-1][i]; }
            const double gamma = (yy > 0.0) ? (sy / yy) : 1.0;
            for (int i = 0; i < D; ++i) dir[i] *= gamma;
        }
        for (int k = 0; k < m; ++k) {
            double yd = 0.0; for (int i = 0; i < D; ++i) yd += Y[k][i] * dir[i];
            const double beta = RHO[k] * yd;
            for (int i = 0; i < D; ++i) dir[i] += (alpha[k] - beta) * S[k][i];
        }
        double gTd = 0.0; for (int i = 0; i < D; ++i) gTd += g[i] * dir[i];
        if (!(gTd < 0.0)) {                                   // fall back to steepest descent
            for (int i = 0; i < D; ++i) dir[i] = -g[i];
            S.clear(); Y.clear(); RHO.clear();
            gTd = 0.0; for (int i = 0; i < D; ++i) gTd += g[i] * dir[i];
        }

        // cap the first trial displacement so a large gradient doesn't fling the pose out of the box
        double dinf = 0.0; for (int i = 0; i < D; ++i) dinf = std::max(dinf, std::fabs(dir[i]));
        double t = std::min(1.0, 0.5 / std::max(1e-9, dinf));

        snapshot_coords(conf, snap);
        double f_new = f; bool ok = false;
        for (int ls = 0; ls < 20; ++ls) {
            for (int i = 0; i < D; ++i) delta[i] = t * dir[i];
            restore_coords(conf, snap);
            grid_retract(conf, m_dihedral_indices, m_torsion_moved, delta);
            f_new = scorer.score_fast_grad(conf, forces);     // forces at trial (kept if accepted)
            if (f_new <= f + 1e-4 * t * gTd) { ok = true; break; }
            t *= 0.5;
        }
        if (!ok) { restore_coords(conf, snap); break; }

        grid_pose_gradient(conf, forces, m_dihedral_indices, m_torsion_moved, gnew);
        std::vector<double> s(D), y(D); double ys = 0.0;
        for (int i = 0; i < D; ++i) { s[i] = delta[i]; y[i] = gnew[i] - g[i]; ys += y[i]*s[i]; }
        if (ys > 1e-12) {
            S.push_back(std::move(s)); Y.push_back(std::move(y)); RHO.push_back(1.0/ys);
            if (static_cast<int>(S.size()) > history) { S.pop_front(); Y.pop_front(); RHO.pop_front(); }
        }
        g = gnew; f = f_new;
    }
    return f;
}

// Grid-based refine used by the R2 gate: apply the discrete solution to a fresh pose, minimise it on
// the fast grid score (analytic L-BFGS, cheap), then score with the FULL ECHO score for ranking.
// Torsion-only L-BFGS on the fast grid score.
//
// Structurally identical to grid_minimise, with the 6 rigid-body DOFs pinned: the gradient, the
// quasi-Newton direction and the applied increment are all zeroed on [0..5], so the search runs
// purely in torsion space. grid_pose_gradient already supplies the exact analytic torsion
// derivative (axis_t . SUM_{i in moved[t]} (p_i - k_t) x f_i), so this costs no extra scorer
// evaluations per iteration beyond the ones grid_minimise would do anyway.
//
// Why a separate stage: the coupled step trades torsion error against rigid-body error, so it can
// settle in a rotamer basin it cannot leave -- which matches what we measure on GAD, where the
// anchor half docks to 0.55-0.81 A every run while the distal tail stays 3-5 A out. --dock still
// runs a split TR/torsion ladder; dock2 dropped it for the coupled L-BFGS (see :1298-1303).
double AntColonyOptimizer::grid_minimise_torsions(RDKit::Conformer& conf, const ECHOScore& scorer,
                                                  int max_iters) const {
    const int T = static_cast<int>(m_dihedral_indices.size());
    if (T <= 0 || max_iters <= 0) return scorer.score_fast(conf);
    const int D = 6 + T;
    const int T0 = 6;                       // first torsion index; [0..5] stay pinned

    std::vector<Eigen::RowVector3d> forces;
    std::vector<double> g(D), gnew(D), dir(D, 0.0), delta(D, 0.0);
    std::vector<RDGeom::Point3D> snap;

    auto pin_rigid = [&](std::vector<double>& v) { for (int i = 0; i < T0; ++i) v[i] = 0.0; };

    double f = scorer.score_fast_grad(conf, forces);
    grid_pose_gradient(conf, forces, m_dihedral_indices, m_torsion_moved, g);
    pin_rigid(g);

    std::deque<std::vector<double>> S, Y;
    std::deque<double> RHO;
    const int history = 6;

    for (int it = 0; it < max_iters; ++it) {
        double gmax = 0.0; for (int i = T0; i < D; ++i) gmax = std::max(gmax, std::fabs(g[i]));
        if (gmax < 1e-6) break;

        for (int i = 0; i < D; ++i) dir[i] = -g[i];
        const int m = static_cast<int>(S.size());
        std::vector<double> alpha(m, 0.0);
        for (int k = m - 1; k >= 0; --k) {
            double sd = 0.0; for (int i = T0; i < D; ++i) sd += S[k][i] * dir[i];
            alpha[k] = RHO[k] * sd;
            for (int i = T0; i < D; ++i) dir[i] -= alpha[k] * Y[k][i];
        }
        if (m > 0) {
            double sy = 0.0, yy = 0.0;
            for (int i = T0; i < D; ++i) { sy += S[m-1][i]*Y[m-1][i]; yy += Y[m-1][i]*Y[m-1][i]; }
            const double gamma = (yy > 0.0) ? (sy / yy) : 1.0;
            for (int i = T0; i < D; ++i) dir[i] *= gamma;
        }
        for (int k = 0; k < m; ++k) {
            double yd = 0.0; for (int i = T0; i < D; ++i) yd += Y[k][i] * dir[i];
            const double beta = RHO[k] * yd;
            for (int i = T0; i < D; ++i) dir[i] += (alpha[k] - beta) * S[k][i];
        }
        pin_rigid(dir);

        double gTd = 0.0; for (int i = T0; i < D; ++i) gTd += g[i] * dir[i];
        if (!(gTd < 0.0)) {
            for (int i = 0; i < D; ++i) dir[i] = -g[i];
            pin_rigid(dir);
            S.clear(); Y.clear(); RHO.clear();
            gTd = 0.0; for (int i = T0; i < D; ++i) gTd += g[i] * dir[i];
            if (!(gTd < 0.0)) break;
        }

        // first-trial cap: 0.5 rad (~28.6 deg) on the largest torsion step
        double dinf = 0.0; for (int i = T0; i < D; ++i) dinf = std::max(dinf, std::fabs(dir[i]));
        double t = std::min(1.0, 0.5 / std::max(1e-9, dinf));

        snapshot_coords(conf, snap);
        double f_new = f; bool ok = false;
        for (int ls = 0; ls < 20; ++ls) {
            for (int i = 0; i < D; ++i) delta[i] = t * dir[i];   // [0..5] are already 0
            restore_coords(conf, snap);
            grid_retract(conf, m_dihedral_indices, m_torsion_moved, delta);
            f_new = scorer.score_fast_grad(conf, forces);
            if (f_new <= f + 1e-4 * t * gTd) { ok = true; break; }
            t *= 0.5;
        }
        if (!ok) { restore_coords(conf, snap); break; }

        grid_pose_gradient(conf, forces, m_dihedral_indices, m_torsion_moved, gnew);
        pin_rigid(gnew);
        std::vector<double> sv(D, 0.0), yv(D, 0.0); double ys = 0.0;
        for (int i = T0; i < D; ++i) { sv[i] = delta[i]; yv[i] = gnew[i] - g[i]; ys += yv[i]*sv[i]; }
        if (ys > 1e-12) {
            S.push_back(std::move(sv)); Y.push_back(std::move(yv)); RHO.push_back(1.0/ys);
            if (static_cast<int>(S.size()) > history) { S.pop_front(); Y.pop_front(); RHO.pop_front(); }
        }
        g = gnew; f = f_new;
    }
    return f;
}

AntColonyOptimizer::SplitNmResult
AntColonyOptimizer::refinePoseGrid(const std::vector<double>& discSol,
                                   const ECHOScore& scorer,
                                   int grid_steps) const {
    RDKit::ROMol workMol(m_original_mol);
    RDKit::Conformer& conf = workMol.getConformer();
    apply_ant_solution(conf, discSol);                       // discrete baseline pose
    grid_minimise(conf, scorer, grid_steps);                 // analytic grid L-BFGS (in place)
    if (pre.config().torsion_refine_steps > 0)               // optional torsion-only follow-up
        grid_minimise_torsions(conf, scorer, pre.config().torsion_refine_steps);
    SplitNmResult out;
    out.score   = scorer.score(conf, pre.config().repCap_inner_nm, pre.config().inner_map_score);
    out.mol     = std::move(workMol);
    out.discSol = discSol;
    return out;
}

//initiliser
AntColonyOptimizer::AntColonyOptimizer(const PreComputedData &precomputed_data,
                                            const RDKit::ROMol &original_mol,
                                            ECHOWeights weights):
    pre(precomputed_data),
    m_original_mol(original_mol),
    m_scorer_base(precomputed_data, weights),
    // Was a hard-coded 1234567ULL, which made every run on every machine replay one
    // identical ACO trajectory. Now driven by --dock-seed. Reading through the ctor
    // PARAMETER (not the `pre` member) keeps this independent of member-init order.
    m_baseseed(precomputed_data.config().dock_seed),
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

    // Precompute each torsion's moved-atom set (l-side of the j-k bond): BFS from k, blocking j.
    // Matches RDKit _toBeMovedIndices(mol, jAtomId, kAtomId) so fast_set_dihedral_deg == setDihedralDeg.
    m_torsion_moved.reserve(m_dihedral_indices.size());
    {
        const unsigned nAtoms = m_original_mol.getNumAtoms();
        for (const auto& q : m_dihedral_indices) {
            const unsigned jj = q[1], kk = q[2];
            std::vector<char> vis(nAtoms, 0);
            vis[jj] = 1;                                  // don't cross back to j
            std::vector<unsigned> moved, stack;
            stack.push_back(kk); vis[kk] = 1;
            while (!stack.empty()) {
                const unsigned a = stack.back(); stack.pop_back();
                moved.push_back(a);
                const RDKit::Atom* at = m_original_mol.getAtomWithIdx(a);
                auto nbrs = m_original_mol.getAtomNeighbors(at);
                for (auto it = nbrs.first; it != nbrs.second; ++it) {
                    const unsigned nb = static_cast<unsigned>(*it);
                    if (!vis[nb]) { vis[nb] = 1; stack.push_back(nb); }
                }
            }
            m_torsion_moved.push_back(std::move(moved));
        }
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
        fast_set_dihedral_deg(conf, inds[0], inds[1], inds[2], inds[3],
                              sol[4 + k], m_torsion_moved[k]);
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
    const std::vector<double>& base_tors_deg, // from discSol, size = nTors
    const std::vector<std::vector<unsigned int>>& torsion_moved  // precomputed l-side atom sets
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
        fast_set_dihedral_deg(conf, inds[0], inds[1], inds[2], inds[3], angle, torsion_moved[t]);
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
        applyNormalizedDeltasOnBaseline(pre, conf, x_norm, base_tors_deg, m_torsion_moved);
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
    applyNormalizedDeltasOnBaseline(pre, conf, x_final, base_tors_deg, m_torsion_moved);

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
        applyNormalizedDeltasOnBaseline(pre, conf, x_norm, base_tors_deg, m_torsion_moved);
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
    applyNormalizedDeltasOnBaseline(pre, conf, x_final, base_tors_deg, m_torsion_moved);

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
        applyNormalizedDeltasOnBaseline(pre, conf, x_norm, base_tors_deg, m_torsion_moved);
        return scorer.score(conf, rep_max, map_score_function);
    };

    std::vector<double> x_best_full(D, 0.5);
    double bestScore = eval_full(x_best_full);

    // ---- R1 (docking_v2): numerical-gradient bounded L-BFGS on the full 6+nTors delta
    // vector, replacing the baseline's split TR-only + torsion-only Nelder-Mead ladders.
    // L-BFGS optimizes the coupled rigid-body+torsion space directly (no block split) and
    // reaches a comparable/tighter minimum in far fewer scorer evaluations, which is what
    // makes cluster-then-refine (R2) affordable. x = 0.5 == the discrete baseline pose
    // (zero delta); minimize_box evaluates it first and only ever improves on it.
    {
        std::vector<double> x_final = x_best_full;
        lbfgs_refine::LbfgsParams lp;
        lp.max_iters = manyTors ? 80 : 60;
        lp.history   = 8;
        lp.fd_h      = 1e-2;   // ~0.04 A / ~3.6 deg finite-difference step in physical units
        auto res = lbfgs_refine::minimize_box(eval_full, x_final, lp);

        std::vector<double> disc_refined = convertRealSpaceToDiscrete(discSol, res.x);
        restore_coords(conf, baseline_coords);
        applyNormalizedDeltasOnBaseline(pre, conf, res.x, base_tors_deg, m_torsion_moved);

        SplitNmResult out;
        out.score   = res.f;
        out.mol     = std::move(workMol);
        out.discSol = std::move(disc_refined);
        return out;
    }

#if 0  // ---- baseline split TR/torsion Nelder-Mead: superseded by the R1 L-BFGS block above (kept for reference / NM-fallback) ----
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
    applyNormalizedDeltasOnBaseline(pre, conf, x_final, base_tors_deg, m_torsion_moved);

    SplitNmResult out;
    out.score   = bestScore;
    out.mol     = std::move(workMol);
    out.discSol = std::move(disc_refined);
    return out;
#endif  // ---- end baseline NM (refinePoseSplitNmFromDiscrete) ----
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
    const std::vector<double> &ini_tors_deg,  // base torsions (deg)
    const std::vector<std::vector<unsigned int>>& torsion_moved  // precomputed l-side atom sets
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
        fast_set_dihedral_deg(conf, inds[0], inds[1], inds[2], inds[3], angle, torsion_moved[t]);
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
    applyNormalizedSolution(pre, conf, result.best_point, ini_trans_xyz, ini_rot_deg, ini_tors_deg, m_torsion_moved);

    return { result.best_value, std::move(workMol) };
}
*/


// Final polish seeded directly from a refined conformer.
//
// The discrete route (runLocalNelderMeadFromSeeds + unpackDiscreteSolution) rebuilds the pose from
// the seed's `discSol`, which lives on the 30 deg torsion grid. refinePoseGrid never updates
// discSol at all, and the FD path writes back only a quantised approximation -- so in both cases
// the R2 refinement is scored, used for ranking and seed selection, and then discarded before the
// polish starts. This variant takes the refined conformer as the zero-delta baseline instead.
std::pair<double, RDKit::ROMol>
AntColonyOptimizer::runLocalPolishFromMol(
    const RDKit::ROMol &seedMol,
    const ECHOScore &scorer,
    double rep_max,
    double map_score_function
) const
{
    RDKit::ROMol workMol(seedMol);
    RDKit::Conformer &conf = workMol.getConformer();

    std::vector<RDGeom::Point3D> baseline_coords;
    snapshot_coords(conf, baseline_coords);

    // applyNormalizedDeltasOnBaseline sets torsions ABSOLUTELY to base + delta, so the baseline
    // angles must be the seed's own dihedrals for x = 0.5 to reproduce seedMol exactly.
    const size_t nTors = m_dihedral_indices.size();
    std::vector<double> base_tors_deg(nTors, 0.0);
    for (size_t t = 0; t < nTors; ++t) {
        const auto &q = m_dihedral_indices[t];
        base_tors_deg[t] = MolTransforms::getDihedralDeg(
            conf, static_cast<int>(q[0]), static_cast<int>(q[1]),
                  static_cast<int>(q[2]), static_cast<int>(q[3]));
    }

    const size_t D = 6 + nTors;

    auto eval = [&](const std::vector<double> &x_in) -> double {
        std::vector<double> x = x_in;
        for (double &v : x) v = std::clamp(v, 0.0, 1.0);
        restore_coords(conf, baseline_coords);
        applyNormalizedDeltasOnBaseline(pre, conf, x, base_tors_deg, m_torsion_moved);
        return scorer.score(conf, rep_max, map_score_function);
    };

    std::vector<double> x0(D, 0.5);
    lbfgs_refine::LbfgsParams lp;
    lp.max_iters = (nTors > 10) ? 100 : 70;   // same budget as the discrete polish
    lp.history   = 8;
    lp.fd_h      = 1e-2;
    auto res = lbfgs_refine::minimize_box(eval, x0, lp);

    restore_coords(conf, baseline_coords);
    applyNormalizedDeltasOnBaseline(pre, conf, res.x, base_tors_deg, m_torsion_moved);
    return { res.f, std::move(workMol) };
}


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
        applyNormalizedSolution(pre, conf, x_norm, ini_trans_xyz, ini_rot_deg, ini_tors_deg, m_torsion_moved);
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

    // ---- R1 (docking_v2): numerical-gradient bounded L-BFGS final polish, replacing the
    // staged Nelder-Mead restart schedule below. Same `obj` closure and normalized box.
    {
        std::vector<double> x_lb = x_best;
        lbfgs_refine::LbfgsParams lp;
        lp.max_iters = (nTors > 10) ? 100 : 70;   // final polish: a little deeper than inner refine
        lp.history   = 8;
        lp.fd_h      = 1e-2;
        auto res = lbfgs_refine::minimize_box(obj, x_lb, lp);
        restore_coords(conf, baseCoords);
        applyNormalizedSolution(pre, conf, res.x, ini_trans_xyz, ini_rot_deg, ini_tors_deg, m_torsion_moved);
        return { res.f, std::move(workMol) };
    }

#if 0  // ---- baseline staged Nelder-Mead polish: superseded by the R1 L-BFGS block above (kept for reference) ----
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
    applyNormalizedSolution(pre, conf, x_best, ini_trans_xyz, ini_rot_deg, ini_tors_deg, m_torsion_moved);

    return { f_best, std::move(workMol) };
#endif  // ---- end baseline NM (runLocalNelderMeadFromSeeds) ----
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
    // TEMP recall diagnostic: best heavy-atom RMSD-to-native at each stage (sampled/refined/returned)
    const bool HASREF = pre.has_reference();
    double diag_samp = std::numeric_limits<double>::infinity();
    double diag_ref  = std::numeric_limits<double>::infinity();
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

        // Base (original) ligand coords, snapshotted once. Each ant is scored on a per-thread
        // conformer that is reset to this base then transformed by apply_ant_solution — so we
        // avoid a full RDKit::ROMol deep-copy per ant (the dominant ant-loop cost). apply_ant_solution
        // transforms from base coords, so this is byte-identical to copying m_original_mol per ant.
        std::vector<RDGeom::Point3D> base_coords;
        snapshot_coords(m_original_mol.getConformer(), base_coords);

        #pragma omp parallel
        {
            // Each ant is seeded from (baseseed, iteration, ant index) rather than
            // from the thread id. Two bugs this fixes:
            //   * per-THREAD seeding made results depend on --ncpu (measured: 4 vs 8
            //     threads moved pose scores by up to 0.5 units), because the thread
            //     count decides which ant draws which stream;
            //   * the generator used to be `static thread_local`, so it was seeded
            //     once per thread per PROCESS and the stream carried across ligands
            //     -- the same molecule docked twice in one run scored -6.217939 then
            //     -6.221098. Ligand order changed every later ligand's result.
            // `iter` must be in the mix or every iteration replays the same ants.
            // The generator is declared once per thread and re-seeded per ant to
            // avoid rebuilding mt19937's 2.5 kB state each time.
            std::mt19937 rng;
            ECHOScore scorer = m_scorer_base;
            RDKit::ROMol workMol(m_original_mol);          // ONE copy per thread (was per ant)
            RDKit::Conformer& conf = workMol.getConformer();
            #pragma omp for schedule(static) reduction(min:diag_samp)
            for (int a = 0; a < static_cast<int>(config.n_global_search); ++a) {
                 rng.seed(static_cast<uint32_t>(
                     splitmix64(m_baseseed ^ (uint64_t(iter) << 32) ^ uint64_t(a))));
                 auto& ant = m_ants_vec[a];
                 construct_solution(rng, ant.sol, alpha_now);

                 restore_coords(conf, base_coords);        // cheap O(n_atoms) reset to base
                 apply_ant_solution(conf, ant.sol);
                 // fast_sample: score the ant-sampling stage by trilinear vdW-grid lookup (cheap),
                 // so n_global_search can be cranked; refine below still uses the full score().
                 ant.score = config.fast_sample
                     ? scorer.score_fast(conf, repCap_discrete)
                     : scorer.score(conf, repCap_discrete, inner_map_score);
                 if (HASREF) diag_samp = std::min(diag_samp, rmsd_to_ref(conf, pre.reference_heavy()));
             }
        }//end omp para
        
        
        // ---- R2 (docking_v2): cluster-then-refine gate ----
        // The baseline refines the top-K ants by raw discrete score, which are often
        // near-identical poses in one basin; a near-native pose sitting in a different
        // sub-basin (and scoring slightly worse at the coarse 30-deg grid) never earns a
        // refinement. Instead we shortlist the top-M by score, greedily pick up to K
        // RMSD-diverse representatives (best-scoring member of each basin), and refine
        // those — so distinct basins each get an L-BFGS polish for the same K-refinement
        // budget. This is the cheap, targeted substitute for cranking n_local_search up.
        const unsigned K = std::min<unsigned>(config.n_local_search, m_ants_vec.size());
        if (K==0) {
            continue;
        }
        const unsigned M = std::min<unsigned>(
            std::max<unsigned>(K * 8u, 64u), (unsigned)m_ants_vec.size());
        std::partial_sort(
          m_ants_vec.begin(), m_ants_vec.begin() + M, m_ants_vec.end(),
          [](const auto &A,const auto &B){ return A.score < B.score; }
        );

        // Poses for the shortlist (index i in [0,M) == m_ants_vec[i], score-sorted).
        std::vector<RDKit::ROMol> shortMols;
        shortMols.reserve(M);
        for (unsigned i = 0; i < M; ++i) {
            RDKit::ROMol mol(m_original_mol);
            apply_ant_solution(mol.getConformer(), m_ants_vec[i].sol);
            shortMols.push_back(std::move(mol));
        }

        // Greedy heavy-atom-RMSD clustering: the first (best-scoring) ant of each basin
        // becomes a representative; if fewer than K distinct basins exist, top up with the
        // next-best-by-score so we never refine fewer poses than the baseline would.
        // recall: detect refine basins at the FINER pose_min_rmsd so a distinct near-native basin
        // earns its own rep (and gets refined) instead of being merged into a better-scoring decoy.
        const double clusterRms = (config.pose_min_rmsd > 0.0) ? config.pose_min_rmsd : config.rms_cutoff;
        std::vector<unsigned> reps;
        reps.reserve(K);
        for (unsigned i = 0; i < M && reps.size() < K; ++i) {
            bool distinct = true;
            for (unsigned r : reps) {
                if (GeometryUtils::heavy_atom_rmsd(shortMols[i], shortMols[r]) < clusterRms) {
                    distinct = false; break;
                }
            }
            if (distinct) reps.push_back(i);
        }
        for (unsigned i = 0; i < M && reps.size() < K; ++i) {
            if (std::find(reps.begin(), reps.end(), i) == reps.end()) reps.push_back(i);
        }

        const unsigned R = (unsigned)reps.size();
        const int nThreads = std::min<int>(R, config.n_cpu);
        std::vector<Refined> refined(R);

        #pragma omp parallel num_threads(nThreads)
        {
            ECHOScore scorer = m_scorer_base;

            #pragma omp for schedule(dynamic,1) reduction(min:diag_ref)
            for (int k = 0; k < static_cast<int>(R); ++k) {
                const auto& champ = m_ants_vec[reps[k]];
                // fast_sample: cheap analytic grid L-BFGS refine (reuses the vdW grids);
                // otherwise the baseline finite-difference full-score refine.
                SplitNmResult res = config.fast_sample
                    ? refinePoseGrid(champ.sol, scorer, config.grid_min_steps)
                    : refinePoseSplitNmFromDiscrete(champ.sol, scorer, config.repCap_inner_nm, inner_map_score);

                refined[k].realScore = res.score;
                refined[k].realMol   = std::move(res.mol);
                refined[k].discSol   = std::move(res.discSol);
                refined[k].realNorm.clear();
                if (HASREF) diag_ref = std::min(diag_ref,
                    rmsd_to_ref(refined[k].realMol.getConformer(), pre.reference_heavy()));
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
          // recall (fast_sample): keep EVERY refined basin this iteration as a seed, not just the
          // top m_diversity_now — removes the "discard the 4th-best pose" bottleneck.
          const unsigned keepCount = config.fast_sample
              ? static_cast<unsigned>(rorder.size())
              : std::min<unsigned>(m_diversity_now, (unsigned)rorder.size());
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
    // recall: polish more seeds (the grid refine made seeds cheap and plentiful)
    const unsigned extraSeeds = config.fast_sample ? config.topN * 5u : config.topN * 2u;
    const unsigned seedCount =
        std::min<unsigned>(static_cast<unsigned>(bestPerIter.size()), config.topN + extraSeeds);
    
    // Collect all candidates produced by final polish (NM only)
    // Sized up front and written by loop index: merging thread-local vectors under
    // an omp critical made candidateList's order depend on which thread arrived
    // first, and the sort below is not stable, so exactly-tied scores could permute
    // and the greedy RMSD dedup would then keep different representatives. That
    // would break --dock-seed's promise of --ncpu independence on ties.
    std::vector<IterBest> candidateList(seedCount);
    
    const int nThreads = std::max(1, config.n_cpu);
    #pragma omp parallel num_threads(nThreads)
    {
        ECHOScore scorer = m_scorer_base;
        #pragma omp for schedule(dynamic,1)
        for (int i = 0; i < static_cast<int>(seedCount); ++i) {
            const auto &seed = bestPerIter[static_cast<std::size_t>(i)];

            double nmScore;
            RDKit::ROMol nmMol;
            if (config.polish_from_refined && seed.mol.getNumConformers() > 0) {
                // Start from the R2-refined geometry. The discrete branch below would re-derive
                // the pose from the 30 deg torsion grid, discarding that refinement.
                std::tie(nmScore, nmMol) = runLocalPolishFromMol(
                    seed.mol, scorer, config.repCap_final_nm, outer_map_score);
            } else {
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
                std::tie(nmScore, nmMol) = runLocalNelderMeadFromSeeds(
                    ini_trans_xyz, ini_rot_deg, ini_tors_deg,
                    scorer,
                    config.repCap_final_nm,
                    outer_map_score
                );
            }
            
            IterBest cand;
            cand.score   = nmScore;
            cand.discSol = seed.discSol;     // keep original discrete seed (or replace if you also map back)
            cand.mol     = std::move(nmMol);

            candidateList[static_cast<std::size_t>(i)] = std::move(cand);
        }
    }//end para
    // Sort candidates by score ascending (lower is better). stable_sort so that
    // exactly-tied scores keep seed order and the dedup below is reproducible.
    std::stable_sort(candidateList.begin(), candidateList.end(),
                     [](const IterBest &A, const IterBest &B){ return A.score < B.score; });

    // Recall-oriented return: keep the best-scoring pose per (finer) RMSD basin, up to topN modes,
    // within an energy window of the best (Vina-style num_modes + energy_range). A near-native that
    // scores near the top and sits in its own basin now survives instead of being merged/clipped.
    const double retRms = (config.pose_min_rmsd > 0.0) ? config.pose_min_rmsd : config.rms_cutoff;
    const double bestScore = candidateList.empty() ? 0.0 : candidateList.front().score;
    for (const auto &cand : candidateList) {
        if (selected.size() >= static_cast<std::size_t>(config.topN)) break;
        if (config.energy_range > 0.0 && cand.score > bestScore + config.energy_range) break;

        bool tooClose = false;
        for (const auto &sel : selected) {
            const double rmsd = GeometryUtils::heavy_atom_rmsd(cand.mol, sel.mol);
            if (rmsd < retRms) { tooClose = true; break; }
        }
        if (!tooClose) selected.push_back(cand);
    }

    // TEMP recall diagnostic: where is the near-native lost? sampled vs refined vs returned RMSD.
    if (HASREF) {
        double diag_ret = std::numeric_limits<double>::infinity();
        for (const auto &cand : selected)
            diag_ret = std::min(diag_ret, rmsd_to_ref(cand.mol.getConformer(), pre.reference_heavy()));
        std::cerr << "[recall-diag] best RMSD-to-native  sampled=" << diag_samp
                  << "  refined=" << diag_ref
                  << "  returned=" << diag_ret
                  << "  n_returned=" << selected.size() << "\n";
    }

    //need conformer to coords
    for (const auto &cand : selected) {
      auto coords = conformerToCoords(cand.mol.getConformer());
      out.append(py::make_tuple(cand.score, coords));
    }

    return out;
}
     

