#include "ScoringFunctions.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <mutex>
#include <thread>
#include <iostream>
#include <iomanip>

#include "GeometryUtils.h"

namespace {

struct Terms {
    double aromatic = 0.0;
    double nonbond = 0.0;
    double saltbridge_raw = 0.0;
    double hbond_raw = 0.0;
    double ligand_intra = 0.0;
    double ligand_torsion = 0.0;
    double electro_attractive = 0.0;
    double electro_repulsive_clamp = 0.0;
    double desolvation_penalty_scaled = 0.0;
    double hphobe_raw_hpho = 0.0;
    double hphobe_raw_hpil = 0.0;
    double hphob_enc_gt_7_only_hpho = 0.0;
    double hphob_enc_gt_7_only_hpil_unsat = 0.0;
    double unsat_polar = 0.0;
    double bias = 0.0;
    double constraint = 0.0;
};

inline double apply_weights(const Terms& t, const ECHOWeights& w) {
    return
        w.hphobe_raw_hpil * t.hphobe_raw_hpil +
        w.hphobe_raw_hpho * t.hphobe_raw_hpho +
        w.hphob_enc_gt_7_only_hpil_unsat * t.hphob_enc_gt_7_only_hpil_unsat +
        w.hphob_enc_gt_7_only_hpho * t.hphob_enc_gt_7_only_hpho +
        w.desolvation_penalty_scaled * t.desolvation_penalty_scaled +
        w.electro_repulsive_clamp * t.electro_repulsive_clamp +
        w.electro_attractive * t.electro_attractive +
        w.saltbridge_raw * t.saltbridge_raw +
        w.unsat_polar * t.unsat_polar +
        w.ligand_torsion * t.ligand_torsion +
        w.ligand_intra * t.ligand_intra +
        w.nonbond * t.nonbond +
        w.hbond_raw * t.hbond_raw +
        w.aromatic * t.aromatic;
}


struct Scratch {
    MatX3d ligXYZ;                    
    MatX3d ring_coords;                 

    std::vector<uint8_t> skip_flat;     
    std::vector<char> is_hbond_atom; 
    std::vector<int> neigh;        
    std::vector<Eigen::RowVector3d> ligH; 
};

static thread_local Scratch scratch;

inline void fill_lig_xyz(const RDKit::Conformer &conf, MatX3d &ligXYZ) {
    const int N_all = conf.getNumAtoms();
    ligXYZ.resize(N_all, 3);
    for (int i = 0; i < N_all; ++i) {
        const auto p = conf.getAtomPos(i);
        ligXYZ(i, 0) = p.x;
        ligXYZ(i, 1) = p.y;
        ligXYZ(i, 2) = p.z;
    }
}

inline Eigen::RowVector3d ligand_centroid_heavy(const MatX3d &ligXYZ, int n_heavy) {
    Eigen::RowVector3d c(0.0, 0.0, 0.0);
    for (int i = 0; i < n_heavy; ++i) {
        c[0] += ligXYZ(i, 0);
        c[1] += ligXYZ(i, 1);
        c[2] += ligXYZ(i, 2);
    }
    c /= static_cast<double>(n_heavy);
    return c;
}

inline void compute_bias(Terms& t,
                         const Eigen::RowVector3d &lig_centroid,
                         const AlgorithmConfig &cfg) {
    t.bias = 0.0;
    if (cfg.bias_radius <= 0.0) return;

    const double dist_sq = (lig_centroid - cfg.binding_site_centroid).squaredNorm();
    const double r_sq = cfg.bias_radius * cfg.bias_radius;

    if (dist_sq > r_sq) {
        const double dist = std::sqrt(dist_sq);
        const double delta = dist - cfg.bias_radius; // > 0
        t.bias = 0.5 * delta * delta;
    }
}

inline void reset_skip_and_flags(Scratch& s, int n_heavy, int n_prot_atoms) {
    
    const std::size_t skip_sz = static_cast<std::size_t>(n_heavy) * static_cast<std::size_t>(n_prot_atoms);
    s.skip_flat.resize(skip_sz);
    std::fill(s.skip_flat.begin(), s.skip_flat.end(), uint8_t{0});

    s.is_hbond_atom.resize(static_cast<std::size_t>(n_heavy));
    std::fill(s.is_hbond_atom.begin(), s.is_hbond_atom.end(), char{0});

    s.neigh.clear();
    s.neigh.reserve(64);
}

inline void compute_aromatic(Terms &t,
                            Scratch &s,
                            const MatX3d &ligXYZ,
                            int n_heavy,
                            int n_prot_atoms,
                            double rep_max,
                            const LigandData &lig,        
                            const ProteinData &prot,      
                            const AromaticScorer &arom_scorer ){
                            
    (void)n_heavy; //TODO! use or lose
    (void)n_prot_atoms;

    auto SKIP = [&](int ai, int pi)->uint8_t& {
        return s.skip_flat[static_cast<std::size_t>(ai) * static_cast<std::size_t>(prot.positions.rows())
                         + static_cast<std::size_t>(pi)];
    };

    for (std::size_t lr = 0; lr < lig.ring_idx.size(); ++lr) {
        const int lig_type = lig.ring_types[lr];
        const auto& lig_idx = lig.ring_idx[lr];

        s.ring_coords.resize(static_cast<int>(lig_idx.size()), 3);
        for (int k = 0; k < static_cast<int>(lig_idx.size()); ++k) {
            const int atom_i = lig_idx[static_cast<std::size_t>(k)];
            s.ring_coords(k, 0) = ligXYZ(atom_i, 0);
            s.ring_coords(k, 1) = ligXYZ(atom_i, 1);
            s.ring_coords(k, 2) = ligXYZ(atom_i, 2);
        }

        for (std::size_t pr = 0; pr < prot.ring_idx.size(); ++pr) {
            const int prot_type = prot.ring_types[pr];
            const auto& prot_coords = prot.ring_coords[pr];
            const auto& prot_idx = prot.ring_idx[pr];

            auto [sc, clash_mat] = arom_scorer.score_interaction(
                prot_type, lig_type,
                prot_coords, s.ring_coords,
                prot_idx, lig_idx
            );

            if (std::isfinite(sc)) {
                if (sc <= 0.0) {
                    t.aromatic += sc;
                } else {
                    const double x = sc / rep_max;
                    t.aromatic += sc * std::tanh(x);
                }
            }

            if (clash_mat.size() != 0) {
                const int Ni = static_cast<int>(clash_mat.rows());
                const int Nj = static_cast<int>(clash_mat.cols());
                for (int i = 0; i < Ni; ++i) {
                    const int ai = lig_idx[static_cast<std::size_t>(i)];
                    for (int j = 0; j < Nj; ++j) {
                        const int pi = prot_idx[static_cast<std::size_t>(j)];
                        const double clash_vdw = clash_mat(i, j);
                        if (clash_vdw > 0.0) {
                            const double x = clash_vdw / rep_max;
                            t.aromatic += clash_vdw * std::tanh(x);
                        }
                        SKIP(ai, pi) = 1u;
                    }
                }
            }
        }
    }
}

inline double score_max_angle(const Eigen::RowVector3d &D,
                              const std::vector<Eigen::RowVector3d> &Hvec,
                              const Eigen::RowVector3d &Apos) {
    double best = -1.0;
    for (const auto &H : Hvec) {
        const double ang = GeometryUtils::calc_bond_angle(D, H, Apos);
        if (ang > best) best = ang;
    }
    return best;
}

inline void compute_nonbond_hbond_salt(
    Terms &t,
    Scratch &s,
    const MatX3d &ligXYZ,
    int n_heavy,
    int n_prot_atoms,
    double interaction_cutoff,
    double cutoff_sq,
    double rep_max,
    const PreComputedData &pre,
    const LigandData &lig,
    const ProteinData &prot,
    const AlgorithmConfig &cfg,
    const HBondData &hb
) {
    auto SKIP = [&](int ai, int pi)->uint8_t& {
        return s.skip_flat[static_cast<std::size_t>(ai) * static_cast<std::size_t>(n_prot_atoms)
                         + static_cast<std::size_t>(pi)];
    };

    constexpr double A_salt = -3.85504; //TODO! salt bridge terms in development
    constexpr double B_salt =  0.345362;
    constexpr double C_salt = -144.293;
    constexpr double salt_cutoff = 5.0;
    constexpr double salt_rmin   = 2.0;
    const double salt_cutoff_strict = 4.0;
    const double hbond_cutoff_strict = 3.5;

    for (int ai = 0; ai < n_heavy; ++ai) {
        const Eigen::RowVector3d lp(ligXYZ(ai,0), ligXYZ(ai,1), ligXYZ(ai,2));

        s.neigh.clear();
        cfg.prot_hash.query(lp, interaction_cutoff, prot.positions, s.neigh);

        const double qL = lig.formal_charge[ai];

        for (int pi : s.neigh) {
            if (SKIP(ai, pi)) continue;

            const auto pp_row = prot.positions.row(pi);
            const Eigen::RowVector3d pp(pp_row[0], pp_row[1], pp_row[2]);

            const double dx = lp[0] - pp[0];
            const double dy = lp[1] - pp[1];
            const double dz = lp[2] - pp[2];
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 > cutoff_sq) continue;

            double r = std::sqrt(r2);
            
            if (r < 2.0) {
                /*
                const double d = 2.0 - r;
            
                // penalty scaled by rep_max (tune constants)
                const double A = rep_max * 20.0;
                const double B = 4.0;  // exp steepness
            
                const double pen = A * (std::exp(B * d) - 1.0);
            
                t.nonbond += pen;     // or a dedicated t.clash term
                continue;             // SKIP hbond/saltbridge/attractive terms for this pair
                */
                double u = (2.0 - r) / 2.0;
                if (u < 0.0) u = 0.0;
                if (u > 1.0) u = 1.0;
            
                // smoothstep f(u) = u^2(3-2u) : 0->1, smooth ends
                //const double f = u * u * (3.0 - 2.0 * u);
                const double f = (u * u * (3.0 - 2.0 * u)) / 2.0;
                
                // penalty ranges: rep_max .. 2*rep_max
                const double pen = rep_max  * (1.0 + f);
            
                // add as a clash / nonbond penalty
                t.nonbond += pen;      // or t.clash += pen;
            
                // don't allow any attractive interactions for this pair
                continue;
            }
            
            //if (r < 2.0) r = 2.0;

            const double r2c = r*r;
            const double r4  = r2c*r2c;
            const double r6  = r4*r2c;

            double val = hb.A_values(pi, ai) * std::exp(-hb.B_values(pi, ai) * r)
                       - hb.C_values(pi, ai) / r6;

            bool is_hbond = false;

            // H-bond override 
            if (hb.hbond_mask(pi, ai) && r < 6.0) {
                const auto& protH = prot.hydrogen_coords[pi];

                s.ligH.clear();
                s.ligH.reserve(lig.hydrogen_idx[ai].size());
                for (int hid : lig.hydrogen_idx[ai]) {
                    s.ligH.emplace_back(ligXYZ(hid,0), ligXYZ(hid,1), ligXYZ(hid,2));
                }

                const int role_p = prot.atom_roles(pi);
                const int role_l = pre.get_hbond_role(lig.atom_types(ai));

                double max_ang = -1.0;
                int donor_type = -1, acceptor_type = -1;

                if ((role_p==1 || role_p==3) && (role_l==2 || role_l==3)) {
                    const double a = protH.empty() ? -1.0 : score_max_angle(pp, protH, lp);
                    if (a > max_ang) {
                        max_ang       = a;
                        donor_type    = prot.atom_types(pi);
                        acceptor_type = lig.atom_types(ai);
                    }
                }
                if ((role_p==2 || role_p==3) && (role_l==1 || role_l==3)) {
                    const double a = s.ligH.empty() ? -1.0 : score_max_angle(lp, s.ligH, pp);
                    if (a > max_ang) {
                        max_ang       = a;
                        donor_type    = lig.atom_types(ai);
                        acceptor_type = prot.atom_types(pi);
                    }
                }

                if (donor_type!=-1 && acceptor_type!=-1 && max_ang > 110.0) {
                    auto dit = hb.donor_index.find(donor_type);
                    auto ait = hb.acceptor_index.find(acceptor_type);
                    if (dit!=hb.donor_index.end() && ait!=hb.acceptor_index.end()) {
                        const int di  = dit->second;
                        const int ai2 = ait->second;
                        const int row = di * hb.hbond_A + ai2;

                        const double Anew = GeometryUtils::eval_poly(&hb.polyA_flat(row, 0), hb.polyA_deg, max_ang);
                        const double Bnew = GeometryUtils::eval_poly(&hb.polyB_flat(row, 0), hb.polyB_deg, max_ang);
                        const double Cnew = GeometryUtils::eval_poly(&hb.polyC_flat(row, 0), hb.polyC_deg, max_ang);

                        const double rc = (r < 2.0 ? 2.0 : r);
                        const double rc2 = rc*rc;
                        const double rc6 = rc2*rc2*rc2;

                        val = Anew * std::exp(-Bnew * rc) - Cnew / rc6;

                        if (val < 0.0) {
                            is_hbond = true;
                            t.hbond_raw += val;
                            if (r <= hbond_cutoff_strict) {
                                s.is_hbond_atom[static_cast<std::size_t>(ai)] = 1;
                            }
                        }
                    }
                }
            }

            // Salt bridge
            const double qP = prot.formal_charge[pi];
            if (qL != 0.0 && qP != 0.0 && (qL * qP) < 0.0) {
                if (r <= salt_cutoff) {
                    const double rs  = (r < salt_rmin ? salt_rmin : r);
                    const double rs2 = rs*rs;
                    const double rs4 = rs2*rs2;
                    const double rs6 = rs4*rs2;

                    const double V_salt = A_salt * std::exp(-B_salt * rs) - C_salt / rs6;
                    t.saltbridge_raw += V_salt;

                    if (r <= salt_cutoff_strict && V_salt < 0.0) {
                        s.is_hbond_atom[static_cast<std::size_t>(ai)] = 1;
                    }
                }
            }

            // Nonbond term (with soft-capped repulsion)
            double contrib;
            if (val <= 0.0) {
                contrib = val;
            } else {
                const double x = val / rep_max;
                contrib = rep_max * std::tanh(x);
            }

            if (is_hbond) {
                // attractive hbonds already counted in hbond_raw; keep only repulsive in nonbond
                if (contrib > 0.0) t.nonbond += contrib;
            } else {
                t.nonbond += contrib;
            }
        }
    }
}

inline void compute_ligand_intra(Terms& t,
                                const MatX3d &ligXYZ,
                                int n_heavy,
                                double cutoff_sq,
                                const LigandIntraData &ls) {
    for (int i = 0; i < n_heavy; ++i) {
        const double xi = ligXYZ(i,0), yi = ligXYZ(i,1), zi = ligXYZ(i,2);
        for (int j = i + 1; j < n_heavy; ++j) {
            if (ls.intra_bond_distances(i,j) <= 3) continue;
            if (ls.constrained_pairs_to_ignore.count({i, j})) continue;

            const double dx = xi - ligXYZ(j,0);
            const double dy = yi - ligXYZ(j,1);
            const double dz = zi - ligXYZ(j,2);
            const double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 > cutoff_sq) continue;

            double r = std::sqrt(r2);
            if (r < 2.0) r = 2.0;

            const double Aij = ls.intra_A_values(i,j);
            const double Bij = ls.intra_B_values(i,j);
            const double Cij = ls.intra_C_values(i,j);

            const double r2c = r*r;
            const double r4  = r2c*r2c;
            const double r6  = r4*r2c;

            const double e = Aij * std::exp(-Bij * r) - Cij / r6;
            if (e >= 0.0) t.ligand_intra += e;
        }
    }
}

inline void compute_ligand_torsion(Terms& t,
                                  const MatX3d &ligXYZ,
                                  const LigandIntraData& ls) {
    for (int k = 0; k < ls.torsion_end; ++k) {
        const auto& idxs = ls.ligand_torsion_idxs[k];
        const auto& prof = ls.ligand_torsion_scores[k];
        if (prof.empty()) continue;

        const int i0 = idxs[0], i1 = idxs[1], i2 = idxs[2], i3 = idxs[3];

        const Eigen::RowVector3d p1(ligXYZ(i0,0), ligXYZ(i0,1), ligXYZ(i0,2));
        const Eigen::RowVector3d p2(ligXYZ(i1,0), ligXYZ(i1,1), ligXYZ(i1,2));
        const Eigen::RowVector3d p3(ligXYZ(i2,0), ligXYZ(i2,1), ligXYZ(i2,2));
        const Eigen::RowVector3d p4(ligXYZ(i3,0), ligXYZ(i3,1), ligXYZ(i3,2));

        const double ang = GeometryUtils::calc_dihedral(p1, p2, p3, p4);
        t.ligand_torsion += GeometryUtils::find_closest_torsion_score(ang, prof);
    }
}

inline void compute_electro_and_desolv(Terms &t,
                                      const MatX3d &ligXYZ,
                                      double electro_clamp,
                                      const LigandData &lig,
                                      const GridData &eg,
                                      const GridData &sg) {
    const std::size_t Nq = lig.partial_charges.size();
    for (std::size_t i = 0; i < Nq; ++i) {
        const Eigen::RowVector3d lp(ligXYZ(static_cast<int>(i),0),
                                    ligXYZ(static_cast<int>(i),1),
                                    ligXYZ(static_cast<int>(i),2));

        const double env = GeometryUtils::trilinear_sample(
            sg.data, lp, sg.origin, sg.apix, sg.nz, sg.ny, sg.nx
        );
        const double phi = GeometryUtils::trilinear_sample(
            eg.data, lp, eg.origin, eg.apix, eg.nz, eg.ny, eg.nx
        );

        const double q = lig.partial_charges[i];
        const double interaction = q * phi;

        if (interaction < 0.0) {
            t.electro_attractive += interaction;
        } else {
            t.electro_repulsive_clamp += std::min(interaction, electro_clamp);
        }

        t.desolvation_penalty_scaled += (q * q) * env;
    }
}

inline void compute_hydrophobics(Terms& t,
                                const MatX3d& ligXYZ,
                                int n_heavy,
                                const Scratch& s,
                                const LigandData &lig,
                                const GridData &hrag,
                                const GridData &heng) {
    for (int i = 0; i < n_heavy; ++i) {
        const Eigen::RowVector3d lp(ligXYZ(i,0), ligXYZ(i,1), ligXYZ(i,2));

        const double hraw = GeometryUtils::trilinear_sample(
            hrag.data, lp, hrag.origin, hrag.apix, hrag.nz, hrag.ny, hrag.nx
        );
        const double henc = GeometryUtils::trilinear_sample(
            heng.data, lp, heng.origin, heng.apix, heng.nz, heng.ny, heng.nx
        );

        const double logp = lig.per_atom_logp[i];
        if (logp >= 0.0) t.hphobe_raw_hpho += logp * hraw;
        else             t.hphobe_raw_hpil += logp * hraw;

        if (henc >= 0.7) {
            if (logp >= 0.0) {
                t.hphob_enc_gt_7_only_hpho += logp * henc;
            } else {
                if (!s.is_hbond_atom[static_cast<std::size_t>(i)]) {
                    t.hphob_enc_gt_7_only_hpil_unsat += logp * henc;
                }
            }
        }
    }
}

inline void compute_unsat_polar(Terms& t,
                                const MatX3d &ligXYZ,
                                int n_heavy,
                                const Scratch &s,
                                const LigandData &lig,
                                const GridData &sg) {
    for (int i = 0; i < n_heavy; ++i) {
        const bool is_polar = (lig.hbond_atom_mask[i] != 0) || (lig.formal_charge[i] != 0.0);
        if (!is_polar) continue;
        if (s.is_hbond_atom[static_cast<std::size_t>(i)]) continue;

        const Eigen::RowVector3d lp(ligXYZ(i,0), ligXYZ(i,1), ligXYZ(i,2));
        const double env = GeometryUtils::trilinear_sample(
            sg.data, lp, sg.origin, sg.apix, sg.nz, sg.ny, sg.nx
        );
        t.unsat_polar += env;
    }
}

inline void compute_constraint(Terms &t,
                               const MatX3d &ligXYZ,
                               const LigandIntraData& ls) {
    t.constraint = 0.0;
    const double k = 10.0;

    for (std::size_t i = 0; i < ls.constrained_pair_idxs.size(); ++i) {
        
        const auto &[idx1, idx2] = ls.constrained_pair_idxs[i];
        //const auto& pr = ls.constrained_pair_idxs[i];
        
        //const int idx1 = pr.first;
        //const int idx2 = pr.second;
        const double r0 = ls.constrained_atom_distances[i];

        const double dx = ligXYZ(idx1,0) - ligXYZ(idx2,0);
        const double dy = ligXYZ(idx1,1) - ligXYZ(idx2,1);
        const double dz = ligXYZ(idx1,2) - ligXYZ(idx2,2);

        const double r  = std::sqrt(dx*dx + dy*dy + dz*dz);
        const double dr = r - r0;

        t.constraint += k * (dr * dr);
    }
}

} 



namespace {
    static std::mutex g_terms_print_mtx;

    inline void debug_print_terms(const Terms &t,
                                  double total_before_negate,
                                  double total_after_all,
                                  const ECHOWeights &w)
    {
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << std::setprecision(17);

        oss << "---- ECHOScore Terms (thread " << std::this_thread::get_id() << ") ----\n";

        // Raw term values
        oss << "aromatic, " << t.aromatic << "\n";
        oss << "nonbond, " << t.nonbond << "\n";
        oss << "saltbridge_raw, " << t.saltbridge_raw << "\n";
        oss << "hbond_raw, " << t.hbond_raw << "\n";
        oss << "ligand_intra, " << t.ligand_intra << "\n";
        oss << "ligand_torsion, " << t.ligand_torsion << "\n";
        oss << "electro_attractive, " << t.electro_attractive << "\n";
        oss << "electro_repulsive_clamp, " << t.electro_repulsive_clamp << "\n";
        oss << "desolvation_penalty_scaled, " << t.desolvation_penalty_scaled << "\n";
        oss << "hphobe_raw_hpho, " << t.hphobe_raw_hpho << "\n";
        oss << "hphobe_raw_hpil, " << t.hphobe_raw_hpil << "\n";
        oss << "hphob_enc_gt_7_only_hpho, " << t.hphob_enc_gt_7_only_hpho << "\n";
        oss << "hphob_enc_gt_7_only_hpil_unsat, " << t.hphob_enc_gt_7_only_hpil_unsat << "\n";
        oss << "unsat_polar, " << t.unsat_polar << "\n";
        oss << "bias, " << t.bias << "\n";
        oss << "constraint, " << t.constraint << "\n";
        
        oss << "---- Weights (snapshot) ----\n";
        oss << "w.aromatic, " << w.aromatic << "\n";
        oss << "w.nonbond, " << w.nonbond << "\n";
        oss << "w.saltbridge_raw, " << w.saltbridge_raw << "\n";
        oss << "w.hbond_raw, " << w.hbond_raw << "\n";
        oss << "w.ligand_intra, " << w.ligand_intra << "\n";
        oss << "w.ligand_torsion, " << w.ligand_torsion << "\n";
        oss << "w.electro_attractive, " << w.electro_attractive << "\n";
        oss << "w.electro_repulsive_clamp, " << w.electro_repulsive_clamp << "\n";
        oss << "w.desolvation_penalty_scaled, " << w.desolvation_penalty_scaled << "\n";
        oss << "w.hphobe_raw_hpho, " << w.hphobe_raw_hpho << "\n";
        oss << "w.hphobe_raw_hpil, " << w.hphobe_raw_hpil << "\n";
        oss << "w.hphob_enc_gt_7_only_hpho, " << w.hphob_enc_gt_7_only_hpho << "\n";
        oss << "w.hphob_enc_gt_7_only_hpil_unsat, " << w.hphob_enc_gt_7_only_hpil_unsat << "\n";
        oss << "w.unsat_polar, " << w.unsat_polar << "\n";
        oss << "---- Totals ----\n";
        oss << "linear_combo_before_negate, " << total_before_negate << "\n";
        oss << "final_total_after_negate_plus_penalties, " << total_after_all << "\n";
        oss << "----------------------------------------\n";

        
        {
            std::lock_guard<std::mutex> lk(g_terms_print_mtx);
            std::cout << oss.str();
        }
    }
}

double ECHOScore::score(const RDKit::Conformer &conf, double rep_max) const {
    Terms terms;

    // setup
    const auto &lig = pre.ligand();
    const auto &prot = pre.protein();
    const auto &cfg = pre.config();
    const auto &hb = pre.hbond();
    const auto &ls = pre.ligand_score();

    const int n_prot_atoms = prot.positions.rows();
    const int heavy_end = static_cast<int>(lig.heavy_end_idx);
    const int n_heavy = heavy_end + 1;

    const double cutoff_sq = interaction_cutoff * interaction_cutoff;

    
    fill_lig_xyz(conf, scratch.ligXYZ);
    reset_skip_and_flags(scratch, n_heavy, n_prot_atoms);
    
    {
        const Eigen::RowVector3d c = ligand_centroid_heavy(scratch.ligXYZ, n_heavy);
        compute_bias(terms, c, cfg);
    }
    
    //interaction terms
    compute_aromatic(terms, scratch, scratch.ligXYZ, n_heavy, n_prot_atoms,
                     rep_max, lig, prot, pre.aromatic_scorer());
                     
    compute_nonbond_hbond_salt(terms, scratch, scratch.ligXYZ,
                               n_heavy, n_prot_atoms,
                               interaction_cutoff, cutoff_sq, rep_max,
                               pre, lig, prot, cfg, hb);
                               
    compute_ligand_intra(terms, scratch.ligXYZ, n_heavy, cutoff_sq, ls);
    compute_ligand_torsion(terms, scratch.ligXYZ, ls);
    compute_electro_and_desolv(terms, scratch.ligXYZ, electro_clamp,
                               lig, pre.electro_grid(), pre.environment_grid());
                               
    compute_hydrophobics(terms, scratch.ligXYZ, n_heavy, scratch,
                         lig, pre.hydrohpobic_grid_raw(), pre.hydrophobic_enclosure_grid());
                         
    compute_unsat_polar(terms, scratch.ligXYZ, n_heavy, scratch,
                        lig, pre.environment_grid());
                        
    compute_constraint(terms, scratch.ligXYZ, ls);
    //get score
    double total_linear = apply_weights(terms, weights);
    double total = -total_linear + terms.bias + terms.constraint;
    
    //debug_print_terms(terms, total_linear, total, weights);
    return total;
}
