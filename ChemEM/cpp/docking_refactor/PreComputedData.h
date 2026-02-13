#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <set>
#include <cmath>
#include <utility>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "Scorers.h"

namespace py = pybind11;

//use row major its faster to copy from numpy objects.
using MatrixXd = Eigen::Matrix<double , Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXi = Eigen::Matrix<int, Eigen::Dynamic, 1>;
//using RowVector3d = Eigen::matrix<double, 1, 3>;
using MaskMatrix = Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using TorsionProfile = Eigen::Array<double, 1, 361>;
using MatX3d =   Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;


struct ProteinData{
    
    MatrixXd positions; 
    VectorXi atom_types;
    VectorXi atom_roles;
    std::vector<std::vector<Eigen::RowVector3d>> hydrogen_coords; 
    std::vector<int> ring_types;
    std::vector<MatrixXd> ring_coords; //(Å, cartisian), protein rings can have variable numbers of atoms
    //std::vector<MatX3d> ring_coords;
    //std::vector<VectorXi> ring_idx;
    std::vector<std::vector<int>> ring_idx;
    Eigen::VectorXd formal_charge;
    std::vector<int> halogen_acceptors;
    std::vector<int> halogen_acceptor_roots;
    std::unordered_set<int> halogen_acceptors_set;
    std::vector<int> halogen_acceptor_root_by_atom;

};

struct LigandData{
    //TODO!! find this automatically.
    int n_heavy;
    std::size_t heavy_end_idx;
    VectorXi atom_types;
    std::vector<Eigen::VectorXi> hydrogen_idx;
    std::vector<int> ring_types;
    std::vector<std::vector<int>> ring_idx;
    std::vector<double> partial_charges;
    std::vector<int> halogen_donor_idx;
    std::vector<int> halogen_donor_root_idx;
    std::vector<double> per_atom_logp;
    std::vector<int> hbond_atom_mask;
    std::vector<double> formal_charge;
};

struct AromaticData {
    Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> keys;
    // Grouping coefficients by axis/type
    std::vector<int> nxA, nyA, nxB, nyB, nxC, nyC;
    std::vector<std::vector<double>> coefA, coefB, coefC; 
};



struct HBondData {
    
    int hbond_D, hbond_A; 
    int polyA_deg, polyB_deg, polyC_deg;

    MatrixXd polyA_flat, polyB_flat, polyC_flat;
    MatrixXd A_values, B_values, C_values;
    
    std::unordered_map<int, int> donor_index, acceptor_index;
    std::unordered_set<int> donor_types_set, acceptor_types_set;

    VectorXi donor_types, acceptor_types;
    MaskMatrix hbond_mask;
};



struct LigandIntraData {

    MatrixXd intra_A_values;
    MatrixXd intra_B_values;
    MatrixXd intra_C_values;
    MatrixXi intra_bond_distances;
    std::vector<std::tuple<int, int>> constrained_pair_idxs;
    std::vector<double> constrained_atom_distances;
    std::set<std::pair<int, int>> constrained_pairs_to_ignore;
    int n_torsions, torsion_end;
    std::vector<std::array<int,4>> ligand_torsion_idxs;
    std::vector<std::vector<std::pair<int,double>>> ligand_torsion_scores;
};



  
struct ScoringWeights {
    double nonbond, dsasa, hphob, electro, ligand_torsion, ligand_intra;
    double vdw, hbond, aromatic, halogen, hphob_enc, constraint;
    double repCap_discrete, repCap_inner_nm, repCap_final_nm; // must be positive min 0.0
    double rms_cutoff = 1.0;
};

struct DensityData {
    
    int nbins;
    double mi_weight;
    double sci_weight;
    double resolution;
    double sigma_coeff;
    std::vector<double> atom_masses;  
    bool normalise = true; 
};


struct GridData {
    Eigen::RowVector3d origin, apix;
    std::vector<double> data;
    int nx, ny, nz;
};

struct GridDataHeader {
    Eigen::RowVector3d origin, apix;
    int nx, ny, nz;
};

struct WaterData {
    bool enabled = false;
    int atom_type = 28;
    int max_per_ligand = 0;
    double hbond_distance = 2.8;
    double min_protein_clearance = 2.2;
};

//spatial hash map 
struct SpatialHash {
    double cell = 4.5;                 
    Eigen::RowVector3d origin = Eigen::RowVector3d::Zero();
    // packed 3D cell key -> list of protein atom indices
    std::unordered_map<std::uint64_t, std::vector<int>> bins;

    static inline std::uint64_t packKey(int ix, int iy, int iz) noexcept {
        constexpr std::int64_t OFF = (1ll << 20); 
        const std::uint64_t x = static_cast<std::uint64_t>(static_cast<std::int64_t>(ix) + OFF);
        const std::uint64_t y = static_cast<std::uint64_t>(static_cast<std::int64_t>(iy) + OFF);
        const std::uint64_t z = static_cast<std::uint64_t>(static_cast<std::int64_t>(iz) + OFF);
        return (x << 42) | (y << 21) | z;
    }

    inline Eigen::Vector3i cellOf(const Eigen::RowVector3d& p) const {
        Eigen::RowVector3d q = (p - origin) / cell;
        return { int(std::floor(q[0])), int(std::floor(q[1])), int(std::floor(q[2])) };
    }

    void build(const MatrixXd& prot_pos, double cell_size) {
        cell = cell_size;
        origin = prot_pos.colwise().minCoeff(); // keep integer keys small
        bins.clear();
        bins.reserve(size_t(prot_pos.rows() * 1.3));
        for (int i = 0; i < prot_pos.rows(); ++i) {
            Eigen::Vector3i c = cellOf(prot_pos.row(i));
            bins[packKey(c[0], c[1], c[2])].push_back(i);
        }
    }

    // Query neighbors within radius r; returns atom indices in `out`.
    inline void query(const Eigen::RowVector3d& q,
                      double r,
                      const MatrixXd& prot_pos,
                      std::vector<int>& out) const {
        out.clear();
        const double r2 = r * r;
        Eigen::Vector3i c = cellOf(q);
        const int rad = std::max(1, int(std::ceil(r / cell)));
        for (int dx = -rad; dx <= rad; ++dx)
        for (int dy = -rad; dy <= rad; ++dy)
        for (int dz = -rad; dz <= rad; ++dz) {
            auto it = bins.find(packKey(c[0]+dx, c[1]+dy, c[2]+dz));
            if (it == bins.end()) continue;
            const auto& v = it->second;
            for (int pi : v) {
                double d2 = (q - prot_pos.row(pi)).squaredNorm();
                if (d2 <= r2) out.push_back(pi);
            }
        }
    }
};



struct AlgorithmConfig {
    int n_cpu;
    int n_global_search;
    int n_local_search;
    int topN = 20;
    double interaction_cutoff = 6.0;
    double electro_clamp = 6.0;
    unsigned int iterations;
    double repCap0 = 2.0;
    double repCap1 = 5.0;
    double repCap_inner_nm = 10.0;
    double repCap_final_nm = 15.0;
    double rms_cutoff = 2.0;
    double a_lo = 0.25;
    double a_mid = 0.45;
    double a_hi = 0.70;
    unsigned updateN_now = 3;
    Eigen::RowVector3d binding_site_centroid;
    double bias_radius;
    double nb_cell = 4.5;
    //std::vector<uint8_t> skip_mask;
    std::vector<Eigen::RowVector3i> translation_points;
    std::vector<std::vector<double>> all_arrays;
    std::vector<std::vector<size_t>> adjacency;
    SpatialHash prot_hash; // The spatial lookup engine
    bool no_map = false;
    int inner_map_score = 1;
    int outer_map_score = 0;
};


/**
  *@class PreComputedData 
  *@brief This class holds all data related to docking search and scoring. 
  * This class handels the conversion of data from python/numpy objects to C++ structures.
  * Data is immutable after creation for thread safety. 
*/
class PreComputedData{
public:
    explicit PreComputedData(py::object py_pc);
    ///getters
    const ProteinData& protein() const { return m_protein_data ; }
    const LigandData& ligand() const { return m_ligand_data ; }
    const GridDataHeader& binding_site_grid() const { return m_binding_site_grid ; }
    const HBondData&     hbond()     const { return m_hbond_score_data; }
    const AromaticData&  aromatic()  const { return m_aromatic_score_data; }
    const LigandIntraData& ligand_score() const { return m_ligand_intra; }
    const ScoringWeights& weights()   const { return m_weights; }
    const AlgorithmConfig& config()  const { return m_config; }
    const GridData& environment_grid() const { return m_environment_grid ;}
    const GridData& electro_grid() const { return m_electro_grid ;}
    const GridData& electro_grid_raw() const { return m_electro_grid_raw ;}
    const GridData& hydrohpobic_grid_raw() const { return m_hydrophobic_grid_raw ; }
    const GridData& hydrophobic_enclosure_grid() const { return m_hydrophobic_enclosure_grid ; }
    //const GridData& desolvation_polar_grid() const { return m_desolvation_polar_grid ; }
    //const GridData& desolvation_hydrophobic_grid() const { return m_desolvation_hydrohpobic_grid ; }
    //const GridData& delta_sasa_grid() const { return m_delta_sasa_grid ; }
    const AromaticScorer& aromatic_scorer() const { return m_aromatic_scorer; }
    const GridData& density_grid() const { return m_density_grid; }
    const GridData& sci_grid() const {return m_sci_grid; }
    const GridData& sci_first_derivative_grid() const { return m_sci_first_derivitive_grid; } 
    const GridData& sci_second_derivative_grid() const { return m_sci_score_second_derivative_grid; } 
    const DensityData& density_data() const { return m_density_data; }
    //const HalogenScorer&  halogen_scorer()  const { return m_halogen_scorer; }
    [[nodiscard]] inline int get_hbond_role(int atom_type) const noexcept {
        int mask = 0;
        if (m_hbond_score_data.donor_types_set.count(atom_type))   mask |= 1; 
        if (m_hbond_score_data.acceptor_types_set.count(atom_type)) mask |= 2; 
        return mask;
    }
    
   
private: 
    
    ProteinData m_protein_data;
    LigandData m_ligand_data; 
    //grids
    GridDataHeader m_binding_site_grid; 
    GridData m_environment_grid;
    GridData m_electro_grid; 
    GridData m_electro_grid_raw;
    GridData m_hydrophobic_grid_raw;
    GridData m_hydrophobic_enclosure_grid;
    //GridData m_desolvation_polar_grid;
    //GridData m_desolvation_hydrohpobic_grid;
    //GridData m_delta_sasa_grid; 
    HBondData m_hbond_score_data;
    AromaticData m_aromatic_score_data;
    LigandIntraData m_ligand_intra;
    ScoringWeights m_weights;
    WaterData m_water_data;
    AlgorithmConfig m_config;
    AromaticScorer m_aromatic_scorer;
    GridData m_density_grid;
    GridData m_sci_grid;
    GridData m_sci_first_derivitive_grid;
    GridData m_sci_score_second_derivative_grid;
    DensityData m_density_data;
    
    //HalogenScorer  m_halogen_scorer;
    void validate_protein_data();
    void validate_ligand_data();
    void validate_weights();
    
};
