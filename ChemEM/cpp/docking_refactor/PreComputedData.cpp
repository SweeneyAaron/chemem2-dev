
#include <pybind11/pybind11.h> 
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <string>
#include <stdexcept>
#include <iostream>
#include "PreComputedData.h"

namespace py = pybind11;

void PreComputedData::validate_protein_data(){
    const size_t n_protein_atoms = m_protein_data.positions.rows();
    const size_t n_ring_types = m_protein_data.ring_types.size();
    
    if (m_protein_data.atom_types.size() != n_protein_atoms){
        throw std::runtime_error("[Error] N protein atoms (" + std::to_string(n_protein_atoms) + 
                                 " ) != to N atom types (" + std::to_string(m_protein_data.atom_types.size()) + ").");
    }
    
    if (m_protein_data.atom_roles.size() != n_protein_atoms){
        throw std::runtime_error( "[Error] N protein atoms (" + std::to_string(n_protein_atoms) + 
                                    ") != N atom roles (" + std::to_string(m_protein_data.atom_roles.size()) + ").");
    }
    
    if (m_protein_data.formal_charge.size() != n_protein_atoms){
        throw std::runtime_error(" [Error] N protein atoms (" + std::to_string(n_protein_atoms) + 
                                  ") != N formal charge (" + std::to_string(m_protein_data.formal_charge.size()) + ").");
    }
    
    if (m_protein_data.ring_coords.size() != n_ring_types){
        throw std::runtime_error("[Error] N ring types (" + std::to_string(n_ring_types) + 
                                  ") != N ring_coords (" + std::to_string(m_protein_data.ring_coords.size()) + ").");
    }
    
    if (m_protein_data.ring_idx.size() != n_ring_types){
        throw std::runtime_error("[Error] N ring types (" + std::to_string(n_ring_types) + 
                                ") != N ring_idx (" + std::to_string(m_protein_data.ring_idx.size()) + ").");
    }
    
}

void PreComputedData::validate_ligand_data() {
    const size_t n_ligand_atoms = m_ligand_data.atom_types.size(); 
    
    if ( m_ligand_data.partial_charges.size() != n_ligand_atoms) {
        throw std::runtime_error("N ligand atoms (" + std::to_string(n_ligand_atoms) + 
                            ") != N partial charges (" + std::to_string(m_ligand_data.partial_charges.size()) + ")." );
    }
    
    //should be compared to full atoms
    //if (m_ligand_data.formal_charge.size() != n_ligand_atoms) {
    //    throw std::runtime_error("N ligand atoms (" + std::to_string(n_ligand_atoms) + 
    //                             ") != N formal charges (" + std::to_string(m_ligand_data.formal_charge.size()) + ").");
    //}
    
    if (m_ligand_data.ring_types.size() != m_ligand_data.ring_idx.size()){
        throw std::runtime_error("N ligand ring types (" + std::to_string(m_ligand_data.ring_types.size()) +
                                 ") != ligand ring indices (" + std::to_string(m_ligand_data.ring_idx.size()) + ").");
    }

}


void PreComputedData::validate_weights() {
    /*
    RepCap weights clamp the repulsion from QM terms, if a recpcap is < 0.0 
    this will clamp at the attractive parts of the interaction. 
    Since we use a buckingham-like term for scoring it will clamp at two places in the well.
    i.e. -1.0 may clamp atom a - atom b distance at both 4.0 and 5.0 Å.
    
    */
    auto check_weights_and_clamp = [](double &weight, const std::string &weight_id){
        if (weight < 0.0) {
            std::cerr << "[Warning] weight (" << weight_id << ") can't be < 0.0 (" << weight
                        << ") resetting to 0.0." << std::endl; 
            weight = 0.0;
        }
    };
    
    check_weights_and_clamp(m_weights.repCap_discrete , "repCap_discrete");
    check_weights_and_clamp(m_weights.repCap_inner_nm , "repCap_inner_nm");
    check_weights_and_clamp(m_weights.repCap_final_nm , "repCap_final_nm");
    
    }


void load_grid(GridData &grid, py::object &py_pc, const std::string &grid_name){
    
    grid.origin = py_pc.attr((grid_name + "_origin").c_str()).cast<Eigen::RowVector3d>();
    grid.apix = py_pc.attr((grid_name + "_apix").c_str()).cast<Eigen::RowVector3d>();
    
    auto np_ds = py_pc.attr((grid_name + "_grid").c_str()).cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    auto buf = np_ds.unchecked<3>();
    
    grid.nz = static_cast<int>(buf.shape(0));
    grid.ny = static_cast<int>(buf.shape(1));
    grid.nx = static_cast<int>(buf.shape(2));
    
    grid.data.resize(static_cast<size_t>(grid.nx) * grid.ny * grid.nz );
    
    for (int z = 0; z < grid.nz; z++){
        for (int y = 0; y < grid.ny; y++  ) {
            for (int x = 0; x < grid.nx; x++ ){
                grid.data[(z * grid.ny + y) * grid.nx + x] = buf(z,y,x);
            }
        }
    }
}


void load_grid_header(GridDataHeader &grid, py::object &py_pc, const std::string &grid_name){
    /*
    Note that unlike the GridData Loader, for the grid header loader 
    the nx, ny, nz specifiers from the python object must share the same prefix.
    */
    
    grid.origin = py_pc.attr((grid_name + "_origin").c_str()).cast<Eigen::RowVector3d>(); 
    grid.apix = py_pc.attr((grid_name + "_apix").c_str()).cast<Eigen::RowVector3d>();
    grid.nx = py_pc.attr((grid_name + "_nx").c_str()).cast<int>();
    grid.ny = py_pc.attr((grid_name + "_ny").c_str()).cast<int>(); 
    grid.nz = py_pc.attr((grid_name + "_nz").c_str()).cast<int>();
    

}



void load_polyX_values_flat(MatrixXd &poly_grid_flat, int &poly_deg, int hbond_D, int hbond_A, py::object &py_pc, const std::string &grid_name){

    auto arr = py_pc.attr(grid_name.c_str()).cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    int deg = arr.shape(2) - 1;
    poly_deg = deg;
    // flatten (D,A,deg+1) into (D*A, deg+1)
    poly_grid_flat = Eigen::Map<MatrixXd>(arr.mutable_data(), hbond_D*hbond_A, deg+1).eval();
    
}



PreComputedData::PreComputedData(py::object py_pc){
    
    //-----protein data----
    try {
        m_protein_data.positions = py_pc.attr("protein_positions").cast<MatrixXd>();
        m_protein_data.atom_types = py_pc.attr("protein_atom_types").cast<VectorXi>();  
        m_protein_data.atom_roles = py_pc.attr("protein_atom_roles").cast<VectorXi>(); 
        m_protein_data.hydrogen_coords = py_pc.attr("protein_hydrogens").cast<std::vector<std::vector<Eigen::RowVector3d>>>();
        m_protein_data.ring_types = py_pc.attr("protein_ring_type_ints").cast<std::vector<int>>();
        m_protein_data.ring_coords = py_pc.attr("protein_ring_coords").cast<std::vector<MatrixXd>>();
        //m_protein_data.ring_coords = py_pc.attr("protein_ring_coords").cast<std::vector<MatX3d>>();
        //m_protein_data.ring_idx = py_pc.attr("protein_ring_idx").cast<std::vector<VectorXi>>();
        m_protein_data.ring_idx = py_pc.attr("protein_ring_idx").cast<std::vector<std::vector<int>>>();
        
        m_protein_data.formal_charge = py_pc.attr("protein_formal_charge").cast<Eigen::VectorXd>(); 
        m_protein_data.halogen_acceptors = py_pc.attr("halogen_bond_acceptor_indices").cast<std::vector<int>>();
        m_protein_data.halogen_acceptor_roots = py_pc.attr("halogen_bond_acceptor_root_indices").cast<std::vector<int>>();
        /** * Build fast-lookup maps for halogen acceptor geometry.
         * Creates an O(1) direct-mapped vector for roots and a set for quick filtering,
         * allowing the scorer to instantly identify acceptors and their parent atoms.
         */
        {
            const int P = m_protein_data.positions.rows();
            m_protein_data.halogen_acceptors_set.clear();
            m_protein_data.halogen_acceptor_root_by_atom.assign(P, -1);
            for (size_t i = 0; i <  m_protein_data.halogen_acceptors.size(); ++i) {
                const int a =  m_protein_data.halogen_acceptors[i];
                const int r =  m_protein_data.halogen_acceptor_roots[i];
                 m_protein_data.halogen_acceptors_set.insert(a);
                 if (a >= 0 && a < P)  m_protein_data.halogen_acceptor_root_by_atom[a] = r;
            }
        }
        
        validate_protein_data();
    
    } catch (const py::cast_error &e) {
        throw std::runtime_error("[Error] A Python protein object could not be converted to the required C++ type: " + std::string(e.what()));
    } catch (const py::error_already_set &e) {
        throw std::runtime_error("[Error] The protein input object is missing a required property: " + std::string(e.what()));
    } catch (const std::exception &e) {
        throw std::runtime_error("[Error] PreComputedData Protein data fialed to initilise: " + std::string(e.what()));
    }
    
    //-----ligand data-----
    
    try {
    
    
        m_ligand_data.heavy_end_idx = py_pc.attr("ligand_heavy_end_index").cast<std::size_t>();
        m_ligand_data.atom_types = py_pc.attr("ligand_atom_types").cast<VectorXi>(); 
        m_ligand_data.hydrogen_idx = py_pc.attr("ligand_hydrogen_idx").cast<std::vector<Eigen::VectorXi>>();
        m_ligand_data.ring_types = py_pc.attr("ligand_ring_type_ints").cast<std::vector<int>>();
        m_ligand_data.ring_idx = py_pc.attr("ligand_ring_indices").cast<std::vector<std::vector<int>>>();
        m_ligand_data.partial_charges = py_pc.attr("ligand_charges").cast<std::vector<double>>();
        m_ligand_data.halogen_donor_idx = py_pc.attr("halogen_bond_donor_indices").cast<std::vector<int>>(); 
        m_ligand_data.halogen_donor_root_idx = py_pc.attr("halogen_bond_donor_root_indices").cast<std::vector<int>>(); 
        m_ligand_data.per_atom_logp = py_pc.attr("per_atom_logp").cast<std::vector<double>>();
        m_ligand_data.hbond_atom_mask = py_pc.attr("ligand_hbond_atom").cast<std::vector<int>>(); 
        m_ligand_data.formal_charge = py_pc.attr("ligand_formal_charge").cast<std::vector<double>>(); 
        
        validate_ligand_data();
    
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[Error] A Python liand object could not be converted to the required C++ type: " + std::string(e.what()));
    } catch (const py::error_already_set& e) {
        throw std::runtime_error("[Error] The ligand object is missing a required parameter: " + std::string(e.what())); 
    } catch (const std::exception& e) { 
        throw std::runtime_error("[Error] PreComputedData Ligand data failed to initilise: " + std::string(e.what())); 
    }
    //-----binding site data-----
    
    try {
        load_grid_header(m_binding_site_grid, py_pc, "binding_site_grid");
    } catch (const std::exception &e) {
        throw std::runtime_error("[Error] PreComputedData binding_site_grid failed to initilise" + std::string(e.what()));
    }
    
    
    //-----scoring grids data-----
    
    try {
        load_grid(m_environment_grid, py_pc, "env_scaled");
    } catch (const std::exception &e) {
        throw std::runtime_error("[Error] PreComputedData environment_grid data failed to initilise: " + std::string(e.what()));
    }
    
    //using this one
    try {
        load_grid(m_electro_grid, py_pc, "electro_scaled");
    } catch (const std::exception &e) {
        throw std::runtime_error("[Error] PreComputedData electro_scaled data failed to initilise: " + std::string(e.what()));
    }
    
    try {
        load_grid(m_electro_grid_raw, py_pc, "electro_raw");
    } catch (const std::exception &e) {
        throw std::runtime_error("[Error] PreComputedData electro_raw data failed to initilise: " + std::string(e.what()));
    }
    
    try {
        load_grid(m_hydrophobic_grid_raw, py_pc, "hydrophob_raw");
    } catch (const std::exception &e) {
        throw std::runtime_error("[Error] PreComputedData hydrophob_raw data failed to initilise: " + std::string(e.what()));
    }
    
    try {
        load_grid(m_hydrophobic_enclosure_grid, py_pc, "hydrophob_enc");
    } catch (const std::exception &e) {
        throw std::runtime_error("[Error] PreComputedData hydrophob_enc data failed to initilise: " + std::string(e.what()));
    }
    
    try {
        load_grid(m_desolvation_polar_grid, py_pc, "desolvation_polar");
    } catch (const std::exception &e) {
        throw std::runtime_error("[Error] PreComputedData desolvation_polar data failed to initilise: " + std::string(e.what()));
    }
    
    try {
        load_grid(m_desolvation_hydrohpobic_grid, py_pc, "desolv_hphob");
    } catch (const std::exception &e) {
        throw std::runtime_error("[Error] PreComputedData desolv_hphob data failed to initilise: " + std::string(e.what()));
    }
    
     try {
         load_grid(m_delta_sasa_grid, py_pc, "delta_sasa");
     } catch (const std::exception &e) {
         throw std::runtime_error("[Error] PreComputedData delta_sasa data failed to initilise: " + std::string(e.what()));
     }
     //-----Score data-----
     //---Hbond score---
     
     
     try {
         //TODO! these are nonbonded!
         m_hbond_score_data.A_values  = py_pc.attr("A_values").cast<MatrixXd>();
         m_hbond_score_data.B_values  = py_pc.attr("B_values").cast<MatrixXd>();
         m_hbond_score_data.C_values  = py_pc.attr("C_values").cast<MatrixXd>();
         
         
         m_hbond_score_data.donor_types = py_pc.attr("hbond_donor_types").cast<VectorXi>();
         m_hbond_score_data.hbond_D = static_cast<int>(m_hbond_score_data.donor_types.size());
         for(int i = 0; i < m_hbond_score_data.hbond_D; ++i) {
             int type_id = m_hbond_score_data.donor_types(i);
             m_hbond_score_data.donor_index[m_hbond_score_data.donor_types(i)] = i;
             m_hbond_score_data.donor_types_set.insert(type_id);
         }
        
         // 2. Load Acceptors
         m_hbond_score_data.acceptor_types = py_pc.attr("hbond_acceptor_types").cast<VectorXi>();
         m_hbond_score_data.hbond_A = static_cast<int>(m_hbond_score_data.acceptor_types.size());
         for(int i = 0; i < m_hbond_score_data.hbond_A; ++i) {
            int type_id = m_hbond_score_data.acceptor_types(i);
            m_hbond_score_data.acceptor_index[m_hbond_score_data.acceptor_types(i)] = i;
            m_hbond_score_data.acceptor_types_set.insert(type_id);
         }
         
         load_polyX_values_flat(m_hbond_score_data.polyA_flat,
                                m_hbond_score_data.polyA_deg,
                                m_hbond_score_data.hbond_D,
                                m_hbond_score_data.hbond_A, 
                                py_pc, "hbond_polyA");
        
         load_polyX_values_flat(m_hbond_score_data.polyB_flat,
                                m_hbond_score_data.polyB_deg,
                                m_hbond_score_data.hbond_D,
                                m_hbond_score_data.hbond_A, 
                                py_pc, "hbond_polyB");
                                
         load_polyX_values_flat(m_hbond_score_data.polyC_flat,
                                m_hbond_score_data.polyC_deg,
                                m_hbond_score_data.hbond_D,
                                m_hbond_score_data.hbond_A, 
                                py_pc, "hbond_polyC");
         
         auto arr_hm = py_pc.attr("hbond_donor_acceptor_mask").cast<py::array_t<bool>>();
         
         int P = static_cast<int>(arr_hm.shape(0));
         int L = static_cast<int>(arr_hm.shape(1));
         
         m_hbond_score_data.hbond_mask.resize(P,L);
         for (int i=0; i<P; i++){
             for (int j=0; j<L; j++){
                 m_hbond_score_data.hbond_mask(i,j) = arr_hm.at(i, j);
             }
         }
        
     
     } catch (const py::cast_error &e) {
         throw std::runtime_error("[Error] A Python Hbond score data object could not be converted to the required C++ type: " + std::string(e.what()));
     } catch (const py::error_already_set &e) { 
         throw std::runtime_error("[Error] PreComputedData object is missing a required Hbond score parameter: " + std::string(e.what()));
     } catch (const std::exception &e) { 
         throw std::runtime_error("[Error] PrecomputedData Hbond score parameters failed to initilise: " + std::string(e.what()));
     }
     
     //---aromatic score data---
     try {
         m_aromatic_score_data.keys = py_pc.attr("arom_keys").cast<Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>();
         int M = static_cast<int>(m_aromatic_score_data.keys.rows());
         
         auto a_dims = py_pc.attr("arom_dimsA").cast<py::array_t<int>>();
         auto b_dims = py_pc.attr("arom_dimsB").cast<py::array_t<int>>();
         auto c_dims = py_pc.attr("arom_dimsC").cast<py::array_t<int>>();
         
         m_aromatic_score_data.nxA.resize(M);  m_aromatic_score_data.nyA.resize(M);
         m_aromatic_score_data.nxB.resize(M);  m_aromatic_score_data.nyB.resize(M);
         m_aromatic_score_data.nxC.resize(M);  m_aromatic_score_data.nyC.resize(M);
         
         for (int i=0; i < M; i++){
             m_aromatic_score_data.nxA[i] = a_dims.at(i,0);
             m_aromatic_score_data.nyA[i] = a_dims.at(i,1); 
             
             m_aromatic_score_data.nxB[i] = b_dims.at(i,0);
             m_aromatic_score_data.nyB[i] = b_dims.at(i,1);
             
             m_aromatic_score_data.nxC[i] = c_dims.at(i,0);
             m_aromatic_score_data.nyC[i] = c_dims.at(i,1);
             
         }
         
         auto arrA = py_pc.attr("arom_coefA").cast<py::list>();
         auto arrB = py_pc.attr("arom_coefB").cast<py::list>();
         auto arrC = py_pc.attr("arom_coefC").cast<py::list>();
         
         m_aromatic_score_data.coefA.resize(M);
         m_aromatic_score_data.coefB.resize(M);
         m_aromatic_score_data.coefC.resize(M);
         
         for (int i = 0; i < M; ++i) {
            
            auto mA = arrA[i].cast<py::array_t<double, py::array::c_style>>();
            auto mB = arrB[i].cast<py::array_t<double, py::array::c_style>>();
            auto mC = arrC[i].cast<py::array_t<double, py::array::c_style>>();
        
            size_t sizeA = static_cast<size_t>(mA.size());
            size_t sizeB = static_cast<size_t>(mB.size());
            size_t sizeC = static_cast<size_t>(mC.size());
        
            m_aromatic_score_data.coefA[i].assign(mA.data(), mA.data() + sizeA);
            m_aromatic_score_data.coefB[i].assign(mB.data(), mB.data() + sizeB);
            m_aromatic_score_data.coefC[i].assign(mC.data(), mC.data() + sizeC);
        }
         
         
         
     } catch (const py::cast_error &e){
          throw std::runtime_error("[Error] A Python Aromaticscore data object could not be converted to the required C++ type: " + std::string(e.what()));
     } catch (const py::error_already_set &e){
          throw std::runtime_error("[Error] PreComputedData object is missing a required Aromatic score parameter: " + std::string(e.what()));
     } catch (const std::exception &e) {
         throw std::runtime_error("[Error] PreComputedData Aromatic score failed to initilise: " + std::string(e.what()));
     }
     //-----ligand intra data-----
     try {
         auto arr_LA = py_pc.attr("LIGAND_INTRA_A_VALUES").cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        m_ligand_intra.intra_A_values = Eigen::Map<const MatrixXd>(arr_LA.data(), arr_LA.shape(0), arr_LA.shape(1));
        
        auto arr_LB = py_pc.attr("LIGAND_INTRA_B_VALUES").cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        m_ligand_intra.intra_B_values = Eigen::Map<const MatrixXd>(arr_LB.data(), arr_LB.shape(0), arr_LB.shape(1));
        
        auto arr_LC = py_pc.attr("LIGAND_INTRA_C_VALUES").cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        m_ligand_intra.intra_C_values = Eigen::Map<const MatrixXd>(arr_LC.data(), arr_LC.shape(0), arr_LC.shape(1));
        
        auto arr_LBD = py_pc.attr("ligand_bond_distances").cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
        m_ligand_intra.intra_bond_distances = Eigen::Map<const MatrixXi>(arr_LBD.data(), arr_LBD.shape(0), arr_LBD.shape(1));
        
       //-----bond distance arrays----
       
        const py::sequence py_constraints = py_pc.attr("constrain_atoms").cast<py::sequence>();
        m_ligand_intra.constrained_pair_idxs.clear();
        m_ligand_intra.constrained_pair_idxs.reserve(py_constraints.size());
    
        for (py::handle item : py_constraints) {
            auto [i, j] = item.cast<std::tuple<int,int>>();
    
            if (i < 0 || j < 0) {
                throw std::runtime_error("constrain_atoms contains a negative atom index");
            }
            if (i > j) std::swap(i, j);
            if (i == j) continue; // or throw
    
            m_ligand_intra.constrained_pair_idxs.emplace_back(i, j);
        }

        
        m_ligand_intra.constrained_atom_distances = py_pc.attr("constrain_atoms_dist").cast<std::vector<double>>();
        m_ligand_intra.constrained_pairs_to_ignore.clear();
        for (const auto& [i, j] : m_ligand_intra.constrained_pair_idxs) {
            m_ligand_intra.constrained_pairs_to_ignore.insert({i, j});
        }
        
        //-----ligand torsion data-----
       
        m_ligand_intra.torsion_end = py_pc.attr("end_torsions").cast<int>();
        const py::sequence py_idxs   = py_pc.attr("ligand_torsion_idxs").cast<py::sequence>();
        const py::sequence py_scores = py_pc.attr("ligand_torsion_scores").cast<py::sequence>();
        m_ligand_intra.n_torsions = static_cast<int>(py_idxs.size());
        
        m_ligand_intra.ligand_torsion_idxs.resize(m_ligand_intra.n_torsions);
        m_ligand_intra.ligand_torsion_scores.resize(m_ligand_intra.n_torsions);
        
        for (int t = 0; t < m_ligand_intra.n_torsions; ++t) {
            
            auto idx_arr = py_idxs[t].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
            if (idx_arr.ndim() != 1 || idx_arr.shape(0) != 4) {
                throw std::runtime_error("ligand_torsion_idxs[t] must be a 1D array of length 4");
            }
            auto iv = idx_arr.unchecked<1>();
            m_ligand_intra.ligand_torsion_idxs[t] = { iv(0), iv(1), iv(2), iv(3) };
            auto &prof_out = m_ligand_intra.ligand_torsion_scores[t];
            prof_out.clear();
        
            if (t < m_ligand_intra.torsion_end) {
                const py::sequence prof = py_scores[t].cast<py::sequence>();
                prof_out.reserve(prof.size());
        
                for (py::handle item : prof) {
                    
                    auto p = item.cast<std::pair<int,double>>();
                    prof_out.emplace_back(p.first, p.second);
                }
                std::sort(prof_out.begin(), prof_out.end(),
                          [](const auto& a, const auto& b){ return a.first < b.first; });
            }
        }
        
        //TODO! validate ligand score data
        
     } catch (const py::cast_error &e){
         throw std::runtime_error("[Error] A Python Aromaticscore data object could not be converted to the required C++ type:" + std::string(e.what()));
     } catch (const py::error_already_set &e) {
         throw std::runtime_error("[Error] PreComputedData object is missing a required ligand score parameter: "  + std::string(e.what()));
     } catch (const std::exception &e) {
         throw std::runtime_error("[Error] PreComputedData ligand score failed to initilise: " + std::string(e.what()));
     }
     
     //-----weights data-----
     try {
         m_weights.nonbond = py_pc.attr("w_nonbond").cast<double>(); 
         m_weights.dsasa = py_pc.attr("w_dsasa").cast<double>(); 
         m_weights.hphob = py_pc.attr("w_hphob").cast<double>(); 
         m_weights.electro = py_pc.attr("w_electro").cast<double>(); 
         m_weights.ligand_torsion = py_pc.attr("w_ligand_torsion").cast<double>(); 
         m_weights.ligand_intra = py_pc.attr("w_ligand_intra").cast<double>(); 
         m_weights.vdw = py_pc.attr("w_vdw").cast<double>(); 
         m_weights.hbond = py_pc.attr("w_hbond").cast<double>(); 
         m_weights.aromatic = py_pc.attr("w_aromatic").cast<double>(); 
         m_weights.halogen = py_pc.attr("w_halogen").cast<double>();
         m_weights.hphob_enc = py_pc.attr("w_hphob_enc").cast<double>();
         m_weights.constraint = py_pc.attr("w_constraint").cast<double>(); 
         m_weights.repCap_discrete = py_pc.attr("repCap_discrete").cast<double>();
         m_weights.repCap_inner_nm = py_pc.attr("repCap_inner_nm").cast<double>();
         m_weights.repCap_final_nm = py_pc.attr("repCap_final_nm").cast<double>();
         m_weights.rms_cutoff = py_pc.attr("rms_cutoff").cast<double>();
         validate_weights();
         
     } catch (const py::cast_error &e) {
         throw std::runtime_error("[Error] A Python weights data object could not be converted to the required C++ type: " + std::string(e.what()));
     } catch (const py::error_already_set &e) { 
         throw std::runtime_error("[Error] PreComputedData object is missing a required weights data parameter: " + std::string(e.what()));
     } catch (const std::exception &e) { 
         throw std::runtime_error("[Error] PreComputedData weights data failed to initilise: " + std::string(e.what()));
     }
     
     //-----algorithm config data-----
     
     
     
     try { 
         m_config.n_cpu = py_pc.attr("ncpu").cast<int>();
         m_config.n_global_search = py_pc.attr("n_global_search").cast<int>(); 
         m_config.n_local_search = py_pc.attr("n_local_search").cast<int>();
         m_config.repCap0 = py_pc.attr("repCap0").cast<double>();
         m_config.repCap1 = py_pc.attr("repCap1").cast<double>();
         m_config.repCap_inner_nm = py_pc.attr("repCap_inner_nm").cast<double>();
         m_config.repCap_final_nm = py_pc.attr("repCap_final_nm").cast<double>();
         m_config.rms_cutoff = py_pc.attr("rms_cutoff").cast<double>();
         m_config.a_lo = py_pc.attr("a_lo").cast<double>();
         m_config.a_mid = py_pc.attr("a_mid").cast<double>();
         m_config.a_hi = py_pc.attr("a_hi").cast<double>();
         m_config.iterations = py_pc.attr("iterations").cast<unsigned int>();
         m_config.bias_radius = py_pc.attr("bias_radius").cast<double>(); 
         m_config.binding_site_centroid = py_pc.attr("binding_site_centroid").cast<Eigen::RowVector3d>();
         m_config.nb_cell = py_pc.attr("nb_cell").cast<double>(); 
         
         auto py_all = py_pc.attr("all_arrays").cast<py::list>();
         m_config.translation_points.clear();
         m_config.all_arrays.clear();
         
         for (py::ssize_t dim =0; dim < py_all.size(); dim++){
             auto item = py_all[dim]; 
             if (dim == 0){
                 /* Treat translation points as (z,y,x) vectors of SASA voxels.
                 This avoids seeding ligand solutions within the protein itself 
                 and accelates convergance.
                 */
                 auto tpl_list = item.cast<py::list>();
                 m_config.translation_points.reserve(tpl_list.size());
                 for (auto const& tup_obj : tpl_list) {
                     auto tup = tup_obj.cast<py::tuple>();
                     if (tup.size() != 3) throw std::runtime_error("Translation tuple must have size 3.");
                    
                     m_config.translation_points.emplace_back(
                         tup[0].cast<int>(), 
                         tup[1].cast<int>(), 
                         tup[2].cast<int>()
                     );
                 }
            } else {
                auto arr = item.cast<py::array_t<int, py::array::c_style>>();
                std::vector<double> vec;
                vec.reserve(arr.size());
                for (py::ssize_t i = 0; i < arr.size(); ++i) {
                    vec.push_back(static_cast<double>(*arr.data(i)));
                }
                m_config.all_arrays.push_back(std::move(vec));

                }
            }
            
            auto py_adj = py_pc.attr("adjacency").cast<py::list>();
            m_config.adjacency.clear();
            m_config.adjacency.reserve(py_adj.size());
        
            for (auto const& row_obj : py_adj) {
                auto row = row_obj.cast<py::list>();
                
                std::vector<size_t> nbrs;
                nbrs.reserve(row.size());
        
                for (auto const& idx_obj : row) {
                    nbrs.push_back(idx_obj.cast<size_t>());
                }
                m_config.adjacency.push_back(std::move(nbrs));
            }
            
            m_config.prot_hash.build(m_protein_data.positions, m_config.nb_cell);
            
        } catch (const py::cast_error &e){
            throw std::runtime_error("[Error] A Python config data object could not be converted to the required C++ type: " + std::string(e.what()));
        } catch (const py::error_already_set &e) {
            throw std::runtime_error("[Error] PreComputedData object is missing a required config data parameter: " + std::string(e.what()));
        } catch (const std::exception &e) {
            throw std::runtime_error("[Error] PreComputedData config data failed to initilise: " + std::string(e.what()));
        }
        
        try {
            m_aromatic_scorer.load_from(py_pc);
        } catch (const std::exception &e){
            throw std::runtime_error("[Error] PreComputedData aromatic scorer failed to initlise.");
        }
        
        
};
//TODO! skip mask 
//TODO! Densmap

