#pragma once 

#include <vector>
#include <utility>   
#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>


namespace py = pybind11;

using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatX3d =   Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

struct AromaticScorer {
    
    
    // load data from Python-side container
    void load_from(py::object py_pc);

    // score one ring–ring interaction
    std::pair<double, MatrixXd> score_interaction(int prot_type,
                      int lig_type,
                      const MatrixXd& prot_coords,
                      const MatrixXd& lig_coords,
                      const std::vector<int> &prot_hidx,
                      const std::vector<int> &lig_hidx) const;
    
   
    
private:
    
    // members loaded from Python
    MatrixXi m_keys;  
    int m_M;
    std::vector<int> m_nxA, m_nyA, m_nxB, m_nyB, m_nxC, m_nyC;
    
    
    std::vector<MatrixXd> m_coefA, m_coefB, m_coefC;
    
    Eigen::VectorXi m_kxA, m_kyA, m_kxB, m_kyB, m_kxC, m_kyC; 
    
    std::vector<std::vector<double>> m_knotsA_x, m_knotsA_y,
                                     m_knotsB_x, m_knotsB_y,
                                     m_knotsC_x, m_knotsC_y;
     
    
    MatrixXi m_dimsA, m_dimsB, m_dimsC;
    
    
    double bspline_basis(int i, int k,
                         const std::vector<double> &knots,
                         double x) const;
    
    double eval_spline(double x, double y,
                       const MatrixXd &coef,
                       const std::vector<double> &knotx,
                       const std::vector<double> &knoty,
                       int px, int py,
                       int nx, int ny) const;
};



