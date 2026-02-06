
#include <vector>            
#include <cmath>             
#include <algorithm>         
#include <limits>            
#include <utility>           

#include <Eigen/Dense>       
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>    
#include <pybind11/eigen.h>  
#include <pybind11/numpy.h>
#include <stdexcept>
#include "GeometryUtils.h"
#include "Scorers.h"

/**
 * @brief Calculates the B-spline basis function value using the Cox-de Boor recursion formula.
 * * This is a recursive implementation that determines the influence of a specific 
 * control point 'i' on the final curve at coordinate 'x'.
 * * @param i      The index of the basis function (corresponding to a control point).
 * @param k      The degree of the spline (e.g., 3 for cubic).
 * @param knots  The knot vector defining the parameter space.
 * @param x      The evaluation coordinate (offset or angle).
 * @return       The value of the i-th basis function of degree k at point x.
 * * @note If k=0, the function returns a step function (1.0 if x is within the knot span).
 * @warning Recursive implementation; depth is proportional to the spline degree 'k'.
 */
double AromaticScorer::bspline_basis(int i, int k,
                                     const std::vector<double>& knots,
                                     double x) const {
    if (k == 0) {
        if ((knots[i] <= x && x < knots[i+1]) ||
            (x == knots.back() && i+1 == int(knots.size())-1))
            return 1.0;
        return 0.0;
    }
    // left term
    double denom1 = knots[i+k] - knots[i];
    double term1 = denom1>0
        ? (x - knots[i]) / denom1 * bspline_basis(i, k-1, knots, x)
        : 0.0;
    // right term
    double denom2 = knots[i+k+1] - knots[i+1];
    double term2 = denom2>0
        ? (knots[i+k+1] - x) / denom2 * bspline_basis(i+1, k-1, knots, x)
        : 0.0;
    return term1 + term2;
}


/**
 * @brief Evaluates a 2D B-spline surface at a specific (x, y) coordinate.
 * * This function uses a tensor-product approach: it calculates 1D basis functions for 
 * both dimensions and then combines them with a coefficient matrix to find the 
 * interpolated value on the aromatic scoring grid.
 * @param x      The first coordinate for evaluation (e.g., angle).
 * @param y      The second coordinate for evaluation (e.g., offset).
 * @param coef   The 2D Matrix of spline coefficients for the current ring pair.
 * @param knotx  The knot vector for the x-dimension.
 * @param knoty  The knot vector for the y-dimension.
 * @param px     The degree of the spline in the x-dimension.
 * @param py     The degree of the spline in the y-dimension.
 * @param nx     The number of control points/basis functions in the x-dimension.
 * @param ny     The number of control points/basis functions in the y-dimension.
 * @return       The interpolated score (energy parameter) at (x, y).
 */
double AromaticScorer::eval_spline(double x, double y,
                                   const MatrixXd &coef,
                                   const std::vector<double> &knotx,
                                   const std::vector<double> &knoty,
                                   int px, int py,
                                   int nx, int ny) const {
    std::vector<double> Nx(nx), Ny(ny);
    for (int a = 0; a < nx; ++a)
        Nx[a] = bspline_basis(a, px, knotx, x);
    for (int b = 0; b < ny; ++b)
        Ny[b] = bspline_basis(b, py, knoty, y);

    // tensor‐product evaluation
    double result = 0.0;
    for (int a = 0; a < nx; ++a) {
        if (Nx[a] == 0.0) continue;
        for (int b = 0; b < ny; ++b) {
            if (Ny[b] == 0.0) continue;
            result += coef(a, b) * Nx[a] * Ny[b];
        }
    }
    return result;
}


void AromaticScorer::load_from(py::object py_pc){
    
    m_keys = py_pc.attr("arom_keys").cast<MatrixXi>();
    m_M = static_cast<int>(m_keys.rows());
    
    
    m_dimsA = py_pc.attr("arom_dimsA").cast<py::array_t<int, py::array::c_style>>().cast<MatrixXi>();
    m_dimsB = py_pc.attr("arom_dimsB").cast<py::array_t<int, py::array::c_style>>().cast<MatrixXi>();
    m_dimsC = py_pc.attr("arom_dimsC").cast<py::array_t<int, py::array::c_style>>().cast<MatrixXi>();
    
    
    
    m_nxA.resize(m_M);  m_nyA.resize(m_M);
    m_nxB.resize(m_M);  m_nyB.resize(m_M);
    m_nxC.resize(m_M);  m_nyC.resize(m_M);
     
    for (int i=0; i < m_M; i++){
        m_nxA[i] = m_dimsA(i,0);
        m_nyA[i] = m_dimsA(i,1); 
         
        m_nxB[i] = m_dimsB(i,0);
        m_nyB[i] = m_dimsB(i,1);
         
        m_nxC[i] = m_dimsC(i,0);
        m_nyC[i] = m_dimsC(i,1);
    }
     
    auto listA = py_pc.attr("arom_coefA").cast<py::list>();
    auto listB = py_pc.attr("arom_coefB").cast<py::list>();
    auto listC = py_pc.attr("arom_coefC").cast<py::list>();

    m_coefA.resize(m_M);
    m_coefB.resize(m_M);
    m_coefC.resize(m_M);

    for (int i = 0; i < m_M; ++i) {
        
        m_coefA[i] = listA[i].cast<py::array_t<double, py::array::c_style>>().cast<MatrixXd>();
        m_coefB[i] = listB[i].cast<py::array_t<double, py::array::c_style>>().cast<MatrixXd>();
        m_coefC[i] = listC[i].cast<py::array_t<double, py::array::c_style>>().cast<MatrixXd>();

        
        if (m_coefA[i].rows() != m_dimsA(i, 0) || m_coefA[i].cols() != m_dimsA(i, 1)) {
            throw std::runtime_error("Aromatic Coefficient Grid size mismatch at slot " + std::to_string(i));
        }
    }
     
     m_kxA = py_pc.attr("arom_kxA").cast<Eigen::VectorXi>();
     m_kyA = py_pc.attr("arom_kyA").cast<Eigen::VectorXi>();

     m_kxB = py_pc.attr("arom_kxB").cast<Eigen::VectorXi>();
     m_kyB = py_pc.attr("arom_kyB").cast<Eigen::VectorXi>();

     m_kxC = py_pc.attr("arom_kxC").cast<Eigen::VectorXi>();
     m_kyC = py_pc.attr("arom_kyC").cast<Eigen::VectorXi>();
     
     //
     py::list pax = py_pc.attr("arom_knots_xA").cast<py::list>(),
              pay = py_pc.attr("arom_knots_yA").cast<py::list>();
     
     py::list pbx = py_pc.attr("arom_knots_xB").cast<py::list>(),
              pby = py_pc.attr("arom_knots_yB").cast<py::list>();
      
     
     py::list pcx = py_pc.attr("arom_knots_xC").cast<py::list>(),
              pcy = py_pc.attr("arom_knots_yC").cast<py::list>();
     
     
     m_knotsA_x.resize(m_M);
     m_knotsA_y.resize(m_M);
     
     m_knotsB_x.resize(m_M);
     m_knotsB_y.resize(m_M);
     
     m_knotsC_x.resize(m_M);
     m_knotsC_y.resize(m_M);
     
     for (int i = 0; i < m_M; ++i) {
        
        auto arrAX = pax[i].cast<py::array_t<double, py::array::c_style>>();
        auto arrAY = pay[i].cast<py::array_t<double, py::array::c_style>>();
        auto arrBX = pbx[i].cast<py::array_t<double, py::array::c_style>>();
        auto arrBY = pby[i].cast<py::array_t<double, py::array::c_style>>();
        auto arrCX = pcx[i].cast<py::array_t<double, py::array::c_style>>();
        auto arrCY = pcy[i].cast<py::array_t<double, py::array::c_style>>();
    
        m_knotsA_x[i].assign(arrAX.data(), arrAX.data() + arrAX.size());
        m_knotsA_y[i].assign(arrAY.data(), arrAY.data() + arrAY.size());
        
        m_knotsB_x[i].assign(arrBX.data(), arrBX.data() + arrBX.size());
        m_knotsB_y[i].assign(arrBY.data(), arrBY.data() + arrBY.size());
        
        m_knotsC_x[i].assign(arrCX.data(), arrCX.data() + arrCX.size());
        m_knotsC_y[i].assign(arrCY.data(), arrCY.data() + arrCY.size());
        
        
     }
     
     
     // ---------------------------------------
      
}

/**
 * @brief Finds the index (slot) for a pre-computed spline based on ring types and stack mode.
 * @return The index if found, or -1 if no match exists.
 */
inline int find_spline_slot(int t0, int t1, int stack_int, int M, const MatrixXi &keys ) {
     
     for (int i = 0; i < M; ++i) {
       if ( keys(i,0)==t0 && keys(i,1)==t1 && keys(i,2)==stack_int ) { 
           return i;
           }
     }
     //swapped
     for (int i = 0; i < M; ++i) {
       if ( keys(i,0)==t1 && keys(i,1)==t0 && keys(i,2)==stack_int ) { 
           return i;
           }
     }
     
     return -1;
}

/**
 * @brief Computes the aromatic interaction score based on three geometric descriptors.
 * * The total score is derived from the interpolation of three distinct features:
 * 1. Centroid-Centroid Distance
 * 2. Plane-Plane Angle 
 * 3. Ring-Ring Offset 
 * * Each feature is evaluated within specific physical bounds.
 * @warning Scoring may have undefined behavior outside of these bounds.
 */
std::pair<double,MatrixXd>
AromaticScorer::score_interaction(int prot_type,
                                  int lig_type,
                                  const MatrixXd& prot_coords,
                                  const MatrixXd& lig_coords,
                                  const std::vector<int>& prot_hidx,
                                  const std::vector<int>& lig_hidx) const {
    
    
    
    
    Eigen::RowVector3d c1 = GeometryUtils::compute_centroid(prot_coords);
    Eigen::RowVector3d c2 = GeometryUtils::compute_centroid(lig_coords);
    double d_raw = (c2 - c1).norm();
    double d = std::round(d_raw*1000.0)/1000.0;
    if (d <= 2.0 || d >= 8.0) return { std::numeric_limits<double>::quiet_NaN(), {} };
    //-----offset-----
    Eigen::RowVector3d n1 = GeometryUtils::compute_normal(prot_coords);
    Eigen::RowVector3d n2 = GeometryUtils::compute_normal(lig_coords);
    Eigen::RowVector3d delta = c2 - c1;
    double comp1 = delta.dot(n1);
    double comp2 = (-delta).dot(n2);
    double off1 = ( (c2 - comp1*n1) - c1 ).norm();
    double off2 = ( (c1 - comp2*n2) - c2 ).norm();
    double offset_raw = std::min(off1, off2);
    double offset = std::round(offset_raw*1000.0)/1000.0;
    if (offset >= 2.5) return { std::numeric_limits<double>::quiet_NaN(), {} };
    //-----angle-----  
    double dot = std::abs(n1.dot(n2));
    double ang_raw = std::acos(std::min(1.0,std::max(-1.0,dot))) * 180.0/M_PI;
    double ang = std::round(ang_raw*100.0)/100.0;
    if (ang > 90.0) ang = 180.0 - ang;
      
    // choose a p (angle <= 25.0) t (angle >= 65.0)  stack and lookup spline index.
    int stack_int = (ang <= 25.0 ? 0 : (ang >= 65.0 ? 1 : -1));
    if (stack_int<0) return { std::numeric_limits<double>::quiet_NaN(), {} };
    
    
    // Initialize lookup keys; default to Protein (t0) and Ligand (t1)
    int t0 = prot_type, t1 = lig_type;
    
    // Handle T-stack orientation (stack_int == 1).
    // In T-stacks, the scoring grid is asymmetric. We must identify which ring 
    // is the 'Face' (horizontal) and which is the 'Edge' (perpendicular).
    // The convention requires the 'Edge' ring to be the first key (t0).
    if(stack_int == 1) {               
        if (prot_type == lig_type) {
            // identical ring types: force the “swapped” key
            t0 = lig_type;
            t1 = prot_type;
            
        } else {
              
            Eigen::RowVector3d vec_unit = (c2 - c1).normalized();
            double lig_dot  = std::abs(n1.dot(vec_unit));
            double prot_dot = std::abs(n2.dot(-vec_unit));
            if (lig_dot > prot_dot) {
                // ligand ring is “edge”, put prot first in the key
                t0 = lig_type; 
                t1 = prot_type;
            }
        }
    }
    
    // find spline slot
    int slot = find_spline_slot(t0, t1, stack_int,  m_M, m_keys );
    if (slot==-1) return { std::numeric_limits<double>::quiet_NaN(), {} };
    
    double A = eval_spline(ang, offset,
                       m_coefA[slot],
                       m_knotsA_x[slot],
                       m_knotsA_y[slot],
                       m_kxA[slot],
                       m_kyA[slot],
                       m_dimsA(slot,0),
                       m_dimsA(slot,1));

    double B = eval_spline(ang, offset,
                           m_coefB[slot],
                           m_knotsB_x[slot],
                           m_knotsB_y[slot],
                           m_kxB[slot],
                           m_kyB[slot],
                           m_dimsB(slot,0),
                           m_dimsB(slot,1));
    
    double C = eval_spline(ang, offset,
                           m_coefC[slot],
                           m_knotsC_x[slot],
                           m_knotsC_y[slot],
                           m_kxC[slot],
                           m_kyC[slot],
                           m_dimsC(slot,0),
                           m_dimsC(slot,1));

    
    
    double score = GeometryUtils::buckingham(d, A,B,C);
    
    //-----clash score-----
    
    int p_slot = slot;
    if (stack_int != 0) {
        p_slot = find_spline_slot(t0, t1, 0,  m_M, m_keys );
    } 
    
    if (p_slot < 0) {
      // no p-stack spline available, return the ring–ring score
      return { score, {} };
    }
  
    
    double A_p = eval_spline(0.0, 0.0,
                       m_coefA[p_slot],
                       m_knotsA_x[p_slot],
                       m_knotsA_y[p_slot],
                       m_kxA[p_slot],
                       m_kyA[p_slot],
                       m_dimsA(p_slot,0),
                       m_dimsA(p_slot,1));

    double B_p = eval_spline(0.0, 0.0,
                           m_coefB[p_slot],
                           m_knotsB_x[p_slot],
                           m_knotsB_y[p_slot],
                           m_kxB[p_slot],
                           m_kyB[p_slot],
                           m_dimsB(p_slot,0),
                           m_dimsB(p_slot,1));
    
    double C_p = eval_spline(0.0, 0.0,
                           m_coefC[p_slot],
                           m_knotsC_x[p_slot],
                           m_knotsC_y[p_slot],
                           m_kxC[p_slot],
                           m_kyC[p_slot],
                           m_dimsC(p_slot,0),
                           m_dimsC(p_slot,1));

    
    
    
    int Ni = int(lig_hidx.size()), Nj = int(prot_hidx.size());
    MatrixXd clash(Ni, Nj);
    for (int i = 0; i < Ni; ++i) {
      //Eigen::RowVector3d li = lig_coords[i];
      Eigen::RowVector3d li = lig_coords.row(i);
      
      for (int j = 0; j < Nj; ++j) {
        //Eigen::RowVector3d pj = prot_coords[j];
        Eigen::RowVector3d pj = prot_coords.row(j);
        double r = (li-pj).norm();
        if (r < 2.0) r = 2.0;
        double v = GeometryUtils::buckingham(r, A_p,B_p,C_p) / 6.0;
        clash(i,j) = std::max(0.0, std::min(v,10.0));
      }
    }
    
  
  return { score, clash };                       
}


/*
class HalogenScorer {
    public:
        
        void load_from(py::object py_pc);
    
        
         
        double score_interaction(
            const Eigen::RowVector3d& donor_atom,
            const Eigen::RowVector3d& donor_root_atom,
            const Eigne::RowVector3d& acceptor_atom,
            const Eigne::RowVector3d& acceptor_root_atom,
            int donor_atom_type,
            int acceptor_atom_type) const;
    
    private:
    
        
        int m_M; // Number of spline models
        // Key format: (donor_idx, acceptor_idx)
        MatrixXi m_keys;
    
        // Spline data for the A,B,C Buckingham parameter
        VectorXi m_kxA, m_kyA;
        MatrixXi m_dimsA;
        std::vector<MatrixXd> m_coefA;
        std::vector<std::vector<double>> m_knotsA_x, m_knotsA_y;
    
        VectorXi m_kxB, m_kyB;
        MatrixXi m_dimsB;
        std::vector<MatrixXd> m_coefB;
        std::vector<std::vector<double>> m_knotsB_x, m_knotsB_y;

        VectorXi m_kxC, m_kyC;
        MatrixXi m_dimsC;
        std::vector<MatrixXd> m_coefC;
        std::vector<std::vector<double>> m_knotsC_x, m_knotsC_y;
        
        int find_span(int n, int k, double x, const std::vector<double>& knots) const;

        double bspline_basis(int i, int k,
                             const std::vector<double> &knots,
                             double x) const;
        
        double eval_spline(int idx,
                         double x, double y,
                         const MatrixXd &coef,
                         const std::vector<double> &knotx,
                         const std::vector<double> &knoty,
                         int px, int py,
                         int nx, int ny) const;

        static double compute_angle(const Eigen::RowVector3d& p1, const Eigen::RowVector3d& p2, const Eigen::RowVector3d& p3);
    };
*/
