#pragma once
#include <Eigen/Dense>
#include <vector>
#include <GraphMol/ROMol.h>
#include <limits>
#include <GraphMol/Conformer.h>
#include <cmath>

 

namespace GeometryUtils {
    inline  Eigen::RowVector3d compute_centroid(const std::vector<Eigen::RowVector3d> &pts) {
        Eigen::RowVector3d sum = Eigen::RowVector3d::Zero();
        if (pts.empty()) {
            return sum;
        }
        for (auto& p : pts) sum += p;
        return sum / static_cast<double>(pts.size());
    }
    
    template <typename Derived>
    inline Eigen::RowVector3d compute_centroid(const Eigen::MatrixBase<Derived>& pts) {
        if (pts.cols() != 3) {
            throw std::runtime_error("[Error] compute_centroid expects an N x 3 matrix");
        }
        if (pts.rows() == 0) return Eigen::RowVector3d::Zero();
    
        Eigen::RowVector3d c;
        c = pts.colwise().mean();
        return c;
    }
    

    
    inline Eigen::RowVector3d compute_normal(const std::vector<Eigen::RowVector3d> &pts) {
    
        if (pts.size() < 3) {
            throw std::runtime_error("[Error] compute_normal expects >= 3 points");
        }
        Eigen::RowVector3d v1 = pts[1] - pts[0];
        Eigen::RowVector3d v2 = pts[2] - pts[0];
        auto n = v1.cross(v2);
        double nn = n.norm();
        if (nn < 1e-8) {
            throw std::runtime_error("[Error] Degenerate ring passed to GeometryUtils::compute_normal");
        }
        return n / nn;
    }
    
    template <typename Derived>
    inline Eigen::RowVector3d compute_normal(const Eigen::MatrixBase<Derived>& pts) {
        if (pts.cols() != 3) {
            throw std::runtime_error("[Error] compute_normal expects an N x 3 matrix");
        }
        if (pts.rows() < 3) {
            throw std::runtime_error("[Error] compute_normal expects >= 3 points");
        }
    
        const Eigen::RowVector3d c = compute_centroid(pts);
        Eigen::RowVector3d n = Eigen::RowVector3d::Zero();
        const int N = static_cast<int>(pts.rows());
    
        for (int i = 0; i < N; ++i) {
            const Eigen::RowVector3d p0 = pts.row(i) - c;
            const Eigen::RowVector3d p1 = pts.row((i + 1) % N) - c;
            n += p0.cross(p1);
        }
    
        const double nn = n.norm();
        if (nn < 1e-8) {
            throw std::runtime_error("[Error] Degenerate ring passed to GeometryUtils::compute_normal");
        }
        return n / nn;
    }
    
    static inline double calc_bond_angle(const Eigen::RowVector3d &p1,
                                         const Eigen::RowVector3d &p2,
                                         const Eigen::RowVector3d &p3) {
        Eigen::RowVector3d ba = p1 - p2;
        Eigen::RowVector3d bc = p3 - p2;
        double cosang = ba.dot(bc) / (ba.norm() * bc.norm());
        cosang = std::max(-1.0, std::min(1.0, cosang));
        return std::acos(cosang) * 180.0 / M_PI;
    }
    
    inline double buckingham(double r, double A, double B, double C) {
        return A * std::exp(-B*r) - C / std::pow(r,6);
    }
    
    
    inline std::size_t idx3D(int z, int y, int x, int nz, int ny, int nx) {
        return static_cast<std::size_t>((z * ny + y) * nx + x);
    }
    

    
    inline double calc_dihedral(
        const Eigen::RowVector3d &p1,
        const Eigen::RowVector3d &p2,
        const Eigen::RowVector3d &p3,
        const Eigen::RowVector3d &p4
    ) {
      
      
      Eigen::Vector3d b1 = (p2 - p1).transpose();
      Eigen::Vector3d b2 = (p3 - p2).transpose();
      Eigen::Vector3d b3 = (p4 - p3).transpose();

      Eigen::Vector3d n1 = b1.cross(b2);
      Eigen::Vector3d n2 = b2.cross(b3);
      
      double n1_len = n1.norm();
      double n2_len = n2.norm();
      
      if (n1_len < 1e-8 || n2_len < 1e-8) return 0.0; //degenerate 

      n1 /= n1_len;
      n2 /= n2_len;
      
      //Eigen::Vector3d m1 = n1.cross(b2.normalized());
      Eigen::Vector3d m1 = b2.normalized().cross( n1 );
      
      double x = n1.dot(n2);
      double y = m1.dot(n2);
      double angle_rad = std::atan2(y,x);
      
      
      return angle_rad * (180/M_PI);
    }
    
    // Normalize to [-180, 180)
    inline double wrap_deg_pm180(double a) noexcept {
        a = std::fmod(a, 360.0);
        if (a >= 180.0) a -= 360.0;
        if (a <  -180.0) a += 360.0;
        return a;
    }
    
    // Smallest absolute difference on a circle (degrees), result in [0, 180]
    inline double ang_diff_deg(double a, double b) noexcept {
        const double d = wrap_deg_pm180(a - b);
        return std::abs(d);
    }
    
    inline double find_closest_torsion_score(
        double angle_deg,
        const std::vector<std::pair<int,double>>& profile) noexcept
    {
        if (profile.empty()) return 0.0; 
    
        const double a = wrap_deg_pm180(angle_deg);
    
        double best_score = profile.front().second;
        double best_diff  = std::numeric_limits<double>::infinity();
    
        for (const auto& pr : profile) {
            const double p = static_cast<double>(pr.first); // profile angles are ints in [-180,180]
            const double d = ang_diff_deg(a, p);
            if (d < best_diff) {
                best_diff  = d;
                best_score = pr.second;
            }
        }
        return best_score;
    }

    
    inline double trilinear_sample(
        const std::vector<double>& grid,
        const Eigen::RowVector3d& pos,      // (x,y,z) in Å
        const Eigen::RowVector3d& origin,   // grid origin in Å
        const Eigen::RowVector3d& spacing,                     // isotropic spacing in Å
        int nz, int ny, int nx              // dims (z,y,x)
    ) {
        // Convert world coords -> fractional grid coords
        double gx = (pos.x() - origin.x()) / spacing[0];
        double gy = (pos.y() - origin.y()) / spacing[1];
        double gz = (pos.z() - origin.z()) / spacing[2];
    
        // Base indices (lower corner) and fractional offsets
        int x0 = static_cast<int>(std::floor(gx));
        int y0 = static_cast<int>(std::floor(gy));
        int z0 = static_cast<int>(std::floor(gz));
    
        double tx = gx - x0;
        double ty = gy - y0;
        double tz = gz - z0;
    
        // Need x0+1, y0+1, z0+1 inside grid
        if (x0 < 0 || x0 + 1 >= nx ||
            y0 < 0 || y0 + 1 >= ny ||
            z0 < 0 || z0 + 1 >= nz) {
            // outside grid → neutral/zero, pick what you want here
            return 0.0;
        }
    
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        int z1 = z0 + 1;
    
        // Fetch 8 neighbours
        double v000 = grid[idx3D(z0, y0, x0, nz, ny, nx)];
        double v100 = grid[idx3D(z0, y0, x1, nz, ny, nx)];
        double v010 = grid[idx3D(z0, y1, x0, nz, ny, nx)];
        double v110 = grid[idx3D(z0, y1, x1, nz, ny, nx)];
        double v001 = grid[idx3D(z1, y0, x0, nz, ny, nx)];
        double v101 = grid[idx3D(z1, y0, x1, nz, ny, nx)];
        double v011 = grid[idx3D(z1, y1, x0, nz, ny, nx)];
        double v111 = grid[idx3D(z1, y1, x1, nz, ny, nx)];
    
        // Interpolate in x
        double vx00 = v000 * (1.0 - tx) + v100 * tx;
        double vx10 = v010 * (1.0 - tx) + v110 * tx;
        double vx01 = v001 * (1.0 - tx) + v101 * tx;
        double vx11 = v011 * (1.0 - tx) + v111 * tx;
    
        // Interpolate in y
        double vxy0 = vx00 * (1.0 - ty) + vx10 * ty;
        double vxy1 = vx01 * (1.0 - ty) + vx11 * ty;
    
        // Interpolate in z
        double vxyz = vxy0 * (1.0 - tz) + vxy1 * tz;
    
        return vxyz;
    }
    
   static inline double eval_poly(const double* coeffs, int degree, double x) noexcept {
        double result = coeffs[0];
        for (int i = 1; i <= degree; ++i) {
            result = result * x + coeffs[i];
        }
        return result;
    }
    
    
   
    inline double heavy_atom_rmsd(
        const RDKit::ROMol &m1,
        const RDKit::ROMol &m2
    ) {
        const auto &c1 = m1.getConformer();
        const auto &c2 = m2.getConformer();
    
        // Ensure both conformers are 3D and have the same number of atoms
        // (Assuming m1 and m2 are the same molecule/topology for RMSD)
        if (!c1.is3D() || !c2.is3D() || m1.getNumAtoms() != m2.getNumAtoms()) {
            return std::numeric_limits<double>::infinity();
        }
    
        double sum = 0.0;
        std::size_t n = 0;
    
        for (const auto &atom : m1.atoms()) {
            // Atomic number 1 is Hydrogen; anything else is a heavy atom
            if (atom->getAtomicNum() > 1) {
                unsigned int idx = atom->getIdx();
                
                const auto &p1 = c1.getAtomPos(idx);
                const auto &p2 = c2.getAtomPos(idx);
                
                double dx = p1.x - p2.x;
                double dy = p1.y - p2.y;
                double dz = p1.z - p2.z;
                
                sum += dx*dx + dy*dy + dz*dz;
                ++n;
            }
        }
    
        if (n == 0) {
            return std::numeric_limits<double>::infinity();
        }
    
        return std::sqrt(sum / static_cast<double>(n));
    }
}

