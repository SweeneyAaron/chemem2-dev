// density_simulate.hpp
#pragma once

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include "GeometryUtils.h"

// Your requested type alias
using MatX3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

// z-major flattening (x fastest): ((z*ny) + y)*nx + x
inline std::size_t idx3D(const int z, const int y, const int x,
                         const int nz, const int ny, const int nx) {
    (void)nz; // unused but kept to match your old signature style
    return static_cast<std::size_t>((z * ny + y) * nx + x);
}


struct SciChanSummary {
    double mean_nn=0, mean_tri=0, max_nn=0, max_tri=0;
    double top20_tri=0, p50_tri=0, p90_tri=0, p95_tri=0;
    int hit001=0, hit01=0, hit05=0;
};

static inline SciChanSummary summarize_vals(
    const std::vector<double>& nn,
    const std::vector<double>& tri
) {
    SciChanSummary s;
    if (tri.empty()) return s;

    s.mean_nn  = std::accumulate(nn.begin(), nn.end(), 0.0) / (double)nn.size();
    s.mean_tri = std::accumulate(tri.begin(), tri.end(), 0.0) / (double)tri.size();

    s.max_nn   = *std::max_element(nn.begin(), nn.end());
    s.max_tri  = *std::max_element(tri.begin(), tri.end());

    // hits on tri (more stable)
    for (double v : tri) {
        if (v > 0.001) s.hit001++;
        if (v > 0.010) s.hit01++;
        if (v > 0.050) s.hit05++;
    }

    // quantiles (N is small; sorting is fine)
    std::vector<double> sorted = tri;
    std::sort(sorted.begin(), sorted.end());
    auto pick = [&](double q){
        double idx = q * (sorted.size() - 1);
        size_t i0 = (size_t)std::floor(idx);
        size_t i1 = std::min(i0 + 1, sorted.size() - 1);
        double t = idx - (double)i0;
        return sorted[i0]*(1.0 - t) + sorted[i1]*t;
    };
    s.p50_tri = pick(0.50);
    s.p90_tri = pick(0.90);
    s.p95_tri = pick(0.95);

    // top 20% mean
    std::vector<double> tmp = tri;
    int k = std::max(1, (int)tmp.size() / 5);
    std::nth_element(tmp.begin(), tmp.begin() + (k - 1), tmp.end(), std::greater<double>());
    std::sort(tmp.begin(), tmp.begin() + k, std::greater<double>());
    s.top20_tri = std::accumulate(tmp.begin(), tmp.begin() + k, 0.0) / (double)k;

    return s;
}

namespace DensityUtils {
    inline std::vector<double> simulate_density_map(
        const Eigen::RowVector3d& origin,     // (Å) {ox, oy, oz}
        const Eigen::RowVector3d& apix,       // (Å/voxel) {ax, ay, az}
        const int nz, const int ny, const int nx,
        const Eigen::Ref<const MatX3d>& coords_xyz,  // N x 3, x/y/z in columns 0/1/2
        const std::vector<double>& atom_masses,      // size N
        double resolution,
        double sigma_coeff,
        bool normalise,
        int n_heavy = 0
    ) {
        int nAtoms;
        if (n_heavy == 0){ 
            nAtoms = static_cast<int>(coords_xyz.rows());
        } else {
            nAtoms = n_heavy;
        }
        if (nAtoms == 0 || nz <= 0 || ny <= 0 || nx <= 0) {
            return {};
        }
        
        if (apix[0] == 0.0 || apix[1] == 0.0 || apix[2] == 0.0) {
            throw std::runtime_error("simulate_density_map: apix contains zero");
        }
    
        std::vector<double> density(static_cast<std::size_t>(nz) * ny * nx, 0.0);
    
        // Deposit atom masses into nearest voxel
        for (int i = 0; i < nAtoms; ++i) {
            const double x = coords_xyz(i, 0);
            const double y = coords_xyz(i, 1);
            const double z = coords_xyz(i, 2);
    
            const double gx = (x - origin[0]) / apix[0];
            const double gy = (y - origin[1]) / apix[1];
            const double gz = (z - origin[2]) / apix[2];
    
            const int ix = static_cast<int>(std::llround(gx));
            const int iy = static_cast<int>(std::llround(gy));
            const int iz = static_cast<int>(std::llround(gz));
    
            if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
                density[idx3D(iz, iy, ix, nz, ny, nx)] += atom_masses[static_cast<std::size_t>(i)];
            }
        }
    
        // Gaussian blur (separable)
        const double sigma = sigma_coeff * resolution; // Å
        const double apix_iso = (apix[0] + apix[1] + apix[2]) / 3.0;
        if (apix_iso <= 0.0) {
            throw std::runtime_error("simulate_density_map: invalid apix_iso");
        }
    
        const double sigma_vox = sigma / apix_iso; // voxels
        if (sigma_vox > 0.0) {
            const int radius = static_cast<int>(std::ceil(3.0 * sigma_vox)); // ~3σ
            const int ksize  = 2 * radius + 1;
    
            std::vector<double> kernel(static_cast<std::size_t>(ksize));
            double ksum = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                const double t = static_cast<double>(k);
                const double w = std::exp(-0.5 * (t * t) / (sigma_vox * sigma_vox));
                kernel[static_cast<std::size_t>(k + radius)] = w;
                ksum += w;
            }
            if (ksum > 0.0) {
                for (double& w : kernel) w /= ksum;
            }
    
            std::vector<double> tmp1(density.size(), 0.0);
            std::vector<double> tmp2(density.size(), 0.0);
    
            // X
            for (int z = 0; z < nz; ++z) {
                for (int y = 0; y < ny; ++y) {
                    for (int x = 0; x < nx; ++x) {
                        double sum = 0.0, norm = 0.0;
                        for (int k = -radius; k <= radius; ++k) {
                            const int xx = x + k;
                            if (xx < 0 || xx >= nx) continue;
                            const double w = kernel[static_cast<std::size_t>(k + radius)];
                            sum  += w * density[idx3D(z, y, xx, nz, ny, nx)];
                            norm += w;
                        }
                        if (norm > 0.0) sum /= norm;
                        tmp1[idx3D(z, y, x, nz, ny, nx)] = sum;
                    }
                }
            }
    
            // Y
            for (int z = 0; z < nz; ++z) {
                for (int y = 0; y < ny; ++y) {
                    for (int x = 0; x < nx; ++x) {
                        double sum = 0.0, norm = 0.0;
                        for (int k = -radius; k <= radius; ++k) {
                            const int yy = y + k;
                            if (yy < 0 || yy >= ny) continue;
                            const double w = kernel[static_cast<std::size_t>(k + radius)];
                            sum  += w * tmp1[idx3D(z, yy, x, nz, ny, nx)];
                            norm += w;
                        }
                        if (norm > 0.0) sum /= norm;
                        tmp2[idx3D(z, y, x, nz, ny, nx)] = sum;
                    }
                }
            }
    
            // Z (back into density)
            for (int z = 0; z < nz; ++z) {
                for (int y = 0; y < ny; ++y) {
                    for (int x = 0; x < nx; ++x) {
                        double sum = 0.0, norm = 0.0;
                        for (int k = -radius; k <= radius; ++k) {
                            const int zz = z + k;
                            if (zz < 0 || zz >= nz) continue;
                            const double w = kernel[static_cast<std::size_t>(k + radius)];
                            sum  += w * tmp2[idx3D(zz, y, x, nz, ny, nx)];
                            norm += w;
                        }
                        if (norm > 0.0) sum /= norm;
                        density[idx3D(z, y, x, nz, ny, nx)] = sum;
                    }
                }
            }
        }
    
        // Normalise to max=1
        if (normalise) {
            auto max_it = std::max_element(density.begin(), density.end());
            if (max_it != density.end() && *max_it > 0.0) {
                const double max_val = *max_it;
                for (double& v : density) v /= max_val;
            }
        }
    
        return density;
    }
    
        
    inline double calc_mutual_information(
        const std::vector<double>& exp_map,
        const std::vector<double>& sim_map,
        int bins
    ) {
        
        if (sim_map.size() != exp_map.size()) {
            throw std::runtime_error("calc_mutual_information: exp_map and sim_map must have same size");
        }
    
        const std::size_t N = exp_map.size();
        if (N == 0 || bins <= 1) {
            return 0.0;
        }
        auto [exp_min_it, exp_max_it] = std::minmax_element(exp_map.begin(), exp_map.end());
        auto [sim_min_it, sim_max_it] = std::minmax_element(sim_map.begin(), sim_map.end());
        
        double exp_min = *exp_min_it;
        double exp_max = *exp_max_it;
        double sim_min = *sim_min_it;
        double sim_max = *sim_max_it;
        
        if (exp_max == exp_min) {
            exp_max = exp_min + 1.0;
        }
        if (sim_max == sim_min) {
            sim_max = sim_min + 1.0;
        }
        
        const double exp_bin_width = (exp_max - exp_min) / static_cast<double>(bins);
        const double sim_bin_width = (sim_max - sim_min) / static_cast<double>(bins);
        
        std::vector<double> hist_2d(static_cast<std::size_t>(bins) * bins, 0.0);
        
        for (std::size_t n = 0; n < N; ++n) {
            const double xv = exp_map[n];
            const double yv = sim_map[n];
    
            int bx = static_cast<int>((xv - exp_min) / exp_bin_width);
            int by = static_cast<int>((yv - sim_min) / sim_bin_width);
    
            // Clamp to [0, bins-1] (handles edge cases where value == max)
            if (bx < 0) bx = 0;
            else if (bx >= bins) bx = bins - 1;
    
            if (by < 0) by = 0;
            else if (by >= bins) by = bins - 1;
    
            hist_2d[static_cast<std::size_t>(bx) * bins + by] += 1.0;
        }
        
        std::vector<double> joint_prob = hist_2d;
        const double invN = 1.0 / static_cast<double>(N);
        for (double& v : joint_prob) {
            v *= invN;
        }
        std::vector<double> exp_prob(bins, 0.0);
        std::vector<double> sim_prob(bins, 0.0);
        
        for (int i = 0; i < bins; ++i) {
            for (int j = 0; j < bins; ++j) {
                double p = joint_prob[static_cast<std::size_t>(i) * bins + j];
                exp_prob[i] += p;  // row sum
                sim_prob[j] += p;  // col sum
            }
        }
        
        double MI = 0.0;
        const double log2e = 1.0 / std::log(2.0); // convert ln -> log2
    
        for (int i = 0; i < bins; ++i) {
            for (int j = 0; j < bins; ++j) {
                double p = joint_prob[static_cast<std::size_t>(i) * bins + j];
                if (p <= 0.0) continue;
    
                double denom = exp_prob[i] * sim_prob[j];
                if (denom <= 0.0) continue;
    
                // log2(p/denom) = ln(p/denom) * log2e
                MI += p * (std::log(p / denom) * log2e);
            }
        }
    
        return MI;
    }
    

    static inline double sample_trilin_flat(
        const std::vector<double> &grid, int nx, int ny, int nz,
        double fx, double fy, double fz
    ) {
        int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy), z0 = (int)std::floor(fz);
        int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
        if (x0 < 0 || y0 < 0 || z0 < 0 || x1 >= nx || y1 >= ny || z1 >= nz) return 0.0;

        double tx = fx - x0, ty = fy - y0, tz = fz - z0;

        auto I = [&](int x,int y,int z){ return idx3D(z,y,x,nz,ny,nx); };

        double c000 = grid[I(x0,y0,z0)], c100 = grid[I(x1,y0,z0)];
        double c010 = grid[I(x0,y1,z0)], c110 = grid[I(x1,y1,z0)];
        double c001 = grid[I(x0,y0,z1)], c101 = grid[I(x1,y0,z1)];
        double c011 = grid[I(x0,y1,z1)], c111 = grid[I(x1,y1,z1)];

        double c00 = c000*(1-tx) + c100*tx;
        double c10 = c010*(1-tx) + c110*tx;
        double c01 = c001*(1-tx) + c101*tx;
        double c11 = c011*(1-tx) + c111*tx;

        double c0 = c00*(1-ty) + c10*ty;
        double c1 = c01*(1-ty) + c11*ty;

        return c0*(1-tz) + c1*tz;
    }
    
    
    
    inline double score_sci_fast(
        const Eigen::Ref<const MatX3d>& ligand_xyz,   // N x 3 (x,y,z)
        const Eigen::RowVector3d& origin,
        const Eigen::RowVector3d& apix,
        int nx, int ny, int nz,
        const std::vector<double>& sci_grid,
        const std::vector<double>& sci_first_derivative_grid,
        const std::vector<double>& sci_second_derivative_grid,
        int n_heavy = 0
    ) {
        
        int n = (n_heavy > 0) ? n_heavy : static_cast<int>(ligand_xyz.rows());
        if (n <= 0) return 0.0;
        n = std::min(n, static_cast<int>(ligand_xyz.rows()));
    
        
        int inside = 0, outside = 0;
    
        std::vector<double> c0_nn, c0_tri, c1_nn, c1_tri, c2_nn, c2_tri;
        c0_nn.reserve(n); c0_tri.reserve(n);
        c1_nn.reserve(n); c1_tri.reserve(n);
        c2_nn.reserve(n); c2_tri.reserve(n);
    
        // ligand COM in fractional voxel coords
        double lig_fx_sum = 0.0, lig_fy_sum = 0.0, lig_fz_sum = 0.0;
        int lig_com_n = 0;
    
        for (int i = 0; i < n; ++i) {
            const double px = ligand_xyz(i, 0);
            const double py = ligand_xyz(i, 1);
            const double pz = ligand_xyz(i, 2);
    
            const double fx = (px - origin[0]) / apix[0];
            const double fy = (py - origin[1]) / apix[1];
            const double fz = (pz - origin[2]) / apix[2];
    
            const int gx = static_cast<int>(std::round(fx));
            const int gy = static_cast<int>(std::round(fy));
            const int gz = static_cast<int>(std::round(fz));
    
            if (gx < 0 || gx >= nx || gy < 0 || gy >= ny || gz < 0 || gz >= nz) {
                outside++;
                continue;
            }
            inside++;
    
            lig_fx_sum += fx; lig_fy_sum += fy; lig_fz_sum += fz; lig_com_n++;
    
            const std::size_t fi = idx3D(gz, gy, gx, nz, ny, nx);
    
            const double v0_nn = sci_grid[fi];
            const double v1_nn = sci_first_derivative_grid[fi];
            const double v2_nn = sci_second_derivative_grid[fi];
    
            const double v0_tr = DensityUtils::sample_trilin_flat(sci_grid, nx, ny, nz, fx, fy, fz);
            const double v1_tr = DensityUtils::sample_trilin_flat(sci_first_derivative_grid, nx, ny, nz, fx, fy, fz);
            const double v2_tr = DensityUtils::sample_trilin_flat(sci_second_derivative_grid, nx, ny, nz, fx, fy, fz);
    
            c0_nn.push_back(v0_nn); c0_tri.push_back(v0_tr);
            c1_nn.push_back(v1_nn); c1_tri.push_back(v1_tr);
            c2_nn.push_back(v2_nn); c2_tri.push_back(v2_tr);
        }
    
        if (inside == 0) return 0.0;
    
        // summarize channels
        SciChanSummary s0 = summarize_vals(c0_nn, c0_tri);
        SciChanSummary s1 = summarize_vals(c1_nn, c1_tri);
        SciChanSummary s2 = summarize_vals(c2_nn, c2_tri);
    
        // compute map “hot” COM once (threshold = 0.1*max)
        double hot_fx = 0.0, hot_fy = 0.0, hot_fz = 0.0;
        int hot_n = 0;
    
        double c0_max = 0.0;
        for (double v : sci_grid) c0_max = std::max(c0_max, v);
        const double thr = 0.1 * c0_max;
    
        for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            const double v = sci_grid[idx3D(z, y, x, nz, ny, nx)];
            if (v > thr) { hot_fx += x; hot_fy += y; hot_fz += z; hot_n++; }
        }
    
        double com_dist = -1.0; // kept (unused in current return)
        if (lig_com_n > 0 && hot_n > 0) {
            const double lig_fx = lig_fx_sum / lig_com_n;
            const double lig_fy = lig_fy_sum / lig_com_n;
            const double lig_fz = lig_fz_sum / lig_com_n;
            const double hotx = hot_fx / hot_n, hoty = hot_fy / hot_n, hotz = hot_fz / hot_n;
            const double dx = (lig_fx - hotx) * apix[0];
            const double dy = (lig_fy - hoty) * apix[1];
            const double dz = (lig_fz - hotz) * apix[2];
            com_dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        }
    
        (void)outside;
        (void)com_dist;
    
       
        const double sci_sum = (1.0 * s0.mean_tri + 1.0 * s1.mean_tri + 1.0 * s2.mean_tri);
        return sci_sum;
    }

}


