// electrostatic.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <limits>
#include <queue>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace py = pybind11;

//---------------------------------------------------------------------
// Helper Function: Distance-dependent dielectric
//---------------------------------------------------------------------
inline double distance_dependent_dielectric(double r, double eps0, double k) {
    return eps0 + k * r;
}

inline double sigmoidal_dielectric(double r) {
    const double A = -8.5525;
    const double eps0 = 78.4; // Dielectric constant of bulk water at 25°C
    const double B = eps0 - A; // B = ε₀ - A
    const double k = 7.7839;
    const double lambda = 0.003627;
    return A + B / (1.0 + k * std::exp(-lambda * B * r));
}

//---------------------------------------------------------------------
// Function 1: Compute Electrostatic Grid (with cutoff)
//---------------------------------------------------------------------
// Computes a 3D electrostatic potential grid using a cutoff distance.
py::array_t<double> compute_electostatic_grid(
    py::array_t<double> protein_positions,   // (N, 3)
    py::array_t<double> protein_charges,       // (N,)
    py::array_t<double> binding_site_map,      // (nz, ny, nx) [output buffer]
    py::array_t<double> binding_site_origin,   // (3,) origin: (x0, y0, z0)
    py::array_t<double> apix,                  // (3,) grid spacing: (apix_x, apix_y, apix_z)
    double c_factor = 332.06,
    double eps0 = 4.0,
    double k = 0.2,
    double min_r = 0.001,
    double cutoff_distance = 9.0
) {
    auto pos    = protein_positions.unchecked<2>();      // shape: (N, 3)
    auto charges= protein_charges.unchecked<1>();          // shape: (N,)
    auto map    = binding_site_map.mutable_unchecked<3>(); // shape: (nz, ny, nx)
    auto origin = binding_site_origin.unchecked<1>();      // (3,)
    auto apix_arr = apix.unchecked<1>();                   // (3,)

    ssize_t N  = protein_positions.shape(0);
    ssize_t nz = binding_site_map.shape(0);
    ssize_t ny = binding_site_map.shape(1);
    ssize_t nx = binding_site_map.shape(2);

    double x0 = origin(0);
    double y0 = origin(1);
    double z0 = origin(2);
    double apix_x = apix_arr(0);
    double apix_y = apix_arr(1);
    double apix_z = apix_arr(2);

    double cutoff_sq = cutoff_distance * cutoff_distance;

    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (ssize_t iz = 0; iz < nz; ++iz) {
        
        for (ssize_t iy = 0; iy < ny; ++iy) {
            
            for (ssize_t ix = 0; ix < nx; ++ix) {
                double z = z0 + iz * apix_z;
                double y = y0 + iy * apix_y;
                double x = x0 + ix * apix_x;
                double sum_q_over_epsr_r = 0.0;
                for (ssize_t i = 0; i < N; ++i) {
                    double dx = pos(i, 0) - x;
                    double dy = pos(i, 1) - y;
                    double dz = pos(i, 2) - z;
                    double dist_sq = dx * dx + dy * dy + dz * dz;
                    if (dist_sq <= cutoff_sq) {
                        double r = std::sqrt(dist_sq);
                        if (r < min_r)
                            r = min_r;
                        double eps_r = sigmoidal_dielectric(r);
                        //double eps_r = distance_dependent_dielectric(r, eps0, k);
                        sum_q_over_epsr_r += charges(i) / (eps_r * r);
                    }
                }
                map(iz, iy, ix) = c_factor * sum_q_over_epsr_r;
            }
        }
    }
    return binding_site_map;
}




py::array_t<double> compute_hydrophobic_grid_gaussian(
    py::array_t<double> protein_positions,      // (N, 3) atom coordinates
    py::array_t<double> protein_hpi,            // (N,) atom hydrophobicity values
    py::array_t<double> binding_site_map,       // (nz, ny, nx) [output buffer]
    py::array_t<double> binding_site_origin,    // (3,) grid origin (x0, y0, z0)
    py::array_t<double> apix,                   // (3,) grid spacing (apix_x, apix_y, apix_z)
    double sigma = 2.0,                         // Width of the Gaussian function
    double cutoff_distance = 6.0                // Cutoff for considering atom influence (e.g., 3-4 * sigma)
) {
    // Access NumPy arrays without checking bounds in the inner loop for performance.
    auto pos = protein_positions.unchecked<2>();      // shape: (N, 3)
    auto hpi = protein_hpi.unchecked<1>();            // shape: (N,)
    auto map = binding_site_map.mutable_unchecked<3>(); // shape: (nz, ny, nx)
    auto origin = binding_site_origin.unchecked<1>(); // shape: (3,)
    auto apix_arr = apix.unchecked<1>();              // shape: (3,)

    // Get dimensions and parameters from input arrays.
    ssize_t N  = protein_positions.shape(0);
    ssize_t nz = binding_site_map.shape(0);
    ssize_t ny = binding_site_map.shape(1);
    ssize_t nx = binding_site_map.shape(2);

    double x0 = origin(0);
    double y0 = origin(1);
    double z0 = origin(2);
    double apix_x = apix_arr(0);
    double apix_y = apix_arr(1);
    double apix_z = apix_arr(2);

    // Pre-calculate squared values for efficiency.
    double cutoff_sq = cutoff_distance * cutoff_distance;
    double sigma_sq = sigma * sigma;

    // Use OpenMP for parallel processing across the grid.
    // The collapse(3) clause parallelizes all three loops.
    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (ssize_t iz = 0; iz < nz; ++iz) {
        for (ssize_t iy = 0; iy < ny; ++iy) {
            for (ssize_t ix = 0; ix < nx; ++ix) {
                // Calculate the real-space coordinates of the current grid point.
                double x = x0 + ix * apix_x;
                double y = y0 + iy * apix_y;
                double z = z0 + iz * apix_z;
                
                double grid_value = 0.0;

                // Sum the influence of all protein atoms on this grid point.
                for (ssize_t i = 0; i < N; ++i) {
                    double dx = pos(i, 0) - x;
                    double dy = pos(i, 1) - y;
                    double dz = pos(i, 2) - z;
                    double dist_sq = dx * dx + dy * dy + dz * dz;

                    // Optimization: only calculate the Gaussian if the atom is within the cutoff.
                    // The Gaussian value is negligible beyond this distance anyway.
                    if (dist_sq <= cutoff_sq) {
                        
                        // scaled by a Gaussian function of the distance.
                        double weight = exp(-dist_sq / sigma_sq);
                        grid_value += hpi(i) * weight;
                    }
                }
                map(iz, iy, ix) = grid_value;
            }
        }
    }
    return binding_site_map;
    }

//---------------------------------------------------------------------
// Function 2: Compute Hydrophobic Grid
//---------------------------------------------------------------------
// Computes a 3D hydrophobic grid by summing hydrophobicity values 
// for atoms within a specified cutoff.
py::array_t<double> compute_hydrophobic_grid(
    py::array_t<double> protein_positions,   // (N, 3)
    py::array_t<double> protein_hpi,           // (N,)
    py::array_t<double> binding_site_map,      // (nz, ny, nx) [output buffer]
    py::array_t<double> binding_site_origin,   // (3,) origin: (x0, y0, z0)
    py::array_t<double> apix,                  // (3,) grid spacing: (apix_x, apix_y, apix_z)
    double cutoff_distance = 6.0
) {
    auto pos = protein_positions.unchecked<2>();      // shape: (N, 3)
    auto hpi = protein_hpi.unchecked<1>();              // shape: (N,)
    auto map = binding_site_map.mutable_unchecked<3>(); // shape: (nz, ny, nx)
    auto origin = binding_site_origin.unchecked<1>();   // (3,)
    auto apix_arr = apix.unchecked<1>();                // (3,)

    ssize_t N  = protein_positions.shape(0);
    ssize_t nz = binding_site_map.shape(0);
    ssize_t ny = binding_site_map.shape(1);
    ssize_t nx = binding_site_map.shape(2);

    double x0 = origin(0);
    double y0 = origin(1);
    double z0 = origin(2);
    double apix_x = apix_arr(0);
    double apix_y = apix_arr(1);
    double apix_z = apix_arr(2);

    double cutoff_sq = cutoff_distance * cutoff_distance;

    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (ssize_t iz = 0; iz < nz; ++iz) {
        for (ssize_t iy = 0; iy < ny; ++iy) {
            for (ssize_t ix = 0; ix < nx; ++ix) {
                double z = z0 + iz * apix_z;
                double y = y0 + iy * apix_y;
                double x = x0 + ix * apix_x;
                double grid_value = 0.0;
                for (ssize_t i = 0; i < N; ++i) {
                    double dx = pos(i, 0) - x;
                    double dy = pos(i, 1) - y;
                    double dz = pos(i, 2) - z;
                    double dist_sq = dx * dx + dy * dy + dz * dz;
                    if (dist_sq <= cutoff_sq) {
                        grid_value += hpi(i);
                    }
                }
                map(iz, iy, ix) = grid_value;
            }
        }
    }
    return binding_site_map;
}

//---------------------------------------------------------------------
// Function 3: Compute Electrostatic Grid (No Cutoff)
//---------------------------------------------------------------------
// Computes a 3D electrostatic potential grid without any distance cutoff,
// summing contributions from all atoms.
py::array_t<double> compute_electostatic_grid_no_cutoff(
    py::array_t<double> protein_positions,   // (N, 3)
    py::array_t<double> protein_charges,       // (N,)
    py::array_t<double> binding_site_map,      // (nz, ny, nx) [output buffer]
    py::array_t<double> binding_site_origin,   // (3,) origin: (x0, y0, z0)
    py::array_t<double> apix,                  // (3,) grid spacing: (apix_x, apix_y, apix_z)
    double c_factor = 332.06,
    double eps0 = 4.0,
    double k = 0.2,
    double min_r = 0.5
) {
    auto pos    = protein_positions.unchecked<2>();      // shape: (N, 3)
    auto charges= protein_charges.unchecked<1>();          // shape: (N,)
    auto map    = binding_site_map.mutable_unchecked<3>(); // shape: (nz, ny, nx)
    auto origin = binding_site_origin.unchecked<1>();      // (3,)
    auto apix_arr = apix.unchecked<1>();                   // (3,)

    ssize_t N  = protein_positions.shape(0);
    ssize_t nz = binding_site_map.shape(0);
    ssize_t ny = binding_site_map.shape(1);
    ssize_t nx = binding_site_map.shape(2);

    double x0 = origin(0);
    double y0 = origin(1);
    double z0 = origin(2);
    double apix_x = apix_arr(0);
    double apix_y = apix_arr(1);
    double apix_z = apix_arr(2);

    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (ssize_t iz = 0; iz < nz; ++iz) {
        
        for (ssize_t iy = 0; iy < ny; ++iy) {
            for (ssize_t ix = 0; ix < nx; ++ix) {
                double z = z0 + iz * apix_z;
                double y = y0 + iy * apix_y;
                double x = x0 + ix * apix_x;
                double sum_q_over_epsr_r = 0.0;
                for (ssize_t i = 0; i < N; ++i) {
                    double dx = pos(i, 0) - x;
                    double dy = pos(i, 1) - y;
                    double dz = pos(i, 2) - z;
                    double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (r < min_r)
                        r = min_r;
                    double eps_r = sigmoidal_dielectric(r);
                    //double eps_r = distance_dependent_dielectric(r, eps0, k);
                    sum_q_over_epsr_r += charges(i) / (eps_r * r);
                }
                map(iz, iy, ix) = c_factor * sum_q_over_epsr_r;
            }
        }
    }
    return binding_site_map;
}


//---------------------------------------------------------------------
// Function 4: Compute Enclosure Grid
//---------------------------------------------------------------------
py::array_t<double> compute_enclosure_grid(
    py::array_t<double> protein_positions,      // (N, 3) atom coordinates
    py::array_t<double> protein_hpi,            // (N,) atom hydrophobicity (value > 0 is hydrophobic)
    py::array_t<double> enclosure_map,          // (nz, ny, nx) [output buffer]
    py::array_t<double> binding_site_origin,    // (3,) grid origin (x0, y0, z0)
    py::array_t<double> apix,                   // (3,) grid spacing (apix_x, apix_y, apix_z)
    py::array_t<double> probe_vectors,          // (M, 3) normalized direction vectors
    double ray_cutoff = 5.0,                    // Max distance to search along a ray
    double axis_tolerance = 1.0                 // Max perpendicular distance from atom to ray
) {
    auto pos = protein_positions.unchecked<2>();
    auto hpi = protein_hpi.unchecked<1>();
    auto map = enclosure_map.mutable_unchecked<3>();
    auto origin = binding_site_origin.unchecked<1>();
    auto apix_arr = apix.unchecked<1>();
    auto probe_vecs = probe_vectors.unchecked<2>();

    ssize_t N = protein_positions.shape(0);
    ssize_t nz = enclosure_map.shape(0);
    ssize_t ny = enclosure_map.shape(1);
    ssize_t nx = enclosure_map.shape(2);
    ssize_t M = probe_vectors.shape(0);

    double x0 = origin(0), y0 = origin(1), z0 = origin(2);
    double apix_x = apix_arr(0), apix_y = apix_arr(1), apix_z = apix_arr(2);

    double axis_tolerance_sq = axis_tolerance * axis_tolerance;
    const double infinity = std::numeric_limits<double>::infinity();

    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (ssize_t iz = 0; iz < nz; ++iz) {
        for (ssize_t iy = 0; iy < ny; ++iy) {
            for (ssize_t ix = 0; ix < nx; ++ix) {
                double Px = x0 + ix * apix_x;
                double Py = y0 + iy * apix_y;
                double Pz = z0 + iz * apix_z;
                
                int enclosure_score = 0;

                // Iterate through half the vectors; use each vector v and its opposite -v
                for (ssize_t j = 0; j < M / 2; ++j) {
                    double vx = probe_vecs(j, 0);
                    double vy = probe_vecs(j, 1);
                    double vz = probe_vecs(j, 2);

                    double min_dist_pos = infinity;
                    double min_dist_neg = infinity;

                    for (ssize_t i = 0; i < N; ++i) {
                        if (hpi(i) <= 0) continue; // Skip non-hydrophobic atoms

                        double PAx = pos(i, 0) - Px;
                        double PAy = pos(i, 1) - Py;
                        double PAz = pos(i, 2) - Pz;
                        double dist_sq = PAx * PAx + PAy * PAy + PAz * PAz;

                        // Project vector PA onto probe vector v
                        double proj_dist = PAx * vx + PAy * vy + PAz * vz;

                        // Calculate perpendicular distance squared
                        double perp_dist_sq = dist_sq - proj_dist * proj_dist;

                        if (perp_dist_sq < axis_tolerance_sq) {
                            if (proj_dist > 0 && proj_dist < min_dist_pos) {
                                min_dist_pos = proj_dist;
                            } else if (proj_dist < 0) {
                                double neg_dist = -proj_dist;
                                if (neg_dist < min_dist_neg) {
                                    min_dist_neg = neg_dist;
                                }
                            }
                        }
                    } // End atom loop

                    // Check for an opposing hit within the ray cutoff
                    if (min_dist_pos < ray_cutoff && min_dist_neg < ray_cutoff) {
                        enclosure_score++;
                    }
                } // End probe vector loop
                map(iz, iy, ix) = static_cast<double>(enclosure_score);
            }
        }
    }
    return enclosure_map;
}




// ---------------------------------------------------------------
// Helpers for water map
// ---------------------------------------------------------------
inline void clamp3(int &gx, int &gy, int &gz, int nx, int ny, int nz) {
    if (gx < 0) gx = 0; if (gx >= nx) gx = nx - 1;
    if (gy < 0) gy = 0; if (gy >= ny) gy = ny - 1;
    if (gz < 0) gz = 0; if (gz >= nz) gz = nz - 1;
}

inline double sq(double x){ return x*x; }

// Brute clearance check (N could be large; you can KD-tree later if needed)
inline bool has_clearance(double x, double y, double z,
                          const double* prot, ssize_t N,
                          double min_clear_sq)
{
    if (N <= 0) return true; // no protein array provided => skip check
    for (ssize_t i=0; i<N; ++i) {
        double dx = prot[3*i+0] - x;
        double dy = prot[3*i+1] - y;
        double dz = prot[3*i+2] - z;
        if (dx*dx + dy*dy + dz*dz < min_clear_sq) return false;
    }
    return true;
}

// ---------------------------------------------------------------
// Water acceptor site map (protein DONORS → water accepts)
// We mark all voxels whose distance to a donor-H position is in [r_opt - dr, r_opt + dr].
// ---------------------------------------------------------------
py::array_t<double> compute_water_acceptor_site_map(
    py::array_t<double> donor_H_positions,      // (M,3) H coordinates for protein donors
    py::array_t<double> binding_site_map,       // (nz,ny,nx) [output buffer, will be += weight]
    py::array_t<double> binding_site_origin,    // (3,) origin: (x0,y0,z0)
    py::array_t<double> apix,                   // (3,) spacing: (dx,dy,dz)
    py::array_t<double> protein_positions,      // (N,3) (optional: pass empty (0,3) to skip clearance)
    double r_opt = 2.9,                         // target O···H distance
    double dr = 0.6,                            // shell half-width (mark within [r_opt-dr, r_opt+dr])
    double min_protein_clearance = 1.8,         // minimum O distance to any protein atom (Å)
    double weight = 1.0                         // value to add to voxels that qualify
) {
    auto H = donor_H_positions.unchecked<2>();         // (M,3)
    auto map = binding_site_map.mutable_unchecked<3>(); // (nz,ny,nx)
    auto org = binding_site_origin.unchecked<1>();     // (3,)
    auto sp  = apix.unchecked<1>();                    // (3,)
    auto P   = protein_positions.unchecked<2>();       // (N,3)

    const ssize_t M  = donor_H_positions.shape(0);
    const ssize_t nz = binding_site_map.shape(0);
    const ssize_t ny = binding_site_map.shape(1);
    const ssize_t nx = binding_site_map.shape(2);
    const ssize_t N  = protein_positions.shape(0);

    const double x0 = org(0), y0 = org(1), z0 = org(2);
    const double dx = sp(0),  dy = sp(1),  dz = sp(2);

    const double rmin = std::max(0.0, r_opt - dr);
    const double rmax = r_opt + dr;
    const double rmin_sq = rmin*rmin;
    const double rmax_sq = rmax*rmax;
    const double min_clear_sq = min_protein_clearance * min_protein_clearance;

    // For each donor-H, we only iterate over a tight voxel box enclosing the spherical shell.
    #pragma omp parallel for schedule(dynamic)
    for (ssize_t m=0; m<M; ++m) {
        const double Hx = H(m,0), Hy = H(m,1), Hz = H(m,2);

        int gx_min = (int)std::floor((Hx - rmax - x0)/dx);
        int gx_max = (int)std::ceil ((Hx + rmax - x0)/dx);
        int gy_min = (int)std::floor((Hy - rmax - y0)/dy);
        int gy_max = (int)std::ceil ((Hy + rmax - y0)/dy);
        int gz_min = (int)std::floor((Hz - rmax - z0)/dz);
        int gz_max = (int)std::ceil ((Hz + rmax - z0)/dz);
        clamp3(gx_min, gy_min, gz_min, nx, ny, nz);
        clamp3(gx_max, gy_max, gz_max, nx, ny, nz);

        for (int gz=gz_min; gz<=gz_max; ++gz) {
            const double z = z0 + gz*dz;
            for (int gy=gy_min; gy<=gy_max; ++gy) {
                const double y = y0 + gy*dy;
                for (int gx=gx_min; gx<=gx_max; ++gx) {
                    const double x = x0 + gx*dx;
                    const double d2 = sq(Hx-x)+sq(Hy-y)+sq(Hz-z);
                    if (d2 < rmin_sq || d2 > rmax_sq) continue;
                    if (!has_clearance(x,y,z, &P(0,0), N, min_clear_sq)) continue;
                    map(gz,gy,gx) += weight;
                }
            }
        }
    }
    return binding_site_map;
}

// ---------------------------------------------------------------
// Water donor site map (protein ACCEPTORS → water donates)
// Mark voxels on a spherical shell around the acceptor heavy atom.
// ---------------------------------------------------------------
py::array_t<double> compute_water_donor_site_map(
    py::array_t<double> acceptor_positions,     // (K,3) coordinates of protein acceptor heavy atoms
    py::array_t<double> binding_site_map,       // (nz,ny,nx) [output buffer, will be += weight]
    py::array_t<double> binding_site_origin,    // (3,) origin: (x0,y0,z0)
    py::array_t<double> apix,                   // (3,) spacing: (dx,dy,dz)
    py::array_t<double> protein_positions,      // (N,3) (optional: pass empty (0,3) to skip clearance)
    double r_opt = 2.9,                         // target O···A distance
    double dr = 0.6,                            // shell half-width
    double min_protein_clearance = 1.8,         // minimum O distance to any protein atom
    double weight = 1.0                         // value to add to voxels that qualify
) {
    auto A = acceptor_positions.unchecked<2>();         // (K,3)
    auto map = binding_site_map.mutable_unchecked<3>(); // (nz,ny,nx)
    auto org = binding_site_origin.unchecked<1>();      // (3,)
    auto sp  = apix.unchecked<1>();                     // (3,)
    auto P   = protein_positions.unchecked<2>();        // (N,3)

    const ssize_t K  = acceptor_positions.shape(0);
    const ssize_t nz = binding_site_map.shape(0);
    const ssize_t ny = binding_site_map.shape(1);
    const ssize_t nx = binding_site_map.shape(2);
    const ssize_t N  = protein_positions.shape(0);

    const double x0 = org(0), y0 = org(1), z0 = org(2);
    const double dx = sp(0),  dy = sp(1),  dz = sp(2);

    const double rmin = std::max(0.0, r_opt - dr);
    const double rmax = r_opt + dr;
    const double rmin_sq = rmin*rmin;
    const double rmax_sq = rmax*rmax;
    const double min_clear_sq = min_protein_clearance * min_protein_clearance;

    #pragma omp parallel for schedule(dynamic)
    for (ssize_t k=0; k<K; ++k) {
        const double Ax = A(k,0), Ay = A(k,1), Az = A(k,2);

        int gx_min = (int)std::floor((Ax - rmax - x0)/dx);
        int gx_max = (int)std::ceil ((Ax + rmax - x0)/dx);
        int gy_min = (int)std::floor((Ay - rmax - y0)/dy);
        int gy_max = (int)std::ceil ((Ay + rmax - y0)/dy);
        int gz_min = (int)std::floor((Az - rmax - z0)/dz);
        int gz_max = (int)std::ceil ((Az + rmax - z0)/dz);
        clamp3(gx_min, gy_min, gz_min, nx, ny, nz);
        clamp3(gx_max, gy_max, gz_max, nx, ny, nz);

        for (int gz=gz_min; gz<=gz_max; ++gz) {
            const double z = z0 + gz*dz;
            for (int gy=gy_min; gy<=gy_max; ++gy) {
                const double y = y0 + gy*dy;
                for (int gx=gx_min; gx<=gx_max; ++gx) {
                    const double x = x0 + gx*dx;
                    const double d2 = sq(Ax-x)+sq(Ay-y)+sq(Az-z);
                    if (d2 < rmin_sq || d2 > rmax_sq) continue;
                    if (!has_clearance(x,y,z, &P(0,0), N, min_clear_sq)) continue;
                    map(gz,gy,gx) += weight;
                }
            }
        }
    }
    return binding_site_map;
}

// ---------------------------------------------------------------
// Convenience: fill BOTH maps (acceptor and donor) in one call.
// This takes donor H positions and acceptor atom positions separately.
// ---------------------------------------------------------------
py::tuple compute_water_site_maps(
    py::array_t<double> donor_H_positions,      // (M,3)
    py::array_t<double> acceptor_positions,     // (K,3)
    py::array_t<double> binding_site_origin,    // (3,)
    py::array_t<double> apix,                   // (3,)
    py::tuple          map_shapes,              // ((nz,ny,nx), (nz,ny,nx)) for the two outputs
    py::array_t<double> protein_positions,      // (N,3) optional for clearance
    double r_opt = 2.9,
    double dr = 0.6,
    double min_protein_clearance = 1.8,
    double weight_acceptor_map = 1.0,
    double weight_donor_map    = 1.0
) {
    // Allocate output buffers
    auto shapeA = map_shapes[0].cast<py::tuple>();
    auto shapeD = map_shapes[1].cast<py::tuple>();
    const ssize_t nzA = shapeA[0].cast<ssize_t>();
    const ssize_t nyA = shapeA[1].cast<ssize_t>();
    const ssize_t nxA = shapeA[2].cast<ssize_t>();
    const ssize_t nzD = shapeD[0].cast<ssize_t>();
    const ssize_t nyD = shapeD[1].cast<ssize_t>();
    const ssize_t nxD = shapeD[2].cast<ssize_t>();

    py::array_t<double> A_map({nzA, nyA, nxA});
    py::array_t<double> D_map({nzD, nyD, nxD});

    // Zero-initialize
    std::fill_n(A_map.mutable_data(), nzA*nyA*nxA, 0.0);
    std::fill_n(D_map.mutable_data(), nzD*nyD*nxD, 0.0);

    // Fill maps
    compute_water_acceptor_site_map(
        donor_H_positions, A_map, binding_site_origin, apix, protein_positions,
        r_opt, dr, min_protein_clearance, weight_acceptor_map
    );
    compute_water_donor_site_map(
        acceptor_positions, D_map, binding_site_origin, apix, protein_positions,
        r_opt, dr, min_protein_clearance, weight_donor_map
    );
    return py::make_tuple(A_map, D_map);
}

//---------------------------------------------------------------------
//grid maps echo v2
//---------------------------------------------------------------------




// C++ version of make_protein_and_solvent_masks (one call from Python)
py::tuple make_protein_and_solvent_masks_cpp(
    py::array_t<double> atom_coords,   // (N, 3)
    py::array_t<double> atom_radii,    // (N,)
    py::array_t<double> origin,        // (3,)
    py::tuple grid_shape,              // (nz, ny, nx)
    double spacing,
    double probe_radius = 1.4
) {
    // --- Basic input checks ---
    if (atom_coords.ndim() != 2 || atom_coords.shape(1) != 3) {
        throw std::runtime_error("atom_coords must be shape (N, 3)");
    }
    if (atom_radii.ndim() != 1 || atom_radii.shape(0) != atom_coords.shape(0)) {
        throw std::runtime_error("atom_radii must be shape (N,) matching atom_coords");
    }
    if (origin.ndim() != 1 || origin.shape(0) != 3) {
        throw std::runtime_error("origin must be shape (3,)");
    }
    if (grid_shape.size() != 3) {
        throw std::runtime_error("grid_shape must be a tuple/list of length 3 (nz, ny, nx)");
    }

    const auto nz = grid_shape[0].cast<py::ssize_t>();
    const auto ny = grid_shape[1].cast<py::ssize_t>();
    const auto nx = grid_shape[2].cast<py::ssize_t>();

    if (nz <= 0 || ny <= 0 || nx <= 0) {
        throw std::runtime_error("grid dimensions must be positive");
    }

    auto ac = atom_coords.unchecked<2>();
    auto ar = atom_radii.unchecked<1>();
    auto o  = origin.unchecked<1>();

    const py::ssize_t n_atoms = ac.shape(0);

    // --- Allocate and zero the temporary grid (double precision) ---
    py::array_t<double> grid({nz, ny, nx});
    {
        py::buffer_info buf = grid.request();
        auto* ptr = static_cast<double*>(buf.ptr);
        std::fill(ptr, ptr + (nz * ny * nx), 0.0);
    }

    auto g = grid.mutable_unchecked<3>();

    const double inv_spacing = 1.0 / spacing;

    // --- Paint spheres into the grid ---
    //#pragma omp parallel for
    for (py::ssize_t i = 0; i < n_atoms; ++i) {
        const double cx = ac(i, 0);
        const double cy = ac(i, 1);
        const double cz = ac(i, 2);

        const double rad = ar(i) + probe_radius;
        const double r2  = rad * rad;

        // Bounding box in index space (X, Y, Z)
        int ix_min = static_cast<int>(std::floor((cx - rad - o(0)) * inv_spacing));
        int ix_max = static_cast<int>(std::floor((cx + rad - o(0)) * inv_spacing));
        int iy_min = static_cast<int>(std::floor((cy - rad - o(1)) * inv_spacing));
        int iy_max = static_cast<int>(std::floor((cy + rad - o(1)) * inv_spacing));
        int iz_min = static_cast<int>(std::floor((cz - rad - o(2)) * inv_spacing));
        int iz_max = static_cast<int>(std::floor((cz + rad - o(2)) * inv_spacing));

        // Clip to grid bounds
        ix_min = std::max(ix_min, 0);
        iy_min = std::max(iy_min, 0);
        iz_min = std::max(iz_min, 0);
        ix_max = std::min(ix_max, static_cast<int>(nx) - 1);
        iy_max = std::min(iy_max, static_cast<int>(ny) - 1);
        iz_max = std::min(iz_max, static_cast<int>(nz) - 1);

        // Skip if sphere doesn't intersect the grid
        if (ix_min > ix_max || iy_min > iy_max || iz_min > iz_max) {
            continue;
        }

        // Loop over the bounding box and mark inside points
        for (int iz = iz_min; iz <= iz_max; ++iz) {
            const double z = (o(2) + static_cast<double>(iz) * spacing) - cz;
            const double dz2 = z * z;

            for (int iy = iy_min; iy <= iy_max; ++iy) {
                const double y = (o(1) + static_cast<double>(iy) * spacing) - cy;
                const double dy2 = y * y;

                for (int ix = ix_min; ix <= ix_max; ++ix) {
                    const double x = (o(0) + static_cast<double>(ix) * spacing) - cx;
                    const double dx2 = x * x;

                    const double dist2 = dx2 + dy2 + dz2;
                    if (dist2 <= r2) {
                        g(iz, iy, ix) = 1.0;
                    }
                }
            }
        }
    }

    // --- Build protein_mask and solvent_mask as uint8 ---
    py::array_t<std::uint8_t> protein_mask({nz, ny, nx});
    py::array_t<std::uint8_t> solvent_mask({nz, ny, nx});

    auto pm = protein_mask.mutable_unchecked<3>();
    auto sm = solvent_mask.mutable_unchecked<3>();

    for (py::ssize_t iz = 0; iz < nz; ++iz) {
        for (py::ssize_t iy = 0; iy < ny; ++iy) {
            for (py::ssize_t ix = 0; ix < nx; ++ix) {
                const bool inside = (g(iz, iy, ix) > 0.5);
                const std::uint8_t pval = inside ? 1 : 0;
                pm(iz, iy, ix) = pval;
                sm(iz, iy, ix) = static_cast<std::uint8_t>(1 - pval);
            }
        }
    }

    return py::make_tuple(protein_mask, solvent_mask);
}

//---------------------------------------------------------------------
//depth map propergation 
//---------------------------------------------------------------------

// bulk_mask, protein_mask, target_mask are uint8 (0/1) arrays, shape (nz,ny,nx)
// spacing in Å, connectivity = 6 or 26
// Returns float32 depth (Å), np.inf for unreachable.
py::array_t<float> depth_propagation_cpp(
    py::array_t<std::uint8_t> bulk_mask,
    py::array_t<std::uint8_t> protein_mask,
    py::array_t<std::uint8_t> target_mask,
    double spacing = 1.0,
    int connectivity = 6
) {
    if (bulk_mask.ndim() != 3 || protein_mask.ndim() != 3 || target_mask.ndim() != 3) {
        throw std::runtime_error("All masks must be 3D (nz, ny, nx).");
    }

    auto b = bulk_mask.unchecked<3>();
    auto p = protein_mask.unchecked<3>();
    auto t = target_mask.unchecked<3>();

    const py::ssize_t nz = b.shape(0);
    const py::ssize_t ny = b.shape(1);
    const py::ssize_t nx = b.shape(2);

    if (p.shape(0) != nz || p.shape(1) != ny || p.shape(2) != nx ||
        t.shape(0) != nz || t.shape(1) != ny || t.shape(2) != nx) {
        throw std::runtime_error("All masks must have the same shape.");
    }

    // Depth array: initialize to +inf
    py::array_t<float> depth({nz, ny, nx});
    auto d = depth.mutable_unchecked<3>();
    const float INF = std::numeric_limits<float>::infinity();

    for (py::ssize_t z = 0; z < nz; ++z) {
        for (py::ssize_t y = 0; y < ny; ++y) {
            for (py::ssize_t x = 0; x < nx; ++x) {
                d(z, y, x) = INF;
            }
        }
    }

    struct Voxel {
        int z, y, x;
    };

    // Build neighbor list for chosen connectivity
    std::vector<Voxel> neighbors;
    neighbors.reserve(26);
    if (connectivity == 6) {
        neighbors.push_back({ 1,  0,  0});
        neighbors.push_back({-1,  0,  0});
        neighbors.push_back({ 0,  1,  0});
        neighbors.push_back({ 0, -1,  0});
        neighbors.push_back({ 0,  0,  1});
        neighbors.push_back({ 0,  0, -1});
    } else {
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dz == 0 && dy == 0 && dx == 0) continue;
                    neighbors.push_back({dz, dy, dx});
                }
            }
        }
    }

    // Multi-source BFS queue
    std::vector<Voxel> queue;
    queue.reserve(static_cast<std::size_t>(nz * ny * nx / 16 + 1));
    std::size_t head = 0;

    // Seed: bulk voxels in allowed region (not protein)
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const bool is_bulk    = (b(z, y, x) != 0);
                const bool is_protein = (p(z, y, x) != 0);
                if (is_bulk && !is_protein) {
                    d(z, y, x) = 0.0f;
                    queue.push_back({z, y, x});
                }
            }
        }
    }

    // Count how many target voxels still need a depth
    std::int64_t remaining = 0;
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                if (t(z, y, x) != 0 && std::isinf(d(z, y, x))) {
                    remaining++;
                }
            }
        }
    }

    if (remaining == 0 || queue.empty()) {
        // No targets or no sources: depths stay inf, which matches the Python semantics.
        return depth;
    }

    // BFS: shortest-path distance in steps, then * spacing
    const float step_len = static_cast<float>(spacing);

    while (head < queue.size() && remaining > 0) {
        Voxel cur = queue[head++];
        const int zc = cur.z;
        const int yc = cur.y;
        const int xc = cur.x;
        const float cur_depth = d(zc, yc, xc);

        for (const auto &nb : neighbors) {
            const int zn = zc + nb.z;
            const int yn = yc + nb.y;
            const int xn = xc + nb.x;

            if (zn < 0 || zn >= nz || yn < 0 || yn >= ny || xn < 0 || xn >= nx) {
                continue;
            }

            // Cannot cross protein
            if (p(zn, yn, xn) != 0) {
                continue;
            }

            // Only visit if not yet assigned
            if (!std::isinf(d(zn, yn, xn))) {
                continue;
            }

            const float new_depth = cur_depth + step_len;
            d(zn, yn, xn) = new_depth;
            queue.push_back({zn, yn, xn});

            if (t(zn, yn, xn) != 0) {
                remaining--;
                if (remaining <= 0) break;
            }
        }
    }

    return depth;
}

// -----------------------------------------------------------------------------
// C++ version of prepare_depth_for_mrc
// -----------------------------------------------------------------------------
py::array_t<float> prepare_depth_for_mrc_cpp(
    py::array_t<float> depth_map,
    py::object sasa_mask_obj = py::none(),
    py::object fill_unreachable_obj = py::none()
) {
    if (depth_map.ndim() != 3) {
        throw std::runtime_error("depth_map must be 3D (nz, ny, nx).");
    }

    auto dm = depth_map.unchecked<3>();
    const py::ssize_t nz = dm.shape(0);
    const py::ssize_t ny = dm.shape(1);
    const py::ssize_t nx = dm.shape(2);

    // Copy into a new array
    py::array_t<float> depth({nz, ny, nx});
    auto d = depth.mutable_unchecked<3>();

    for (py::ssize_t z = 0; z < nz; ++z) {
        for (py::ssize_t y = 0; y < ny; ++y) {
            for (py::ssize_t x = 0; x < nx; ++x) {
                d(z, y, x) = dm(z, y, x);
            }
        }
    }

    bool have_sasa = !sasa_mask_obj.is_none();
    py::array_t<std::uint8_t> sasa_mask_arr;
    if (have_sasa) {
        sasa_mask_arr = sasa_mask_obj.cast<py::array_t<std::uint8_t>>();
        if (sasa_mask_arr.ndim() != 3 ||
            sasa_mask_arr.shape(0) != nz ||
            sasa_mask_arr.shape(1) != ny ||
            sasa_mask_arr.shape(2) != nx) {
            throw std::runtime_error("sasa_mask must be same shape as depth_map.");
        }
        auto s = sasa_mask_arr.unchecked<3>();
        // Outside SASA -> 0.0
        for (py::ssize_t z = 0; z < nz; ++z) {
            for (py::ssize_t y = 0; y < ny; ++y) {
                for (py::ssize_t x = 0; x < nx; ++x) {
                    if (!static_cast<bool>(s(z, y, x))) {
                        d(z, y, x) = 0.0f;
                    }
                }
            }
        }
    }

    // Find max finite depth
    bool has_finite = false;
    float max_depth = -std::numeric_limits<float>::infinity();
    for (py::ssize_t z = 0; z < nz; ++z) {
        for (py::ssize_t y = 0; y < ny; ++y) {
            for (py::ssize_t x = 0; x < nx; ++x) {
                float v = d(z, y, x);
                if (std::isfinite(v)) {
                    has_finite = true;
                    if (v > max_depth) {
                        max_depth = v;
                    }
                }
            }
        }
    }

    if (!has_finite) {
        // Degenerate case: everything inf/nan -> return all zeros
        py::array_t<float> out({nz, ny, nx});
        auto o = out.mutable_unchecked<3>();
        for (py::ssize_t z = 0; z < nz; ++z) {
            for (py::ssize_t y = 0; y < ny; ++y) {
                for (py::ssize_t x = 0; x < nx; ++x) {
                    o(z, y, x) = 0.0f;
                }
            }
        }
        return out;
    }

    // Decide fill_unreachable
    float fill_unreachable;
    if (fill_unreachable_obj.is_none()) {
        fill_unreachable = max_depth;
    } else {
        fill_unreachable = static_cast<float>(fill_unreachable_obj.cast<double>());
    }

    // Replace inf/nan with chosen value
    for (py::ssize_t z = 0; z < nz; ++z) {
        for (py::ssize_t y = 0; y < ny; ++y) {
            for (py::ssize_t x = 0; x < nx; ++x) {
                float &v = d(z, y, x);
                if (!std::isfinite(v)) {
                    v = fill_unreachable;
                }
            }
        }
    }

    return depth;
}

// -----------------------------------------------------------------------------
// widest path cpp
// -----------------------------------------------------------------------------



// Node in the max-heap
struct Node {
    float radius;
    int z, y, x;
};

struct NodeCompare {
    bool operator()(const Node &a, const Node &b) const {
        // priority_queue puts "largest" first, so we want smaller radius to be "worse"
        return a.radius < b.radius;
    }
};

// C++ widest-path propagation
// r_local      : float32 (nz,ny,nx)
// solvent_mask : uint8   (nz,ny,nx) (0/1)
// seed_mask    : uint8   (nz,ny,nx) (0/1)
// connectivity : 6 or 26
// power        : >=1 (for flow_norm = R_bottle^power / max)
py::tuple widest_path_cpp(
    py::array_t<float> r_local,
    py::array_t<std::uint8_t> solvent_mask,
    py::array_t<std::uint8_t> seed_mask,
    int connectivity = 6,
    int power = 4
) {
    if (r_local.ndim() != 3 || solvent_mask.ndim() != 3 || seed_mask.ndim() != 3) {
        throw std::runtime_error("All inputs must be 3D arrays (nz, ny, nx).");
    }

    auto r  = r_local.unchecked<3>();
    auto sm = solvent_mask.unchecked<3>();
    auto se = seed_mask.unchecked<3>();

    const py::ssize_t nz = r.shape(0);
    const py::ssize_t ny = r.shape(1);
    const py::ssize_t nx = r.shape(2);

    if (sm.shape(0) != nz || sm.shape(1) != ny || sm.shape(2) != nx ||
        se.shape(0) != nz || se.shape(1) != ny || se.shape(2) != nx) {
        throw std::runtime_error("All inputs must have the same shape.");
    }

    // Output: R_bottle
    py::array_t<float> R_bottle({nz, ny, nx});
    auto Rb = R_bottle.mutable_unchecked<3>();
    for (py::ssize_t z = 0; z < nz; ++z)
        for (py::ssize_t y = 0; y < ny; ++y)
            for (py::ssize_t x = 0; x < nx; ++x)
                Rb(z, y, x) = 0.0f;

    // Neighbor offsets
    std::vector<std::array<int,3>> neighbors;
    if (connectivity == 6) {
        neighbors = {
            { 1,  0,  0}, {-1,  0,  0},
            { 0,  1,  0}, { 0, -1,  0},
            { 0,  0,  1}, { 0,  0, -1},
        };
    } else if (connectivity == 26) {
        neighbors.reserve(26);
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dz == 0 && dy == 0 && dx == 0) continue;
                    neighbors.push_back({dz, dy, dx});
                }
            }
        }
    } else {
        throw std::runtime_error("connectivity must be 6 or 26");
    }

    // Max-heap for widest-path
    std::priority_queue<Node, std::vector<Node>, NodeCompare> pq;

    constexpr float eps = 1e-8f;

    // Seed queue with interface seeds
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                if (se(z, y, x) == 0) continue;
                if (sm(z, y, x) == 0) continue;  // should be in solvent
                float radius_here = r(z, y, x);
                if (radius_here <= 0.0f) continue;

                Rb(z, y, x) = radius_here;
                pq.push(Node{radius_here, z, y, x});
            }
        }
    }

    if (pq.empty()) {
        // R_bottle all zeros, flow_norm all zeros
        py::array_t<float> flow_norm({nz, ny, nx});
        auto fn = flow_norm.mutable_unchecked<3>();
        for (py::ssize_t z = 0; z < nz; ++z)
            for (py::ssize_t y = 0; y < ny; ++y)
                for (py::ssize_t x = 0; x < nx; ++x)
                    fn(z, y, x) = 0.0f;
        return py::make_tuple(R_bottle, flow_norm);
    }

    // Widest-path propagation
    while (!pq.empty()) {
        Node cur = pq.top();
        pq.pop();

        float current_radius = cur.radius;
        int z = cur.z;
        int y = cur.y;
        int x = cur.x;

        // Skip outdated entries
        if (current_radius < Rb(z, y, x) - eps) {
            continue;
        }

        for (const auto &nb : neighbors) {
            int zn = z + nb[0];
            int yn = y + nb[1];
            int xn = x + nb[2];

            if (zn < 0 || zn >= nz || yn < 0 || yn >= ny || xn < 0 || xn >= nx)
                continue;

            if (sm(zn, yn, xn) == 0)  // must stay in solvent
                continue;

            float neighbor_radius = r(zn, yn, xn);
            if (neighbor_radius <= 0.0f)
                continue;

            float candidate = std::min(current_radius, neighbor_radius);
            if (candidate > Rb(zn, yn, xn) + eps) {
                Rb(zn, yn, xn) = candidate;
                pq.push(Node{candidate, zn, yn, xn});
            }
        }
    }

    // Compute flow_norm = (R_bottle^power) / max
    py::array_t<float> flow_norm({nz, ny, nx});
    auto fn = flow_norm.mutable_unchecked<3>();

    float max_flow = 0.0f;
    if (power < 1) power = 1;  // safety

    // First pass: compute R_bottle^power and track max
    for (py::ssize_t z = 0; z < nz; ++z) {
        for (py::ssize_t y = 0; y < ny; ++y) {
            for (py::ssize_t x = 0; x < nx; ++x) {
                float val = Rb(z, y, x);
                float f = 0.0f;
                if (val > 0.0f) {
                    f = std::pow(val, static_cast<float>(power));
                    if (f > max_flow) max_flow = f;
                }
                fn(z, y, x) = f;
            }
        }
    }

    if (max_flow > 0.0f) {
        float inv_max = 1.0f / max_flow;
        for (py::ssize_t z = 0; z < nz; ++z) {
            for (py::ssize_t y = 0; y < ny; ++y) {
                for (py::ssize_t x = 0; x < nx; ++x) {
                    fn(z, y, x) *= inv_max;
                }
            }
        }
    }

    return py::make_tuple(R_bottle, flow_norm);
}

//--------------------------------------------------------------------
//hydrophobic 2 
//--------------------------------------------------------------------




// C++ version of propagate_logp_exp_decay
py::array_t<float> propagate_logp_exp_decay_cpp(
    py::array_t<double> logp_centers,   // (N, 3)
    py::array_t<double> logp_values,    // (N,)
    py::array_t<double> origin,         // (3,)
    py::tuple grid_shape,               // (nz, ny, nx)
    double spacing,
    py::object sasa_mask_obj = py::none(),
    double cutoff = 6.0
) {
    // --- Basic checks ---
    if (logp_centers.ndim() != 2 || logp_centers.shape(1) != 3) {
        throw std::runtime_error("logp_centers must be shape (N, 3)");
    }
    if (logp_values.ndim() != 1 || logp_values.shape(0) != logp_centers.shape(0)) {
        throw std::runtime_error("logp_values must be shape (N,) matching logp_centers");
    }
    if (origin.ndim() != 1 || origin.shape(0) != 3) {
        throw std::runtime_error("origin must be shape (3,)");
    }
    if (grid_shape.size() != 3) {
        throw std::runtime_error("grid_shape must be a tuple/list (nz, ny, nx)");
    }

    const auto nz = grid_shape[0].cast<py::ssize_t>();
    const auto ny = grid_shape[1].cast<py::ssize_t>();
    const auto nx = grid_shape[2].cast<py::ssize_t>();

    auto centers = logp_centers.unchecked<2>();  // (N, 3)
    auto values  = logp_values.unchecked<1>();   // (N,)
    auto org     = origin.unchecked<1>();        // (3,)

    const py::ssize_t n_src = centers.shape(0);

    // --- Allocate and zero field ---
    py::array_t<float> field({nz, ny, nx});
    auto fld = field.mutable_unchecked<3>();
    for (py::ssize_t z = 0; z < nz; ++z)
        for (py::ssize_t y = 0; y < ny; ++y)
            for (py::ssize_t x = 0; x < nx; ++x)
                fld(z, y, x) = 0.0f;

    const double cutoff2 = cutoff * cutoff;

    // --- Loop over logP sources ---
    for (py::ssize_t i = 0; i < n_src; ++i) {
        const double v = values(i);
        if (v == 0.0) continue;

        const double cx = centers(i, 0);
        const double cy = centers(i, 1);
        const double cz = centers(i, 2);

        // Bounding box in index space
        int ix_min = static_cast<int>(std::floor((cx - cutoff - org(0)) / spacing));
        int ix_max = static_cast<int>(std::ceil ((cx + cutoff - org(0)) / spacing));
        int iy_min = static_cast<int>(std::floor((cy - cutoff - org(1)) / spacing));
        int iy_max = static_cast<int>(std::ceil ((cy + cutoff - org(1)) / spacing));
        int iz_min = static_cast<int>(std::floor((cz - cutoff - org(2)) / spacing));
        int iz_max = static_cast<int>(std::ceil ((cz + cutoff - org(2)) / spacing));

        // Clip to grid bounds
        ix_min = std::max(ix_min, 0);
        iy_min = std::max(iy_min, 0);
        iz_min = std::max(iz_min, 0);
        ix_max = std::min(ix_max, static_cast<int>(nx) - 1);
        iy_max = std::min(iy_max, static_cast<int>(ny) - 1);
        iz_max = std::min(iz_max, static_cast<int>(nz) - 1);

        if (ix_min > ix_max || iy_min > iy_max || iz_min > iz_max) {
            continue;
        }

        // Loop over local box and accumulate contributions
        for (int iz = iz_min; iz <= iz_max; ++iz) {
            const double z_coord = org(2) + static_cast<double>(iz) * spacing;
            const double dz = z_coord - cz;
            const double dz2 = dz * dz;

            for (int iy = iy_min; iy <= iy_max; ++iy) {
                const double y_coord = org(1) + static_cast<double>(iy) * spacing;
                const double dy = y_coord - cy;
                const double dy2 = dy * dy;

                for (int ix = ix_min; ix <= ix_max; ++ix) {
                    const double x_coord = org(0) + static_cast<double>(ix) * spacing;
                    const double dx = x_coord - cx;
                    const double dx2 = dx * dx;

                    const double dist2 = dx2 + dy2 + dz2;
                    if (dist2 > cutoff2) {
                        continue;
                    }

                    const double dist = std::sqrt(dist2);
                    const float contrib = static_cast<float>(v * std::exp(-dist));

                    fld(iz, iy, ix) += contrib;
                }
            }
        }
    }

    // --- Optional SASA mask ---
    if (!sasa_mask_obj.is_none()) {
        py::array_t<uint8_t> sasa_mask_arr = sasa_mask_obj.cast<py::array_t<uint8_t>>();
        if (sasa_mask_arr.ndim() != 3 ||
            sasa_mask_arr.shape(0) != nz ||
            sasa_mask_arr.shape(1) != ny ||
            sasa_mask_arr.shape(2) != nx) {
            throw std::runtime_error("sasa_mask must be same shape as grid_shape");
        }
        auto sm = sasa_mask_arr.unchecked<3>();
        for (py::ssize_t z = 0; z < nz; ++z) {
            for (py::ssize_t y = 0; y < ny; ++y) {
                for (py::ssize_t x = 0; x < nx; ++x) {
                    if (!static_cast<bool>(sm(z, y, x))) {
                        fld(z, y, x) = 0.0f;
                    }
                }
            }
        }
    }

    return field;
}

//---------------------------------------------------------------------
//electrostatics with cutoff
//---------------------------------------------------------------------

py::array_t<double> compute_electrostatic_grid_cutoff_cpp(
    py::array_t<double> protein_positions,   // (N, 3)
    py::array_t<double> protein_charges,     // (N,)
    py::tuple            grid_shape,         // (nz, ny, nx)
    py::array_t<double>  origin,             // (3,) (x0,y0,z0)
    py::array_t<double>  apix,               // (3,) (ax,ay,az)
    double c_factor = 332.06,
    double min_r    = 0.5,
    double cutoff   = 12.0,
    double switch_dist = 2.0    // <-- new for smoothed maps in Å
) {
    if (protein_positions.ndim() != 2 || protein_positions.shape(1) != 3) {
        throw std::runtime_error("protein_positions must be (N,3)");
    }
    if (protein_charges.ndim() != 1 || protein_charges.shape(0) != protein_positions.shape(0)) {
        throw std::runtime_error("protein_charges must be (N,) matching positions");
    }
    if (origin.ndim() != 1 || origin.shape(0) != 3) {
        throw std::runtime_error("origin must be (3,)");
    }
    if (apix.ndim() != 1 || apix.shape(0) != 3) {
        throw std::runtime_error("apix must be (3,)");
    }
    if (grid_shape.size() != 3) {
        throw std::runtime_error("grid_shape must be (nz,ny,nx)");
    }

    auto pos  = protein_positions.unchecked<2>();   // (N,3)
    auto q    = protein_charges.unchecked<1>();     // (N,)
    auto org  = origin.unchecked<1>();              // (3,)
    auto a    = apix.unchecked<1>();                // (3,)

    const auto nz = grid_shape[0].cast<py::ssize_t>();
    const auto ny = grid_shape[1].cast<py::ssize_t>();
    const auto nx = grid_shape[2].cast<py::ssize_t>();

    const py::ssize_t N = protein_positions.shape(0);

    const double x0 = org(0);
    const double y0 = org(1);
    const double z0 = org(2);
    const double ax = a(0);
    const double ay = a(1);
    const double az = a(2);
    
    
    // Define the distance where switching starts
    const double r_on = cutoff - switch_dist;
    
    
    const double cutoff2 = cutoff * cutoff;

    // allocate and zero grid
    py::array_t<double> grid({nz, ny, nx});
    auto g = grid.mutable_unchecked<3>();
    for (py::ssize_t iz = 0; iz < nz; ++iz)
        for (py::ssize_t iy = 0; iy < ny; ++iy)
            for (py::ssize_t ix = 0; ix < nx; ++ix)
                g(iz, iy, ix) = 0.0;

    // loop over atoms and deposit to nearby voxels
    for (py::ssize_t i = 0; i < N; ++i) {
        const double qi  = q(i);
        if (qi == 0.0) continue;

        const double cx = pos(i, 0);
        const double cy = pos(i, 1);
        const double cz = pos(i, 2);

        // bounding box in index space
        int ix_min = static_cast<int>(std::floor((cx - cutoff - x0) / ax));
        int ix_max = static_cast<int>(std::ceil ((cx + cutoff - x0) / ax));
        int iy_min = static_cast<int>(std::floor((cy - cutoff - y0) / ay));
        int iy_max = static_cast<int>(std::ceil ((cy + cutoff - y0) / ay));
        int iz_min = static_cast<int>(std::floor((cz - cutoff - z0) / az));
        int iz_max = static_cast<int>(std::ceil ((cz + cutoff - z0) / az));

        ix_min = std::max(ix_min, 0);
        iy_min = std::max(iy_min, 0);
        iz_min = std::max(iz_min, 0);
        ix_max = std::min(ix_max, static_cast<int>(nx) - 1);
        iy_max = std::min(iy_max, static_cast<int>(ny) - 1);
        iz_max = std::min(iz_max, static_cast<int>(nz) - 1);

        if (ix_min > ix_max || iy_min > iy_max || iz_min > iz_max) {
            continue;
        }

        for (int iz = iz_min; iz <= iz_max; ++iz) {
            const double z = z0 + iz * az;
            const double dz = cz - z;
            const double dz2 = dz * dz;

            for (int iy = iy_min; iy <= iy_max; ++iy) {
                const double y = y0 + iy * ay;
                const double dy = cy - y;
                const double dy2 = dy * dy;

                for (int ix = ix_min; ix <= ix_max; ++ix) {
                    const double x = x0 + ix * ax;
                    const double dx = cx - x;
                    const double dx2 = dx * dx;

                    const double r2 = dx2 + dy2 + dz2;
                    if (r2 > cutoff2) continue;

                    double r = std::sqrt(r2);
                    if (r < min_r) r = min_r;

                    const double eps_r = sigmoidal_dielectric(r);
                    double contrib = c_factor * qi / (eps_r * r);
                    
                    if (r > r_on) {
                    
                        double s = (r - r_on) / switch_dist;
                        double s2 = s * s;
                        double s3 = s2 * s;
                        double scale = 1.0 - (3.0 * s2 - 2.0 * s3); 

                        contrib *= scale;
                    }

                    g(iz, iy, ix) += contrib;
                }
            }
        }
    }

    return grid;
}



// protein_positions: (N,3) double
// protein_atom_roles: (N,) int (0=ignore, 1=donor, 2=acceptor, 3=both)
// grid_shape: (nz,ny,nx)
// origin:     (3,) [x0,y0,z0] in Å
// apix:       (3,) [ax,ay,az] in Å
//
// d0:       ideal heavy-heavy distance for ligand–water–protein bridge (Å)
// sigma:    Gaussian width in heavy-heavy distance (Å)
// max_dist: max heavy-heavy distance to consider (Å)
//
py::array_t<double> compute_water_bridge_grid_cpp(
    py::array_t<double> protein_positions,   // (N,3)
    py::array_t<int>    protein_atom_roles,  // (N,)
    py::tuple           grid_shape,          // (nz,ny,nx)
    py::array_t<double> origin,              // (3,)
    py::array_t<double> apix,                // (3,)
    double d0       = 5.2,
    double sigma    = 0.35,
    double max_dist = 6.0
) {
    // ---- sanity checks ----
    if (protein_positions.ndim() != 2 || protein_positions.shape(1) != 3) {
        throw std::runtime_error("protein_positions must be (N,3)");
    }
    if (protein_atom_roles.ndim() != 1 ||
        protein_atom_roles.shape(0) != protein_positions.shape(0)) {
        throw std::runtime_error("protein_atom_roles must be (N,) matching positions");
    }
    if (origin.ndim() != 1 || origin.shape(0) != 3) {
        throw std::runtime_error("origin must be (3,)");
    }
    if (apix.ndim() != 1 || apix.shape(0) != 3) {
        throw std::runtime_error("apix must be (3,)");
    }
    if (grid_shape.size() != 3) {
        throw std::runtime_error("grid_shape must be (nz,ny,nx)");
    }

    auto pos  = protein_positions.unchecked<2>();   // (N,3)
    auto role = protein_atom_roles.unchecked<1>();  // (N,)
    auto org  = origin.unchecked<1>();              // (3,)
    auto a    = apix.unchecked<1>();                // (3,)

    const py::ssize_t nz = grid_shape[0].cast<py::ssize_t>();
    const py::ssize_t ny = grid_shape[1].cast<py::ssize_t>();
    const py::ssize_t nx = grid_shape[2].cast<py::ssize_t>();

    const py::ssize_t N  = protein_positions.shape(0);

    const double x0 = org(0);
    const double y0 = org(1);
    const double z0 = org(2);
    const double ax = a(0);
    const double ay = a(1);
    const double az = a(2);

    if (sigma <= 0.0) {
        throw std::runtime_error("sigma must be > 0");
    }
    if (max_dist <= 0.0) {
        throw std::runtime_error("max_dist must be > 0");
    }

    const double inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);
    const double max_dist2      = max_dist * max_dist;

    // ---- allocate and zero grid ----
    py::array_t<double> grid({nz, ny, nx});
    auto g = grid.mutable_unchecked<3>();
    for (py::ssize_t iz = 0; iz < nz; ++iz)
        for (py::ssize_t iy = 0; iy < ny; ++iy)
            for (py::ssize_t ix = 0; ix < nx; ++ix)
                g(iz, iy, ix) = 0.0;

    // ---- accumulate contributions from polar protein atoms ----
    for (py::ssize_t i = 0; i < N; ++i) {
        int r = role(i);
        if (r == 0) {
            continue;  // skip non-polar / non-Hbond atoms
        }

        const double cx = pos(i, 0);
        const double cy = pos(i, 1);
        const double cz = pos(i, 2);

        // bounding box in index space, based on max_dist
        int ix_min = static_cast<int>(std::floor((cx - max_dist - x0) / ax));
        int ix_max = static_cast<int>(std::ceil ((cx + max_dist - x0) / ax));
        int iy_min = static_cast<int>(std::floor((cy - max_dist - y0) / ay));
        int iy_max = static_cast<int>(std::ceil ((cy + max_dist - y0) / ay));
        int iz_min = static_cast<int>(std::floor((cz - max_dist - z0) / az));
        int iz_max = static_cast<int>(std::ceil ((cz + max_dist - z0) / az));

        ix_min = std::max(ix_min, 0);
        iy_min = std::max(iy_min, 0);
        iz_min = std::max(iz_min, 0);
        ix_max = std::min(ix_max, static_cast<int>(nx) - 1);
        iy_max = std::min(iy_max, static_cast<int>(ny) - 1);
        iz_max = std::min(iz_max, static_cast<int>(nz) - 1);

        if (ix_min > ix_max || iy_min > iy_max || iz_min > iz_max) {
            continue;
        }

        for (int iz = iz_min; iz <= iz_max; ++iz) {
            const double z  = z0 + iz * az;
            const double dz = cz - z;
            const double dz2 = dz * dz;

            for (int iy = iy_min; iy <= iy_max; ++iy) {
                const double y  = y0 + iy * ay;
                const double dy = cy - y;
                const double dy2 = dy * dy;

                for (int ix = ix_min; ix <= ix_max; ++ix) {
                    const double x  = x0 + ix * ax;
                    const double dx = cx - x;
                    const double dx2 = dx * dx;

                    const double r2 = dx2 + dy2 + dz2;
                    if (r2 > max_dist2) continue;

                    const double d    = std::sqrt(r2);
                    const double diff = d - d0;

                    // ignore voxels too far from the ideal shell
                    if (std::fabs(diff) > 2.0 * sigma) continue;

                    const double w = std::exp(-diff * diff * inv_two_sigma2);
                    if (w < 1e-6) continue;  // drop tiny contributions

                    g(iz, iy, ix) += w;
                }
            }
        }
    }

    // ---- optional: normalize to [0,1] ----
    double g_max = 0.0;
    for (py::ssize_t iz = 0; iz < nz; ++iz)
        for (py::ssize_t iy = 0; iy < ny; ++iy)
            for (py::ssize_t ix = 0; ix < nx; ++ix)
                g_max = std::max(g_max, g(iz, iy, ix));

    if (g_max > 0.0) {
        const double inv_gmax = 1.0 / g_max;
        for (py::ssize_t iz = 0; iz < nz; ++iz)
            for (py::ssize_t iy = 0; iy < ny; ++iy)
                for (py::ssize_t ix = 0; ix < nx; ++ix)
                    g(iz, iy, ix) *= inv_gmax;
    }

    return grid;
}



// protein_positions: (N,3) double
// protein_atom_roles: (N,) int  (0=ignore, 1=donor, 2=acceptor, 3=both)
// protein_hbond_dirs: (N,3) double, unit-ish vectors for HBond lobe;
//                     (0,0,0) for atoms where you don't want directionality
// grid_shape: (nz,ny,nx)
// origin:     (3,) [x0,y0,z0] in Å
// apix:       (3,) [ax,ay,az] in Å
//
// d0:          ideal heavy-heavy distance for ligand–water–protein bridge (Å)
// sigma:       Gaussian width in heavy-heavy distance (Å)
// max_dist:    max heavy-heavy distance to consider (Å)
// theta_max:   maximum allowed angle between dir and (x - p) in degrees
//              e.g. 60° => cos0 ≈ 0.5
//
py::array_t<double> compute_water_bridge_grid_dir_cpp(
    py::array_t<double> protein_positions,    // (N,3)
    py::array_t<int>    protein_atom_roles,   // (N,)
    py::array_t<double> protein_hbond_dirs,   // (N,3)
    py::tuple           grid_shape,           // (nz,ny,nx)
    py::array_t<double> origin,               // (3,)
    py::array_t<double> apix,                 // (3,)
    double d0        = 5.2,
    double sigma     = 0.35,
    double max_dist  = 6.0,
    double theta_max = 60.0   // degrees
) {
    // ---- sanity checks ----
    if (protein_positions.ndim() != 2 || protein_positions.shape(1) != 3) {
        throw std::runtime_error("protein_positions must be (N,3)");
    }
    if (protein_atom_roles.ndim() != 1 ||
        protein_atom_roles.shape(0) != protein_positions.shape(0)) {
        throw std::runtime_error("protein_atom_roles must be (N,) matching positions");
    }
    if (protein_hbond_dirs.ndim() != 2 ||
        protein_hbond_dirs.shape(0) != protein_positions.shape(0) ||
        protein_hbond_dirs.shape(1) != 3) {
        throw std::runtime_error("protein_hbond_dirs must be (N,3) matching positions");
    }
    if (origin.ndim() != 1 || origin.shape(0) != 3) {
        throw std::runtime_error("origin must be (3,)");
    }
    if (apix.ndim() != 1 || apix.shape(0) != 3) {
        throw std::runtime_error("apix must be (3,)");
    }
    if (grid_shape.size() != 3) {
        throw std::runtime_error("grid_shape must be (nz,ny,nx)");
    }

    auto pos   = protein_positions.unchecked<2>();   // (N,3)
    auto role  = protein_atom_roles.unchecked<1>();  // (N,)
    auto dirs  = protein_hbond_dirs.unchecked<2>();  // (N,3)
    auto org   = origin.unchecked<1>();              // (3,)
    auto a     = apix.unchecked<1>();                // (3,)

    const py::ssize_t nz = grid_shape[0].cast<py::ssize_t>();
    const py::ssize_t ny = grid_shape[1].cast<py::ssize_t>();
    const py::ssize_t nx = grid_shape[2].cast<py::ssize_t>();

    const py::ssize_t N  = protein_positions.shape(0);

    const double x0 = org(0);
    const double y0 = org(1);
    const double z0 = org(2);
    const double ax = a(0);
    const double ay = a(1);
    const double az = a(2);

    if (sigma <= 0.0)  throw std::runtime_error("sigma must be > 0");
    if (max_dist <= 0.0) throw std::runtime_error("max_dist must be > 0");

    const double inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);
    const double max_dist2      = max_dist * max_dist;

    // angular cutoff
    const double theta_max_rad = theta_max * M_PI / 180.0;
    const double cos0          = std::cos(theta_max_rad);  // minimum allowed cos(theta)

    // ---- allocate and zero grid ----
    py::array_t<double> grid({nz, ny, nx});
    auto g = grid.mutable_unchecked<3>();
    for (py::ssize_t iz = 0; iz < nz; ++iz)
        for (py::ssize_t iy = 0; iy < ny; ++iy)
            for (py::ssize_t ix = 0; ix < nx; ++ix)
                g(iz, iy, ix) = 0.0;

    // ---- accumulate contributions from polar protein atoms ----
    for (py::ssize_t i = 0; i < N; ++i) {
        int r = role(i);
        if (r == 0) continue;  // skip non-Hbond atoms

        // fetch and normalize direction
        double dx_dir = dirs(i,0);
        double dy_dir = dirs(i,1);
        double dz_dir = dirs(i,2);
        double norm2  = dx_dir*dx_dir + dy_dir*dy_dir + dz_dir*dz_dir;
        if (norm2 < 1e-6) {
            // no direction info; skip or treat isotropic
            continue;
        }
        double inv_norm = 1.0 / std::sqrt(norm2);
        dx_dir *= inv_norm;
        dy_dir *= inv_norm;
        dz_dir *= inv_norm;

        const double cx = pos(i, 0);
        const double cy = pos(i, 1);
        const double cz = pos(i, 2);

        // bounding box in index space, based on max_dist
        int ix_min = static_cast<int>(std::floor((cx - max_dist - x0) / ax));
        int ix_max = static_cast<int>(std::ceil ((cx + max_dist - x0) / ax));
        int iy_min = static_cast<int>(std::floor((cy - max_dist - y0) / ay));
        int iy_max = static_cast<int>(std::ceil ((cy + max_dist - y0) / ay));
        int iz_min = static_cast<int>(std::floor((cz - max_dist - z0) / az));
        int iz_max = static_cast<int>(std::ceil ((cz + max_dist - z0) / az));

        ix_min = std::max(ix_min, 0);
        iy_min = std::max(iy_min, 0);
        iz_min = std::max(iz_min, 0);
        ix_max = std::min(ix_max, static_cast<int>(nx) - 1);
        iy_max = std::min(iy_max, static_cast<int>(ny) - 1);
        iz_max = std::min(iz_max, static_cast<int>(nz) - 1);

        if (ix_min > ix_max || iy_min > iy_max || iz_min > iz_max) continue;

        for (int iz = iz_min; iz <= iz_max; ++iz) {
            const double z  = z0 + iz * az;
            const double dz = z - cz;            // note: grid - atom
            const double dz2 = dz * dz;

            for (int iy = iy_min; iy <= iy_max; ++iy) {
                const double y  = y0 + iy * ay;
                const double dy = y - cy;
                const double dy2 = dy * dy;

                for (int ix = ix_min; ix <= ix_max; ++ix) {
                    const double x  = x0 + ix * ax;
                    const double dx = x - cx;
                    const double dx2 = dx * dx;

                    const double r2 = dx2 + dy2 + dz2;
                    if (r2 > max_dist2) continue;

                    const double d = std::sqrt(r2);
                    const double diff = d - d0;

                    if (std::fabs(diff) > 2.0 * sigma) continue;

                    // angular part: direction from atom -> grid point vs dir_i
                    double inv_d = (d > 1e-6) ? 1.0 / d : 0.0;
                    double ux = dx * inv_d;
                    double uy = dy * inv_d;
                    double uz = dz * inv_d;

                    double cth = ux*dx_dir + uy*dy_dir + uz*dz_dir;
                    if (cth <= cos0) continue;  // outside lobe

                    // map [cos0,1] -> [0,1] smoothly
                    double ang = (cth - cos0) / (1.0 - cos0);
                    // optional: sharpen
                    ang = ang * ang;

                    double radial = std::exp(-diff * diff * inv_two_sigma2);
                    double w = radial * ang;
                    if (w < 1e-6) continue;

                    g(iz, iy, ix) += w;
                }
            }
        }
    }

    // ---- normalize to [0,1] ----
    double g_max = 0.0;
    for (py::ssize_t iz = 0; iz < nz; ++iz)
        for (py::ssize_t iy = 0; iy < ny; ++iy)
            for (py::ssize_t ix = 0; ix < nx; ++ix)
                g_max = std::max(g_max, g(iz, iy, ix));

    if (g_max > 0.0) {
        double inv_gmax = 1.0 / g_max;
        for (py::ssize_t iz = 0; iz < nz; ++iz)
            for (py::ssize_t iy = 0; iy < ny; ++iy)
                for (py::ssize_t ix = 0; ix < nx; ++ix)
                    g(iz, iy, ix) *= inv_gmax;
    }

    return grid;
}



inline std::size_t idx3D(int z, int y, int x, int ny, int nx) {
    return static_cast<std::size_t>(z) * ny * nx +
           static_cast<std::size_t>(y) * nx +
           static_cast<std::size_t>(x);
}

// Build a thin spherical shell kernel of radius R_sasa (Å)
std::vector<float> make_shell_kernel_R_cpp(
    double R_sasa,
    double spacing,
    double shell_thickness,
    int &kz, int &ky, int &kx,
    int &cz, int &cy, int &cx
) {
    const double R = R_sasa;
    const double sp = spacing;

    const double R_vox = R / sp;
    const double half_thick_vox = (shell_thickness / 2.0) / sp;

    const int r_vox = static_cast<int>(std::ceil(R_vox + half_thick_vox));
    kz = ky = kx = 2 * r_vox + 1;
    cz = cy = cx = r_vox;

    std::vector<float> kernel(static_cast<std::size_t>(kz) * ky * kx, 0.0f);

    for (int dz = -r_vox; dz <= r_vox; ++dz) {
        for (int dy = -r_vox; dy <= r_vox; ++dy) {
            for (int dx = -r_vox; dx <= r_vox; ++dx) {
                const double dist_vox = std::sqrt(
                    static_cast<double>(dx*dx + dy*dy + dz*dz)
                );
                const double dist_ang = dist_vox * sp;
                if (std::abs(dist_ang - R) <= (shell_thickness / 2.0)) {
                    const int zz = cz + dz;
                    const int yy = cy + dy;
                    const int xx = cx + dx;
                    kernel[idx3D(zz, yy, xx, ky, kx)] = 1.0f;
                }
            }
        }
    }

    // Normalize so sum(kernel) ≈ 4πR² (Å²)
    double shell_voxels = 0.0;
    for (float v : kernel) shell_voxels += static_cast<double>(v);
    if (shell_voxels > 0.0) {
        const double area_total = 4.0 * M_PI * R * R;
        const float scale = static_cast<float>(area_total / shell_voxels);
        for (auto &v : kernel) v *= scale;
    }

    return kernel;
}

// Fast generic ΔSASA grid builder (cropped + OpenMP)
py::array_t<float> build_delta_sasa_generic_grid_cpp(
    py::array_t<uint8_t> sasa_mask,  // (nz,ny,nx), 0/1 or bool
    double spacing,
    double R_sasa = 3.0,
    double shell_thickness = 1.0
) {
    if (sasa_mask.ndim() != 3) {
        throw std::runtime_error("sasa_mask must be 3D (nz,ny,nx).");
    }

    auto sm = sasa_mask.unchecked<3>();
    const int nz = static_cast<int>(sm.shape(0));
    const int ny = static_cast<int>(sm.shape(1));
    const int nx = static_cast<int>(sm.shape(2));

    // Find bounding box of SASA region
    bool any_sasa = false;
    int zmin = nz, zmax = -1;
    int ymin = ny, ymax = -1;
    int xmin = nx, xmax = -1;

    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                if (sm(z, y, x)) {
                    any_sasa = true;
                    if (z < zmin) zmin = z;
                    if (z > zmax) zmax = z;
                    if (y < ymin) ymin = y;
                    if (y > ymax) ymax = y;
                    if (x < xmin) xmin = x;
                    if (x > xmax) xmax = x;
                }
            }
        }
    }

    // If no SASA at all, just return zeros
    py::array_t<float> delta({nz, ny, nx});
    auto dg = delta.mutable_unchecked<3>();

    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                dg(z, y, x) = 0.0f;

    if (!any_sasa) {
        return delta;
    }

    // Build shell kernel
    int kz, ky, kx, cz, cy, cx;
    std::vector<float> kernel = make_shell_kernel_R_cpp(
        R_sasa, spacing, shell_thickness,
        kz, ky, kx,
        cz, cy, cx
    );

    // Expand bounding box by kernel radius (so centers that "see" SASA are included)
    int pad_z = cz;
    int pad_y = cy;
    int pad_x = cx;

    int czmin = std::max(0, zmin - pad_z);
    int czmax = std::min(nz - 1, zmax + pad_z);
    int cymin = std::max(0, ymin - pad_y);
    int cymax = std::min(ny - 1, ymax + pad_y);
    int cxmin = std::max(0, xmin - pad_x);
    int cxmax = std::min(nx - 1, xmax + pad_x);

    // 3D convolution over cropped region only
    // mode="constant", cval=0.0
    //#pragma omp parallel for collapse(2) if(nz*ny*nx > 100000) default(shared)
    for (int z = czmin; z <= czmax; ++z) {
        for (int y = cymin; y <= cymax; ++y) {
            for (int x = cxmin; x <= cxmax; ++x) {
                float sum = 0.0f;

                for (int dz = 0; dz < kz; ++dz) {
                    const int gz = z + dz - cz;
                    if (gz < 0 || gz >= nz) continue;

                    for (int dy = 0; dy < ky; ++dy) {
                        const int gy = y + dy - cy;
                        if (gy < 0 || gy >= ny) continue;

                        for (int dx = 0; dx < kx; ++dx) {
                            const int gx = x + dx - cx;
                            if (gx < 0 || gx >= nx) continue;

                            const float kval = kernel[idx3D(dz, dy, dx, ky, kx)];
                            if (kval == 0.0f) continue;

                            if (sm(gz, gy, gx)) {
                                sum += kval;  // base=1 inside SASA, 0 otherwise
                            }
                        }
                    }
                }

                dg(z, y, x) = sum;
            }
        }
    }

    // Zero outside SASA, as in Python version
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                if (!sm(z, y, x)) {
                    dg(z, y, x) = 0.0f;
                }
            }
        }
    }

    return delta;
}




//---------------------------------------------------------------------
// Pybind11 Module Definition
//---------------------------------------------------------------------
PYBIND11_MODULE(grid_maps, m) {
    m.doc() = "Compute 3D electrostatic and hydrophobic grids using optimized C++ and pybind11";
    
    m.def("compute_electostatic_grid", &compute_electostatic_grid,
          "Compute the electrostatic potential grid with cutoff",
          py::arg("protein_positions"),
          py::arg("protein_charges"),
          py::arg("binding_site_map"),
          py::arg("binding_site_origin"),
          py::arg("apix"),
          py::arg("c_factor") = 332.06,
          py::arg("eps0") = 4.0,
          py::arg("k") = 0.2,
          py::arg("min_r") = 0.001,
          py::arg("cutoff_distance") = 9.0);
    
    m.def("compute_hydrophobic_grid", &compute_hydrophobic_grid,
          "Compute the hydrophobic grid",
          py::arg("protein_positions"),
          py::arg("protein_hpi"),
          py::arg("binding_site_map"),
          py::arg("binding_site_origin"),
          py::arg("apix"),
          py::arg("cutoff_distance") = 6.0);
    
    m.def("compute_electostatic_grid_no_cutoff", &compute_electostatic_grid_no_cutoff,
          "Compute the electrostatic potential grid without a distance cutoff",
          py::arg("protein_positions"),
          py::arg("protein_charges"),
          py::arg("binding_site_map"),
          py::arg("binding_site_origin"),
          py::arg("apix"),
          py::arg("c_factor") = 332.06,
          py::arg("eps0") = 4.0,
          py::arg("k") = 0.2,
          py::arg("min_r") = 0.001);
    
    m.def("compute_hydrophobic_grid_gaussian", &compute_hydrophobic_grid_gaussian,
          "Compute the hydrophobic grid",
          py::arg("protein_positions"),
          py::arg("protein_hpi"),
          py::arg("binding_site_map"),
          py::arg("binding_site_origin"),
          py::arg("apix"),
          py::arg("sigma") = 2.0,
          py::arg("cutoff_distance") = 6.0);
    
    m.def("compute_enclosure_grid", &compute_enclosure_grid, "Compute the enclosure score grid via ray-casting",
          py::arg("protein_positions"), py::arg("protein_hpi"), py::arg("enclosure_map"),
          py::arg("binding_site_origin"), py::arg("apix"), py::arg("probe_vectors"),
          py::arg("ray_cutoff") = 5.0, py::arg("axis_tolerance") = 1.0);
    
        // ------------------------------------------------------------------
        // Water site maps (explicit-solvent placement helpers)
        // ------------------------------------------------------------------
        m.def("compute_water_acceptor_site_map", &compute_water_acceptor_site_map,
              "Mark spherical-shell voxels around donor H positions (water as acceptor)",
              py::arg("donor_H_positions"),
              py::arg("binding_site_map"),
              py::arg("binding_site_origin"),
              py::arg("apix"),
              py::arg("protein_positions"),
              py::arg("r_opt") = 2.9,
              py::arg("dr") = 0.6,
              py::arg("min_protein_clearance") = 1.8,
              py::arg("weight") = 1.0);
        
        m.def("compute_water_donor_site_map", &compute_water_donor_site_map,
              "Mark spherical-shell voxels around acceptor atoms (water as donor)",
              py::arg("acceptor_positions"),
              py::arg("binding_site_map"),
              py::arg("binding_site_origin"),
              py::arg("apix"),
              py::arg("protein_positions"),
              py::arg("r_opt") = 2.9,
              py::arg("dr") = 0.6,
              py::arg("min_protein_clearance") = 1.8,
              py::arg("weight") = 1.0);
        
        m.def("compute_water_site_maps", &compute_water_site_maps,
              "Build both water site maps (acceptor and donor) in one call",
              py::arg("donor_H_positions"),
              py::arg("acceptor_positions"),
              py::arg("binding_site_origin"),
              py::arg("apix"),
              py::arg("map_shapes"),
              py::arg("protein_positions"),
              py::arg("r_opt") = 2.9,
              py::arg("dr") = 0.6,
              py::arg("min_protein_clearance") = 1.8,
              py::arg("weight_acceptor_map") = 1.0,
              py::arg("weight_donor_map") = 1.0);
              
        m.def("make_protein_and_solvent_masks_cpp", &make_protein_and_solvent_masks_cpp,
            py::arg("atom_coords"),
            py::arg("atom_radii"),
            py::arg("origin"),
            py::arg("grid_shape"),
            py::arg("spacing"),
            py::arg("probe_radius") = 1.4);
        
        m.def("depth_propagation_cpp", &depth_propagation_cpp,
            py::arg("bulk_mask"),
            py::arg("protein_mask"),
            py::arg("target_mask"),
            py::arg("spacing") = 1.0,
            py::arg("connectivity") = 6);
        

        m.def("prepare_depth_for_mrc_cpp", &prepare_depth_for_mrc_cpp,
            py::arg("depth_map"),
            py::arg("sasa_mask") = py::none(),
            py::arg("fill_unreachable") = py::none());
        
        m.def("widest_path_cpp", &widest_path_cpp,
            py::arg("r_local"),
            py::arg("solvent_mask"),
            py::arg("seed_mask"),
            py::arg("connectivity") = 6,
            py::arg("power") = 4 );
        
        m.def("propagate_logp_exp_decay_cpp", &propagate_logp_exp_decay_cpp,
            py::arg("logp_centers"),
            py::arg("logp_values"),
            py::arg("origin"),
            py::arg("grid_shape"),
            py::arg("spacing"),
            py::arg("sasa_mask") = py::none(),
            py::arg("cutoff") = 6.0);
        
        m.def(
            "compute_electrostatic_grid_cutoff_cpp",
            &compute_electrostatic_grid_cutoff_cpp,
            py::arg("protein_positions"),
            py::arg("protein_charges"),
            py::arg("grid_shape"),
            py::arg("origin"),
            py::arg("apix"),
            py::arg("c_factor") = 332.06,
            py::arg("min_r")    = 0.5,
            py::arg("cutoff")   = 12.0,
            py::arg("switch_dist")   = 12.0);
        
        m.def(
            "build_delta_sasa_generic_grid_cpp",
            &build_delta_sasa_generic_grid_cpp,
            py::arg("sasa_mask"),
            py::arg("spacing"),
            py::arg("R_sasa") = 3.0,
            py::arg("shell_thickness") = 1.0);
        
        m.def(
            "compute_water_bridge_grid_cpp",
            &compute_water_bridge_grid_cpp,
            py::arg("protein_positions"),
            py::arg("protein_atom_roles"),
            py::arg("grid_shape"),
            py::arg("origin"),
            py::arg("apix"),
            py::arg("d0")       = 5.0,
            py::arg("sigma")    = 1.0,
            py::arg("max_dist") = 7.0,
            "Compute a receptor-only water-bridge likelihood grid.\n"
            "Grid value at a point x is max_i exp(-(d(x,p_i)-d0)^2 / (2*sigma^2)),\n"
            "where p_i are polar protein atoms (roles != 0)."
        );

        m.def(
            "compute_water_bridge_grid_dir_cpp",
            &compute_water_bridge_grid_dir_cpp,
            py::arg("protein_positions"),
            py::arg("protein_atom_roles"),
            py::arg("protein_hbond_dirs"),
            py::arg("grid_shape"),
            py::arg("origin"),
            py::arg("apix"),
            py::arg("d0")        = 5.2,
            py::arg("sigma")     = 0.35,
            py::arg("max_dist")  = 6.0,
            py::arg("theta_max") = 60.0,
            "Compute a directional water-bridge likelihood grid.\n"
            "Uses both distance (d0, sigma) and an angular lobe defined by protein_hbond_dirs."
        );

            
       
    

}
