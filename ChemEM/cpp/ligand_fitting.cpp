#include "ligand_fitting.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <random>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <limits>

// Namespace for pybind11
namespace py = pybind11;

// Helper function to convert flat array to Eigen::Tensor
Eigen::Tensor<double, 3> flat_to_tensor(const std::vector<double>& flat, const std::vector<size_t>& dims) {
    if (dims.size() != 3) {
        throw std::runtime_error("D_exp_dims must have exactly 3 dimensions.");
    }
    size_t total_size = dims[0] * dims[1] * dims[2];
    if (flat.size() != total_size) {
        throw std::runtime_error("Flat D_exp array size does not match provided dimensions.");
    }
    
    // Explicitly cast size_t to Eigen::Index
    Eigen::Index dim0 = static_cast<Eigen::Index>(dims[0]);
    Eigen::Index dim1 = static_cast<Eigen::Index>(dims[1]);
    Eigen::Index dim2 = static_cast<Eigen::Index>(dims[2]);
    
    Eigen::Tensor<double, 3> tensor(dim0, dim1, dim2);
    
    for (size_t z = 0; z < dims[0]; ++z) {
        for (size_t y = 0; y < dims[1]; ++y) {
            for (size_t x = 0; x < dims[2]; ++x) {
                // Cast loop indices to Eigen::Index
                tensor(static_cast<Eigen::Index>(z),
                       static_cast<Eigen::Index>(y),
                       static_cast<Eigen::Index>(x)) = flat[z * dims[1] * dims[2] + y * dims[2] + x];
            }
        }
    }
    return tensor;
}

// Compute the normalized cross-correlation (CCC) between two vectors
double compute_ccc_vector(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) {
    double numerator = v1.dot(v2);
    double denominator = v1.norm() * v2.norm();
    if (denominator == 0.0) {
        return 0.0;
    }
    return numerator / denominator;
}

// Trilinear interpolation of the density map
double trilinear_interpolate(
    const Eigen::Tensor<double, 3>& D_exp,
    double x, double y, double z
) {
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int z0 = static_cast<int>(std::floor(z));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    double xd = x - x0;
    double yd = y - y0;
    double zd = z - z0;

    auto get_value = [&](int xi, int yi, int zi) -> double {
        if (xi >= 0 && xi < D_exp.dimension(0) &&
            yi >= 0 && yi < D_exp.dimension(1) &&
            zi >= 0 && zi < D_exp.dimension(2)) {
            return D_exp(xi, yi, zi);
        } else {
            return 0.0;
        }
    };

    double c000 = get_value(x0, y0, z0);
    double c100 = get_value(x1, y0, z0);
    double c010 = get_value(x0, y1, z0);
    double c110 = get_value(x1, y1, z0);
    double c001 = get_value(x0, y0, z1);
    double c101 = get_value(x1, y0, z1);
    double c011 = get_value(x0, y1, z1);
    double c111 = get_value(x1, y1, z1);

    double c00 = c000 * (1 - xd) + c100 * xd;
    double c01 = c001 * (1 - xd) + c101 * xd;
    double c10 = c010 * (1 - xd) + c110 * xd;
    double c11 = c011 * (1 - xd) + c111 * xd;

    double c0 = c00 * (1 - yd) + c10 * yd;
    double c1 = c01 * (1 - yd) + c11 * yd;

    double c = c0 * (1 - zd) + c1 * zd;
    return c;
}

// Compute CCC between the expected and experimental densities
double compute_ccc_molecule_map(
    const std::vector<Eigen::Vector3d>& atom_positions,
    const std::vector<double>& atomic_masses,
    const std::vector<double>& D_exp_flat,
    const std::vector<size_t>& D_exp_dims,
    double voxel_size,
    const Eigen::Vector3d& origin,
    double sigma
) {
    Eigen::Tensor<double, 3> D_exp = flat_to_tensor(D_exp_flat, D_exp_dims);

    std::vector<double> expected_density_list;
    std::vector<double> experimental_density_list;
    
    // Define grid points relative to the atom positions
    double cutoff = sigma; //  3 * sigma Use 3 sigma cutoff
    double step = voxel_size;
    std::vector<double> grid_range;
    for (double r = -cutoff; r <= cutoff; r += step) {
        grid_range.push_back(r);
    }

    size_t grid_size = grid_range.size();
    size_t num_points = grid_size * grid_size * grid_size;

    Eigen::MatrixXd grid_offsets(num_points, 3);
    std::vector<double> gaussian_weights(num_points);

    size_t idx = 0;
    for (double dx : grid_range) {
        for (double dy : grid_range) {
            for (double dz : grid_range) {
                grid_offsets.row(idx) = Eigen::Vector3d(dx, dy, dz);
                double distance_squared = dx * dx + dy * dy + dz * dz;
                gaussian_weights[idx] = std::exp(-distance_squared / (2 * sigma * sigma));
                idx++;
            }
        }
    }

    for (size_t i = 0; i < atom_positions.size(); ++i) {
        const Eigen::Vector3d& atom_pos = atom_positions[i];
        double mass = atomic_masses[i];

        // Shift grid to the atom's position
        Eigen::MatrixXd grid_real_space = grid_offsets.rowwise() + atom_pos.transpose();

        // Convert to voxel indices
        Eigen::MatrixXd grid_voxel_space = (grid_real_space.rowwise() - origin.transpose()) / voxel_size;
        // Reorder to (z, y, x)
        Eigen::MatrixXd grid_voxel_indices = grid_voxel_space.rowwise().reverse();

        // Interpolate experimental density
        for (size_t j = 0; j < num_points; ++j) {
            double x = grid_voxel_indices(j, 0);
            double y = grid_voxel_indices(j, 1);
            double z = grid_voxel_indices(j, 2);
            double interpolated_density = trilinear_interpolate(D_exp, x, y, z);

            experimental_density_list.push_back(interpolated_density);
            expected_density_list.push_back(mass * gaussian_weights[j]);
        }
    }

    // Convert to Eigen vectors
    Eigen::VectorXd experimental_density = Eigen::Map<Eigen::VectorXd>(experimental_density_list.data(), experimental_density_list.size());
    Eigen::VectorXd expected_density = Eigen::Map<Eigen::VectorXd>(expected_density_list.data(), expected_density_list.size());

    // Compute CCC
    return compute_ccc_vector(experimental_density, expected_density);
}

// Rotate and translate atom positions
std::vector<Eigen::Vector3d> translate_and_rotate(
    const Eigen::VectorXd& params,
    const std::vector<Eigen::Vector3d>& atom_positions
) {
    Eigen::Vector3d rotation_vector = params.segment<3>(0);
    Eigen::Vector3d translation_vector = params.segment<3>(3);

    double angle = rotation_vector.norm();
    Eigen::Quaterniond rotation;
    if (angle != 0.0) {
        Eigen::Vector3d axis = rotation_vector.normalized();
        rotation = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
    } else {
        rotation = Eigen::Quaterniond::Identity();
    }

    // Compute centroid
    Eigen::Vector3d centroid(0.0, 0.0, 0.0);
    for (const auto& pos : atom_positions) {
        centroid += pos;
    }
    centroid /= atom_positions.size();

    std::vector<Eigen::Vector3d> transformed_coords;
    transformed_coords.reserve(atom_positions.size());

    for (const auto& pos : atom_positions) {
        Eigen::Vector3d centered = pos - centroid;
        Eigen::Vector3d rotated = rotation * centered;
        Eigen::Vector3d transformed = rotated + centroid + translation_vector;
        transformed_coords.push_back(transformed);
    }

    return transformed_coords;
}

// Local optimization function
std::tuple<double, std::vector<Eigen::Vector3d>> local_optimization(
    const std::vector<Eigen::Vector3d>& atom_positions,
    const std::vector<double>& atomic_masses,
    const std::vector<double>& D_exp_flat,
    const std::vector<size_t>& D_exp_dims,
    double voxel_size,
    const Eigen::Vector3d& origin,
    double sigma,
    const Eigen::Quaterniond& initial_rotation,
    const Eigen::Vector3d& initial_translation,
    double initial_step_size,
    int max_steps
) {
    // Initialize parameters
    Eigen::Quaterniond rotation = initial_rotation;
    Eigen::Vector3d translation = initial_translation;
    double step_size = initial_step_size;
    int step_count = 0;

    // Compute initial CCC
    Eigen::VectorXd params(6);
    Eigen::AngleAxisd angle_axis(rotation);
    params.segment<3>(0) = angle_axis.angle() * angle_axis.axis();
    params.segment<3>(3) = translation;
    auto transformed_coords = translate_and_rotate(params, atom_positions);

    double current_ccc = compute_ccc_molecule_map(transformed_coords, atomic_masses, D_exp_flat, D_exp_dims, voxel_size, origin, sigma);
    double best_ccc = current_ccc;
    Eigen::Quaterniond best_rotation = rotation;
    Eigen::Vector3d best_translation = translation;

    // Optimization loop
    while (step_count < max_steps && step_size > 1e-3) {
        for (int i = 0; i < 4; ++i) {
            // Alternate between translation and rotation
            if (step_count % 2 == 0) {
                // Translation step
                Eigen::Vector3d translation_gradient = Eigen::Vector3d::Zero();
                double delta = 1e-3;
                for (int axis = 0; axis < 3; ++axis) {
                    Eigen::Vector3d delta_translation = Eigen::Vector3d::Zero();
                    delta_translation[axis] = delta;
                    Eigen::Vector3d new_translation = translation + delta_translation;

                    // Update parameters
                    params.segment<3>(0) = angle_axis.angle() * angle_axis.axis();
                    params.segment<3>(3) = new_translation;

                    auto transformed_coords_delta = translate_and_rotate(params, atom_positions);
                    double ccc_delta = compute_ccc_molecule_map(transformed_coords_delta, atomic_masses, D_exp_flat, D_exp_dims, voxel_size, origin, sigma);
                    translation_gradient[axis] = (ccc_delta - current_ccc) / delta;
                }

                // Normalize gradient
                if (translation_gradient.norm() != 0) {
                    translation += step_size * (translation_gradient / translation_gradient.norm());
                }

            } else {
                // Rotation step
                Eigen::Vector3d rotation_gradient = Eigen::Vector3d::Zero();
                double delta = 1e-3;
                for (int axis = 0; axis < 3; ++axis) {
                    Eigen::Vector3d delta_rotation = Eigen::Vector3d::Zero();
                    delta_rotation[axis] = delta;
                    double angle = delta_rotation.norm();
                    Eigen::Quaterniond delta_rotation_quat = (angle != 0.0) ? Eigen::Quaterniond(Eigen::AngleAxisd(angle, delta_rotation.normalized())) : Eigen::Quaterniond::Identity();
                    Eigen::Quaterniond new_rotation = delta_rotation_quat * rotation;

                    // Update parameters
                    Eigen::AngleAxisd new_angle_axis(new_rotation);
                    params.segment<3>(0) = new_angle_axis.angle() * new_angle_axis.axis();
                    params.segment<3>(3) = translation;

                    auto transformed_coords_delta = translate_and_rotate(params, atom_positions);
                    double ccc_delta = compute_ccc_molecule_map(transformed_coords_delta, atomic_masses, D_exp_flat, D_exp_dims, voxel_size, origin, sigma);
                    rotation_gradient[axis] = (ccc_delta - current_ccc) / delta;
                }

                // Normalize gradient
                if (rotation_gradient.norm() != 0) {
                    Eigen::Vector3d delta_rotation = step_size * (rotation_gradient / rotation_gradient.norm());
                    double angle = delta_rotation.norm();
                    Eigen::Quaterniond delta_rotation_quat = (angle != 0.0) ? Eigen::Quaterniond(Eigen::AngleAxisd(angle, delta_rotation.normalized())) : Eigen::Quaterniond::Identity();
                    rotation = delta_rotation_quat * rotation;
                }
            }

            // Update parameters
            Eigen::AngleAxisd angle_axis(rotation);
            params.segment<3>(0) = angle_axis.angle() * angle_axis.axis();
            params.segment<3>(3) = translation;
            transformed_coords = translate_and_rotate(params, atom_positions);
            current_ccc = compute_ccc_molecule_map(transformed_coords, atomic_masses, D_exp_flat, D_exp_dims, voxel_size, origin, sigma);

            step_count++;

            // Update best CCC and parameters if improved
            if (current_ccc > best_ccc) {
                best_ccc = current_ccc;
                best_rotation = rotation;
                best_translation = translation;
            }

            if (step_count >= max_steps) {
                break;
            }
        }

        // Adjust step size
        step_size /= 2;

        // Convergence check
        if (step_size < 1e-3) {
            break;
        }
    }

    // Final transformed coordinates
    Eigen::VectorXd best_params(6);
    Eigen::AngleAxisd best_angle_axis(best_rotation);
    best_params.segment<3>(0) = best_angle_axis.angle() * best_angle_axis.axis();
    best_params.segment<3>(3) = best_translation;
    auto best_coords = translate_and_rotate(best_params, atom_positions);

    return std::make_tuple(best_ccc, best_coords);
}

// Function to generate a random rotation
Eigen::Quaterniond random_rotation(double max_angle) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> angle_dist(-max_angle, max_angle);
    std::uniform_real_distribution<> axis_dist(-1.0, 1.0);

    Eigen::Vector3d axis(axis_dist(gen), axis_dist(gen), axis_dist(gen));
    if (axis.norm() == 0.0) {
        axis = Eigen::Vector3d(1.0, 0.0, 0.0); // Default axis to avoid division by zero
    } else {
        axis.normalize();
    }
    double angle = angle_dist(gen);
    Eigen::AngleAxisd angle_axis(angle, axis);
    return Eigen::Quaterniond(angle_axis); // Correctly convert to Quaterniond
}

// Global search function
std::tuple<double, std::vector<Eigen::Vector3d>, std::vector<double>> global_search(
    const std::vector<Eigen::Vector3d>& atom_positions,
    const std::vector<double>& atomic_masses,
    const std::vector<double>& D_exp_flat,
    const std::vector<size_t>& D_exp_dims,
    double voxel_size,
    const Eigen::Vector3d& origin,
    double sigma,
    int N,
    double initial_step_size,
    int max_steps,
    double max_translation
) {
    double best_ccc = -std::numeric_limits<double>::infinity();
    std::vector<Eigen::Vector3d> best_coords;
    std::vector<double> all_cccs;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> translation_dist(-max_translation, max_translation);

    for (int i = 0; i < N; ++i) {
        // Generate random initial rotation and translation
        Eigen::Quaterniond initial_rotation = random_rotation(M_PI);
        Eigen::Vector3d initial_translation(
            translation_dist(gen), translation_dist(gen), translation_dist(gen)
        );

        // Perform local optimization
        auto [ccc, coords] = local_optimization(
            atom_positions, atomic_masses, D_exp_flat, D_exp_dims, voxel_size, origin, sigma,
            initial_rotation, initial_translation, initial_step_size, max_steps
        );
        all_cccs.push_back(ccc);
        // Update best result if necessary
        if (ccc > best_ccc) {
            best_ccc = ccc;
            best_coords = coords;
        }
    }

    return std::make_tuple(best_ccc, best_coords, all_cccs);
}

std::vector<double> gaussian_kernel_1d(double sigma) {
    if (sigma <= 1e-12) {
        return std::vector<double>{1.0};
    }
    int radius = static_cast<int>(4.0 * sigma + 0.5);
    radius = std::max(radius, 0);
    std::vector<double> kernel(static_cast<size_t>(2 * radius + 1), 0.0);
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        double v = std::exp(-0.5 * (static_cast<double>(i) * static_cast<double>(i)) / (sigma * sigma));
        kernel[static_cast<size_t>(i + radius)] = v;
        sum += v;
    }
    if (sum > 0.0) {
        for (double& v : kernel) {
            v /= sum;
        }
    }
    return kernel;
}

void convolve_axis_constant(
    const std::vector<double>& input,
    std::vector<double>& output,
    size_t nz,
    size_t ny,
    size_t nx,
    const std::vector<double>& kernel,
    int axis
) {
    const int radius = static_cast<int>(kernel.size() / 2);
    std::fill(output.begin(), output.end(), 0.0);
    auto offset = [ny, nx](size_t z, size_t y, size_t x) {
        return (z * ny * nx) + (y * nx) + x;
    };

    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                double acc = 0.0;
                for (int k = -radius; k <= radius; ++k) {
                    long zz = static_cast<long>(z);
                    long yy = static_cast<long>(y);
                    long xx = static_cast<long>(x);
                    if (axis == 0) zz += k;
                    if (axis == 1) yy += k;
                    if (axis == 2) xx += k;
                    if (zz < 0 || yy < 0 || xx < 0) {
                        continue;
                    }
                    if (zz >= static_cast<long>(nz) || yy >= static_cast<long>(ny) || xx >= static_cast<long>(nx)) {
                        continue;
                    }
                    acc += input[offset(static_cast<size_t>(zz), static_cast<size_t>(yy), static_cast<size_t>(xx))]
                           * kernel[static_cast<size_t>(k + radius)];
                }
                output[offset(z, y, x)] = acc;
            }
        }
    }
}

py::array_t<double> simulate_ligand_density_subgrid_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> coords_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> masses_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> origin_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> apix_in,
    py::array_t<int, py::array::c_style | py::array::forcecast> bbox_lo_in,
    py::array_t<int, py::array::c_style | py::array::forcecast> bbox_hi_in,
    double sigma_A,
    bool normalise
) {
    auto coords = coords_in.unchecked<2>();
    auto masses = masses_in.unchecked<1>();
    auto origin = origin_in.unchecked<1>();
    auto apix = apix_in.unchecked<1>();
    auto lo = bbox_lo_in.unchecked<1>();
    auto hi = bbox_hi_in.unchecked<1>();
    if (coords.shape(1) != 3 || masses.shape(0) != coords.shape(0)) {
        throw std::runtime_error("coords must be (N,3) and masses length N");
    }

    const int z0 = lo(0), y0 = lo(1), x0 = lo(2);
    const int z1 = hi(0), y1 = hi(1), x1 = hi(2);
    const size_t nz = static_cast<size_t>(std::max(0, z1 - z0));
    const size_t ny = static_cast<size_t>(std::max(0, y1 - y0));
    const size_t nx = static_cast<size_t>(std::max(0, x1 - x0));
    py::array_t<double> out({
        static_cast<py::ssize_t>(nz),
        static_cast<py::ssize_t>(ny),
        static_cast<py::ssize_t>(nx),
    });
    if (nz == 0 || ny == 0 || nx == 0) {
        return out;
    }

    std::vector<double> grid(nz * ny * nx, 0.0);
    auto offset = [ny, nx](size_t z, size_t y, size_t x) {
        return (z * ny * nx) + (y * nx) + x;
    };

    for (py::ssize_t i = 0; i < coords.shape(0); ++i) {
        int ix = static_cast<int>(std::nearbyint((coords(i, 0) - origin(0)) / apix(0)));
        int iy = static_cast<int>(std::nearbyint((coords(i, 1) - origin(1)) / apix(1)));
        int iz = static_cast<int>(std::nearbyint((coords(i, 2) - origin(2)) / apix(2)));
        if (iz < z0 || iy < y0 || ix < x0 || iz >= z1 || iy >= y1 || ix >= x1) {
            continue;
        }
        grid[offset(static_cast<size_t>(iz - z0), static_cast<size_t>(iy - y0), static_cast<size_t>(ix - x0))] += masses(i);
    }

    const double sigma_z = sigma_A / std::max(std::abs(apix(2)), 1e-12);
    const double sigma_y = sigma_A / std::max(std::abs(apix(1)), 1e-12);
    const double sigma_x = sigma_A / std::max(std::abs(apix(0)), 1e-12);
    std::vector<double> tmp1(grid.size(), 0.0);
    std::vector<double> tmp2(grid.size(), 0.0);
    convolve_axis_constant(grid, tmp1, nz, ny, nx, gaussian_kernel_1d(sigma_z), 0);
    convolve_axis_constant(tmp1, tmp2, nz, ny, nx, gaussian_kernel_1d(sigma_y), 1);
    convolve_axis_constant(tmp2, grid, nz, ny, nx, gaussian_kernel_1d(sigma_x), 2);

    if (normalise) {
        double vmax = 0.0;
        for (double v : grid) {
            vmax = std::max(vmax, v);
        }
        if (vmax > 0.0) {
            for (double& v : grid) {
                v /= vmax;
            }
        }
    }

    auto out_mut = out.mutable_unchecked<3>();
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                out_mut(z, y, x) = grid[offset(z, y, x)];
            }
        }
    }
    return out;
}

double ccc_from_sums_cpp(
    size_t n,
    double sum_a,
    double sumsq_a,
    double sum_b,
    double sumsq_b,
    double sum_ab
) {
    if (n < 4) {
        return 0.0;
    }
    const double nf = static_cast<double>(n);
    const double numerator = sum_ab - ((sum_a * sum_b) / nf);
    double var_a = sumsq_a - ((sum_a * sum_a) / nf);
    double var_b = sumsq_b - ((sum_b * sum_b) / nf);
    if (var_a < 0.0 && var_a > -1e-9) var_a = 0.0;
    if (var_b < 0.0 && var_b > -1e-9) var_b = 0.0;
    const double denom = std::sqrt(std::max(0.0, var_a) * std::max(0.0, var_b));
    if (denom < 1e-12) {
        return 0.0;
    }
    double cc = numerator / denom;
    if (!std::isfinite(cc)) {
        return 0.0;
    }
    return std::max(0.0, cc);
}

double compute_ligand_ccc_decomposed_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> exp_sub_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> sim_sub_in,
    size_t full_nonzero_n,
    double full_nonzero_sum,
    double full_nonzero_sumsq,
    size_t full_finite_n,
    double full_finite_sum,
    double full_finite_sumsq
) {
    auto exp_sub = exp_sub_in.unchecked<3>();
    auto sim_sub = sim_sub_in.unchecked<3>();
    if (exp_sub.shape(0) != sim_sub.shape(0) ||
        exp_sub.shape(1) != sim_sub.shape(1) ||
        exp_sub.shape(2) != sim_sub.shape(2)) {
        throw std::runtime_error("exp_sub and sim_sub shape mismatch");
    }

    size_t bbox_nonzero_n = 0;
    double bbox_nonzero_sum = 0.0;
    double bbox_nonzero_sumsq = 0.0;
    size_t inside_n = 0;
    double inside_sum_a = 0.0;
    double inside_sumsq_a = 0.0;
    double sum_b = 0.0;
    double sumsq_b = 0.0;
    double sum_ab = 0.0;
    size_t finite_n = 0;
    double finite_sum_b = 0.0;
    double finite_sumsq_b = 0.0;
    double finite_sum_ab = 0.0;

    for (py::ssize_t z = 0; z < exp_sub.shape(0); ++z) {
        for (py::ssize_t y = 0; y < exp_sub.shape(1); ++y) {
            for (py::ssize_t x = 0; x < exp_sub.shape(2); ++x) {
                const double a = exp_sub(z, y, x);
                const double b = sim_sub(z, y, x);
                const bool finite = std::isfinite(a) && std::isfinite(b);
                if (!finite) {
                    continue;
                }
                ++finite_n;
                finite_sum_b += b;
                finite_sumsq_b += b * b;
                finite_sum_ab += a * b;
                if (a != 0.0) {
                    ++bbox_nonzero_n;
                    bbox_nonzero_sum += a;
                    bbox_nonzero_sumsq += a * a;
                }
                if (a != 0.0 || b != 0.0) {
                    ++inside_n;
                    inside_sum_a += a;
                    inside_sumsq_a += a * a;
                    sum_b += b;
                    sumsq_b += b * b;
                    sum_ab += a * b;
                }
            }
        }
    }

    size_t n = full_nonzero_n - bbox_nonzero_n + inside_n;
    double sum_a = full_nonzero_sum - bbox_nonzero_sum + inside_sum_a;
    double sumsq_a = full_nonzero_sumsq - bbox_nonzero_sumsq + inside_sumsq_a;
    if (n < 64) {
        n = full_finite_n;
        sum_a = full_finite_sum;
        sumsq_a = full_finite_sumsq;
        sum_b = finite_sum_b;
        sumsq_b = finite_sumsq_b;
        sum_ab = finite_sum_ab;
    }
    return ccc_from_sums_cpp(n, sum_a, sumsq_a, sum_b, sumsq_b, sum_ab);
}

py::array_t<double> compute_local_ccc_per_atom_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> coords_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> density_padded_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> origin_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> apix_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> kernel_centred_in,
    double kernel_norm,
    int radius_vox
) {
    auto coords = coords_in.unchecked<2>();
    auto density = density_padded_in.unchecked<3>();
    auto origin = origin_in.unchecked<1>();
    auto apix = apix_in.unchecked<1>();
    auto kernel = kernel_centred_in.unchecked<1>();
    const int k = (2 * radius_vox) + 1;
    const size_t patch_n = static_cast<size_t>(k * k * k);
    if (kernel.shape(0) != static_cast<py::ssize_t>(patch_n)) {
        throw std::runtime_error("kernel_centred length does not match radius");
    }
    const int nz = static_cast<int>(density.shape(0)) - (2 * radius_vox);
    const int ny = static_cast<int>(density.shape(1)) - (2 * radius_vox);
    const int nx = static_cast<int>(density.shape(2)) - (2 * radius_vox);
    py::array_t<double> out(static_cast<py::ssize_t>(coords.shape(0)));
    auto out_mut = out.mutable_unchecked<1>();

    for (py::ssize_t atom = 0; atom < coords.shape(0); ++atom) {
        int ix = static_cast<int>(std::nearbyint((coords(atom, 0) - origin(0)) / apix(0)));
        int iy = static_cast<int>(std::nearbyint((coords(atom, 1) - origin(1)) / apix(1)));
        int iz = static_cast<int>(std::nearbyint((coords(atom, 2) - origin(2)) / apix(2)));
        if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz || kernel_norm < 1e-12) {
            out_mut(atom) = 0.0;
            continue;
        }
        double sum_patch = 0.0;
        size_t idx = 0;
        for (int dz = 0; dz < k; ++dz) {
            for (int dy = 0; dy < k; ++dy) {
                for (int dx = 0; dx < k; ++dx) {
                    sum_patch += density(iz + dz, iy + dy, ix + dx);
                    ++idx;
                }
            }
        }
        const double mean_patch = sum_patch / static_cast<double>(patch_n);
        double numerator = 0.0;
        double norm_patch_sq = 0.0;
        idx = 0;
        for (int dz = 0; dz < k; ++dz) {
            for (int dy = 0; dy < k; ++dy) {
                for (int dx = 0; dx < k; ++dx) {
                    const double centred = density(iz + dz, iy + dy, ix + dx) - mean_patch;
                    numerator += centred * kernel(static_cast<py::ssize_t>(idx));
                    norm_patch_sq += centred * centred;
                    ++idx;
                }
            }
        }
        const double denom = std::sqrt(norm_patch_sq) * kernel_norm;
        if (denom < 1e-12) {
            out_mut(atom) = 0.0;
        } else {
            out_mut(atom) = std::max(-1.0, std::min(1.0, numerator / denom));
        }
    }
    return out;
}

// Pybind11 module
PYBIND11_MODULE(ligand_fitting, m) {
    m.def("compute_ccc_vector", &compute_ccc_vector, "Compute the normalized cross-correlation (CCC) between two vectors");
    m.def("compute_ccc_molecule_map", &compute_ccc_molecule_map, "Compute CCC between the expected and experimental densities",
          py::arg("atom_positions"),
          py::arg("atomic_masses"),
          py::arg("D_exp_flat"),
          py::arg("D_exp_dims"),
          py::arg("voxel_size"),
          py::arg("origin"),
          py::arg("sigma"));
    m.def("translate_and_rotate", &translate_and_rotate, "Rotate and translate atom positions");
    m.def("local_optimization", &local_optimization, "Perform local optimization",
          py::arg("atom_positions"),
          py::arg("atomic_masses"),
          py::arg("D_exp_flat"),
          py::arg("D_exp_dims"),
          py::arg("voxel_size"),
          py::arg("origin"),
          py::arg("sigma"),
          py::arg("initial_rotation"),
          py::arg("initial_translation"),
          py::arg("initial_step_size"),
          py::arg("max_steps"));
    m.def("global_search", &global_search, "Perform global search",
          py::arg("atom_positions"),
          py::arg("atomic_masses"),
          py::arg("D_exp_flat"),
          py::arg("D_exp_dims"),
          py::arg("voxel_size"),
          py::arg("origin"),
          py::arg("sigma"),
          py::arg("N"),
          py::arg("initial_step_size"),
          py::arg("max_steps"),
          py::arg("max_translation"));
    m.def("simulate_ligand_density_subgrid", &simulate_ligand_density_subgrid_cpp,
          "Simulate ligand density on a bounded subgrid",
          py::arg("coords"),
          py::arg("masses"),
          py::arg("origin"),
          py::arg("apix"),
          py::arg("bbox_lo_zyx"),
          py::arg("bbox_hi_zyx"),
          py::arg("sigma_A"),
          py::arg("normalise") = true);
    m.def("compute_ligand_ccc_decomposed", &compute_ligand_ccc_decomposed_cpp,
          "Compute truncated full-grid ligand CCC from subgrid statistics",
          py::arg("exp_subgrid"),
          py::arg("sim_subgrid"),
          py::arg("full_nonzero_n"),
          py::arg("full_nonzero_sum"),
          py::arg("full_nonzero_sumsq"),
          py::arg("full_finite_n"),
          py::arg("full_finite_sum"),
          py::arg("full_finite_sumsq"));
    m.def("compute_local_ccc_per_atom", &compute_local_ccc_per_atom_cpp,
          "Compute local CCC for each atom from a padded density map",
          py::arg("coords"),
          py::arg("density_padded"),
          py::arg("origin"),
          py::arg("apix"),
          py::arg("kernel_centred"),
          py::arg("kernel_norm"),
          py::arg("radius_vox"));
}
