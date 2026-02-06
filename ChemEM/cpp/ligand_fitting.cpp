#include "ligand_fitting.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <random>
#include <cmath>
#include <tuple>

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
}
