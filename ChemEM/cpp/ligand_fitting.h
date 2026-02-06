#ifndef LIGAND_FITTING_H
#define LIGAND_FITTING_H

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <tuple>

// Function declarations
double compute_ccc_vector(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2);
double compute_ccc_molecule_map(
    const std::vector<Eigen::Vector3d>& atom_positions,
    const std::vector<double>& atomic_masses,
    const std::vector<double>& D_exp_flat,    // Flat array
    const std::vector<size_t>& D_exp_dims,    // Dimensions
    double voxel_size,
    const Eigen::Vector3d& origin,
    double sigma
);
std::vector<Eigen::Vector3d> translate_and_rotate(
    const Eigen::VectorXd& params,
    const std::vector<Eigen::Vector3d>& atom_positions
);
std::tuple<double, std::vector<Eigen::Vector3d>> local_optimization(
    const std::vector<Eigen::Vector3d>& atom_positions,
    const std::vector<double>& atomic_masses,
    const std::vector<double>& D_exp_flat,    // Flat array
    const std::vector<size_t>& D_exp_dims,    // Dimensions
    double voxel_size,
    const Eigen::Vector3d& origin,
    double sigma,
    const Eigen::Quaterniond& initial_rotation,
    const Eigen::Vector3d& initial_translation,
    double initial_step_size,
    int max_steps
);
Eigen::Quaterniond random_rotation(double max_angle);
std::tuple<double, std::vector<Eigen::Vector3d>, std::vector<double>> global_search(
    const std::vector<Eigen::Vector3d>& atom_positions,
    const std::vector<double>& atomic_masses,
    const std::vector<double>& D_exp_flat,    // Flat array
    const std::vector<size_t>& D_exp_dims,    // Dimensions
    double voxel_size,
    const Eigen::Vector3d& origin,
    double sigma,
    int N,
    double initial_step_size,
    int max_steps,
    double max_translation
);

#endif // LIGAND_FITTING_H
