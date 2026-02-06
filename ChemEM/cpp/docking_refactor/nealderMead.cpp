// NelderMeadOptimizer.cpp
#include "nealderMead.h"
#include <cassert>
#include <cmath>
#include <limits>

NelderMeadOptimizer::NelderMeadOptimizer(size_t dimension,
                                         double alpha,
                                         double gamma,
                                         double rho,
                                         double sigma)
    : dim(dimension), alpha(alpha), gamma(gamma), rho(rho), sigma(sigma),
      reflect_point(dimension), expand_point(dimension),
      contract_point(dimension), centroid_point(dimension),
      fvals(dimension + 1)
{
    assert(dim >= 1);
}

NelderMeadOptimizer::Result
NelderMeadOptimizer::optimize(const std::function<double(const Point&)>& func,
                               std::vector<Point>& simplex,
                               int max_iters,
                               double ftol,
                               double xtol) {
    assert(simplex.size() == dim + 1);
    for (auto &p : simplex) assert(p.size() == dim);

    for (size_t i = 0; i < simplex.size(); ++i)
        fvals[i] = func(simplex[i]);

    Result res;
    res.iterations = 0;
    res.converged = false;

    double xtol_sq = xtol * xtol;
    size_t best_idx, worst_idx, second_worst_idx;

    int iter;
    for (iter = 0; iter < max_iters; ++iter) {
        // Identify best, worst, second-worst
        best_idx = worst_idx = 0;
        for (size_t i = 1; i < simplex.size(); ++i) {
            if (fvals[i] < fvals[best_idx]) best_idx = i;
            if (fvals[i] > fvals[worst_idx]) worst_idx = i;
        }
        second_worst_idx = (worst_idx == 0 ? 1 : 0);
        for (size_t i = 0; i < simplex.size(); ++i) {
            if (i == worst_idx) continue;
            if (fvals[i] > fvals[second_worst_idx]) second_worst_idx = i;
        }

        double f_range = fvals[worst_idx] - fvals[best_idx];
        const Point &best_pt = simplex[best_idx];
        double max_dist_sq = 0.0;
        for (size_t i = 0; i < simplex.size(); ++i) {
            if (i == best_idx) continue;
            double d2 = 0.0;
            for (size_t j = 0; j < dim; ++j) {
                double d = simplex[i][j] - best_pt[j];
                d2 += d * d;
            }
            if (d2 > max_dist_sq) max_dist_sq = d2;
        }
        //if (f_range <= ftol || max_dist_sq <= xtol_sq) {
        if (f_range <= ftol && max_dist_sq <= xtol_sq) {
            res.converged = true;
            break;
        }

        // Centroid
        for (size_t j = 0; j < dim; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < simplex.size(); ++i) {
                if (i == worst_idx) continue;
                sum += simplex[i][j];
            }
            centroid_point[j] = sum / static_cast<double>(dim);
        }

        // Reflection
        for (size_t j = 0; j < dim; ++j) {
            reflect_point[j] = centroid_point[j] + alpha * (centroid_point[j] - simplex[worst_idx][j]);
            if (reflect_point[j] < 0.0) reflect_point[j] = 0.0;
            if (reflect_point[j] > 1.0) reflect_point[j] = 1.0;
        }
        double f_r = func(reflect_point);

        if (f_r < fvals[best_idx]) {
            // Expansion
            for (size_t j = 0; j < dim; ++j) {
                expand_point[j] = centroid_point[j] + gamma * (reflect_point[j] - centroid_point[j]);
                if (expand_point[j] < 0.0) expand_point[j] = 0.0;
                if (expand_point[j] > 1.0) expand_point[j] = 1.0;
            }
            double f_e = func(expand_point);
            if (f_e < f_r) {
                simplex[worst_idx] = expand_point;
                fvals[worst_idx]   = f_e;
            } else {
                simplex[worst_idx] = reflect_point;
                fvals[worst_idx]   = f_r;
            }
        }
        else if (f_r < fvals[second_worst_idx]) {
            simplex[worst_idx] = reflect_point;
            fvals[worst_idx]   = f_r;
        }
        else {
            bool outside = (f_r < fvals[worst_idx]);
            for (size_t j = 0; j < dim; ++j) {
                if (outside) {
                    contract_point[j] = centroid_point[j] + rho * (reflect_point[j] - centroid_point[j]);
                } else {
                    contract_point[j] = centroid_point[j] + rho * (simplex[worst_idx][j] - centroid_point[j]);
                }
                if (contract_point[j] < 0.0) contract_point[j] = 0.0;
                if (contract_point[j] > 1.0) contract_point[j] = 1.0;
            }
            double f_c = func(contract_point);
            if (f_c < (outside ? f_r : fvals[worst_idx])) {
                simplex[worst_idx] = contract_point;
                fvals[worst_idx]   = f_c;
            } else {
                // Shrink
                Point best_copy = simplex[best_idx];
                for (size_t i = 0; i < simplex.size(); ++i) {
                    if (i == best_idx) continue;
                    for (size_t j = 0; j < dim; ++j) {
                        simplex[i][j] = best_copy[j] + sigma * (simplex[i][j] - best_copy[j]);
                        if (simplex[i][j] < 0.0) simplex[i][j] = 0.0;
                        if (simplex[i][j] > 1.0) simplex[i][j] = 1.0;
                    }
                    fvals[i] = func(simplex[i]);
                }
            }
        }
        res.iterations = iter + 1;
    }

    // Final best
    best_idx = 0;
    for (size_t i = 1; i < simplex.size(); ++i) {
        if (fvals[i] < fvals[best_idx]) best_idx = i;
    }
    res.best_point = simplex[best_idx];
    res.best_value = fvals[best_idx];
    res.iterations = iter;
    return res;
}