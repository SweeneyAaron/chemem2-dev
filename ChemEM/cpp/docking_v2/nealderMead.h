// NelderMeadOptimizer.hpp
#ifndef NELDER_MEAD_OPTIMIZER_HPP
#define NELDER_MEAD_OPTIMIZER_HPP

#include <vector>
#include <functional>

class NelderMeadOptimizer {
public:
    using Point = std::vector<double>;
    struct Result {
        Point  best_point;
        double best_value;
        int    iterations;
        bool   converged;
    };

    NelderMeadOptimizer(size_t dimension,
                        double alpha = 1.0,
                        double gamma = 2.0,
                        double rho   = 0.5,
                        double sigma = 0.5);

    Result optimize(const std::function<double(const Point&)>& func,
                    std::vector<Point>& simplex,
                    int max_iters,
                    double ftol,
                    double xtol);

private:
    size_t dim;
    double alpha, gamma, rho, sigma;
    Point reflect_point, expand_point, contract_point, centroid_point;
    std::vector<double> fvals;
};

#endif // NELDER_MEAD_OPTIMIZER_HPP

