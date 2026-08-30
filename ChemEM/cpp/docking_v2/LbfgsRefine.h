// LbfgsRefine.h — self-contained projected (box-constrained) L-BFGS with central
// finite-difference gradients, for the docking_v2 experimental engine (improvement R1).
//
// Why this exists: the baseline engine refines poses with derivative-free Nelder-Mead,
// which needs many hundreds of iterations x (D+1) simplex evaluations to reach a tight
// minimum. A quasi-Newton (L-BFGS) step reaches a comparable-or-tighter minimum in far
// fewer evaluations, which is what makes "refine many diverse basins" (R2) affordable.
//
// The ECHO scorer is treated as a black box (exactly as Nelder-Mead treats it): the
// gradient is estimated by central finite differences on the SAME normalized objective
// closure the NM path uses. Variables are box-constrained to [0,1] (the normalized
// pose box); we use projected L-BFGS (two-loop direction + projected Armijo backtracking),
// which is simpler than full L-BFGS-B and robust for these low dimensions (D = 6 + nTors).
//
// FD-step note: the ECHO score has trilinear-grid interpolation + capped clash + hard
// distance/angle gates, so too-small a step reads interpolation noise. Default h = 1e-2
// in normalized units is deliberately not tiny. If a step yields a non-descent direction
// (noisy curvature), we discard the offending (s,y) pair to keep the Hessian model PD.

#pragma once

#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <functional>
#include <cstddef>

namespace lbfgs_refine {

struct LbfgsResult {
    std::vector<double> x;   // best point found (in [0,1]^D)
    double f;                // objective value at x
    int    iters;            // outer iterations performed
    int    nfev;             // objective evaluations used (incl. FD gradient)
    bool   converged;        // hit a tolerance (vs. exhausted max_iters)
};

struct LbfgsParams {
    int    max_iters   = 60;     // outer iterations
    int    history     = 8;      // L-BFGS memory (# stored s,y pairs)
    double fd_h        = 1e-2;   // central finite-difference step (normalized units)
    double gtol        = 1e-4;   // stop when max|projected gradient| < gtol
    double ftol        = 1e-7;   // stop when |f_prev - f| <= ftol*(1+|f|)
    double xtol        = 1e-6;   // stop when step inf-norm < xtol
    double c1          = 1e-4;   // Armijo sufficient-decrease constant
    double ls_shrink   = 0.5;    // backtracking factor
    int    ls_max      = 20;     // max backtracking steps
    double lb          = 0.0;    // box lower bound (all coords)
    double ub          = 1.0;    // box upper bound (all coords)
};

// Objective: any callable double(const std::vector<double>&). It may clamp internally;
// we also keep iterates inside [lb,ub] so it never sees out-of-box points.
template <class Func>
inline LbfgsResult minimize_box(Func&& obj, std::vector<double> x, const LbfgsParams& p = LbfgsParams()) {
    const std::size_t D = x.size();
    LbfgsResult R;
    R.iters = 0;
    R.nfev = 0;
    R.converged = false;
    if (D == 0) { R.x = x; R.f = 0.0; return R; }

    auto clampbox = [&](std::vector<double>& v) {
        for (double& c : v) c = std::min(p.ub, std::max(p.lb, c));
    };
    clampbox(x);

    auto feval = [&](const std::vector<double>& v) -> double {
        ++R.nfev;
        return obj(v);
    };

    // Central finite-difference gradient, one-sided near a bound so probes stay in-box.
    auto grad_fd = [&](const std::vector<double>& v, std::vector<double>& g) {
        std::vector<double> vp = v;
        for (std::size_t i = 0; i < D; ++i) {
            const double xi = v[i];
            double hi = p.fd_h;
            double xph = xi + hi, xmh = xi - hi;
            bool hi_ok = (xph <= p.ub), lo_ok = (xmh >= p.lb);
            double fph, fmh, denom;
            if (hi_ok && lo_ok) {
                vp[i] = xph; fph = feval(vp);
                vp[i] = xmh; fmh = feval(vp);
                denom = 2.0 * hi;
            } else if (hi_ok) {                 // forward diff at the lower bound
                vp[i] = xph;        fph = feval(vp);
                vp[i] = xi;         fmh = feval(vp);
                denom = hi;
            } else if (lo_ok) {                 // backward diff at the upper bound
                vp[i] = xi;         fph = feval(vp);
                vp[i] = xmh;        fmh = feval(vp);
                denom = hi;
            } else {                            // box narrower than h (degenerate): flat
                fph = fmh = 0.0; denom = 1.0;
            }
            g[i] = (fph - fmh) / denom;
            vp[i] = xi;  // restore
        }
    };

    // Projected-gradient inf-norm: gradient component ignored where it pushes into a bound.
    auto proj_grad_inf = [&](const std::vector<double>& v, const std::vector<double>& g) -> double {
        double m = 0.0;
        for (std::size_t i = 0; i < D; ++i) {
            double gi = g[i];
            if (v[i] <= p.lb && gi > 0.0) gi = 0.0;
            if (v[i] >= p.ub && gi < 0.0) gi = 0.0;
            m = std::max(m, std::fabs(gi));
        }
        return m;
    };

    std::vector<double> g(D, 0.0), g_new(D, 0.0), d(D, 0.0), x_new(D, 0.0);
    std::deque<std::vector<double>> S, Y;   // s = x_new - x, y = g_new - g
    std::deque<double> RHO;                 // 1 / (y·s)

    double f = feval(x);
    grad_fd(x, g);
    R.f = f;
    R.x = x;

    for (int it = 0; it < p.max_iters; ++it) {
        R.iters = it + 1;

        if (proj_grad_inf(x, g) < p.gtol) { R.converged = true; break; }

        // ---- L-BFGS two-loop recursion: d = -H_k * g ----
        d.assign(D, 0.0);
        for (std::size_t i = 0; i < D; ++i) d[i] = -g[i];
        const int m = static_cast<int>(S.size());
        std::vector<double> alpha(m, 0.0);
        for (int k = m - 1; k >= 0; --k) {
            double si_di = 0.0;
            for (std::size_t i = 0; i < D; ++i) si_di += S[k][i] * d[i];
            alpha[k] = RHO[k] * si_di;
            for (std::size_t i = 0; i < D; ++i) d[i] -= alpha[k] * Y[k][i];
        }
        // initial Hessian scaling gamma = (s·y)/(y·y) from the most recent pair
        if (m > 0) {
            double sy = 0.0, yy = 0.0;
            for (std::size_t i = 0; i < D; ++i) { sy += S[m-1][i]*Y[m-1][i]; yy += Y[m-1][i]*Y[m-1][i]; }
            const double gamma = (yy > 0.0) ? (sy / yy) : 1.0;
            for (std::size_t i = 0; i < D; ++i) d[i] *= gamma;
        }
        for (int k = 0; k < m; ++k) {
            double yi_di = 0.0;
            for (std::size_t i = 0; i < D; ++i) yi_di += Y[k][i] * d[i];
            const double beta = RHO[k] * yi_di;
            for (std::size_t i = 0; i < D; ++i) d[i] += (alpha[k] - beta) * S[k][i];
        }

        // Ensure a descent direction; if the model produced an ascent dir (FD noise), reset to -g.
        double gTd = 0.0;
        for (std::size_t i = 0; i < D; ++i) gTd += g[i] * d[i];
        if (!(gTd < 0.0)) {
            for (std::size_t i = 0; i < D; ++i) d[i] = -g[i];
            gTd = 0.0;
            for (std::size_t i = 0; i < D; ++i) gTd += g[i] * d[i];
            S.clear(); Y.clear(); RHO.clear();
        }

        // ---- projected Armijo backtracking line search ----
        double t = 1.0;
        double f_new = f;
        bool ls_ok = false;
        for (int ls = 0; ls < p.ls_max; ++ls) {
            for (std::size_t i = 0; i < D; ++i)
                x_new[i] = std::min(p.ub, std::max(p.lb, x[i] + t * d[i]));
            f_new = feval(x_new);
            // sufficient decrease along the actually-taken (projected) step
            double gTstep = 0.0;
            for (std::size_t i = 0; i < D; ++i) gTstep += g[i] * (x_new[i] - x[i]);
            if (f_new <= f + p.c1 * gTstep) { ls_ok = true; break; }
            t *= p.ls_shrink;
        }
        if (!ls_ok) {
            // Line search failed (flat/noisy region) — accept only if it didn't increase f.
            if (!(f_new < f)) { R.converged = true; break; }
        }

        // step / function-value convergence
        double step_inf = 0.0, dx;
        for (std::size_t i = 0; i < D; ++i) { dx = std::fabs(x_new[i] - x[i]); step_inf = std::max(step_inf, dx); }
        const double f_prev = f;

        grad_fd(x_new, g_new);

        // curvature pair (s,y); keep only if y·s > 0 (positive-definite model)
        std::vector<double> s(D), y(D);
        double ys = 0.0;
        for (std::size_t i = 0; i < D; ++i) { s[i] = x_new[i] - x[i]; y[i] = g_new[i] - g[i]; ys += y[i]*s[i]; }
        if (ys > 1e-12) {
            S.push_back(std::move(s));
            Y.push_back(std::move(y));
            RHO.push_back(1.0 / ys);
            if (static_cast<int>(S.size()) > p.history) { S.pop_front(); Y.pop_front(); RHO.pop_front(); }
        }

        x = x_new; f = f_new; g = g_new;
        if (f < R.f) { R.f = f; R.x = x; }

        if (step_inf < p.xtol) { R.converged = true; break; }
        if (std::fabs(f_prev - f) <= p.ftol * (1.0 + std::fabs(f))) { R.converged = true; break; }
    }

    // R.x / R.f already track the best seen
    return R;
}

} // namespace lbfgs_refine
