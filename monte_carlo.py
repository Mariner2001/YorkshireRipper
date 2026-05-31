"""
monte_carlo.py
==============
Monte Carlo simulation study for:
  "Statistical foundations of Rossmo's formula with an application
   to the Yorkshire Ripper case"
  Manuel Ubeda-Flores, University of Almeria

Reproduces Table 3 and Figure 1B of the paper.

Dependencies: numpy, scipy, matplotlib
Usage:        python monte_carlo.py
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# ---------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------
SEED = 2024
np.random.seed(SEED)

# ---------------------------------------------------------------
# Model parameters (generating distribution, eq. 2 of paper)
# ---------------------------------------------------------------
ALPHA_TRUE = 2.0   # distance-decay scale
B_TRUE     = 0.5   # buffer radius
Z_STAR     = np.array([0.0, 0.0])   # true anchor point

# Estimator parameters
BETA_V = 5.0   # buffer softness for variational estimator

# Simulation parameters
NS      = [5, 10, 20, 50, 100, 200, 500]   # sample sizes
R       = 1000                              # replications per n
N_RESTARTS = 3                             # Nelder-Mead restarts

# Rossmo parameters
F_R, G_R = 1.2, 1.2   # decay exponents

# Grid for Rossmo argmax search
NX_GRID, NY_GRID = 60, 60
GRID_MARGIN      = 15.0


# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------

def sample_crimes(n, alpha, B, z_star=Z_STAR, rng=None):
    """
    Draw n crime sites from the 2-D generating density (eq. 2):
      f(x) propto exp(-||x-z*||/alpha) * 1{||x-z*|| > B}
    Uses the fact that R = ||X-z*|| has density
      p(r) propto r * exp(-r/alpha) for r > B.
    We sample R via rejection: draw R ~ Exp(alpha)+B (shifted
    exponential), accept; then draw direction uniformly on S^1.
    """
    if rng is None:
        rng = np.random
    samples = []
    while len(samples) < n:
        r     = rng.exponential(scale=alpha) + B
        theta = rng.uniform(0, 2 * np.pi)
        samples.append(z_star + r * np.array([np.cos(theta), np.sin(theta)]))
    return np.array(samples[:n])


def variational_energy(z, crimes, alpha, beta, B):
    """
    Regularised energy (eq. 3 of paper):
      E(z) = (1/n) sum_i [ ||x_i - z||/alpha + beta*(B - ||x_i-z||)_+^2 ]
    """
    d = np.linalg.norm(crimes - z, axis=1)
    return np.mean(d / alpha + beta * np.maximum(B - d, 0) ** 2)


def variational_estimate(crimes, alpha, beta, B, n_restarts=N_RESTARTS):
    """
    Minimise the variational energy using Nelder-Mead with multiple restarts.
    Returns the estimated anchor point.
    """
    best_val, best_z = np.inf, None
    for _ in range(n_restarts):
        z0  = np.random.uniform(-4, 4, 2)
        res = minimize(
            variational_energy, z0,
            args=(crimes, alpha, beta, B),
            method='Nelder-Mead',
            options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 20000},
        )
        if res.fun < best_val:
            best_val, best_z = res.fun, res.x.copy()
    return best_z


def build_grid(margin=GRID_MARGIN, nx=NX_GRID, ny=NY_GRID):
    """Build a flat grid of candidate locations."""
    xs = np.linspace(-margin, margin, nx)
    ys = np.linspace(-margin, margin, ny)
    GX, GY = np.meshgrid(xs, ys)
    return np.stack([GX.ravel(), GY.ravel()], axis=1)


def rossmo_scores(grid, crimes, f=F_R, g=G_R, B=B_TRUE):
    """
    Evaluate Rossmo's CGT formula (eq. 1) on a grid of candidate points.
    Uses Euclidean distance (consistent with the paper's theoretical framework).

    Parameters
    ----------
    grid   : (M, 2) array of candidate locations
    crimes : (n, 2) array of crime sites
    Returns
    -------
    scores : (M,) array
    """
    scores = np.zeros(len(grid))
    for xi in crimes:
        d   = np.linalg.norm(grid - xi, axis=1)
        out = d > B
        ins = (~out) & (d > 0)
        # Outside buffer: 1/d^f
        scores[out] += 1.0 / d[out] ** f
        # Inside buffer: B^(g-f) / (2B-d)^g
        denom = 2 * B - d[ins]
        good  = denom > 1e-10
        tmp   = np.zeros(ins.sum())
        tmp[good] = B ** (g - f) / denom[good] ** g
        scores[ins] += tmp
    return scores


def rossmo_estimate(crimes, grid):
    """Return the grid point with the highest Rossmo score."""
    sc = rossmo_scores(grid, crimes)
    return grid[sc.argmax()].copy()


# ---------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------

def run_simulation(ns=NS, R=R, seed=SEED):
    np.random.seed(seed)
    grid = build_grid()

    results_var  = {}
    results_ross = {}

    print(f"Monte Carlo simulation: R={R} replications")
    print(f"Parameters: alpha={ALPHA_TRUE}, B={B_TRUE}, beta={BETA_V}")
    print(f"{'n':>6}  {'RMSE_var':>10}  {'RMSE_ross':>11}  "
          f"{'n^-0.5':>8}  {'Factor':>8}  {'SE_var':>9}")
    print("-" * 65)

    for n in ns:
        t0       = time.time()
        errs_v   = []
        errs_r   = []

        for _ in range(R):
            crimes = sample_crimes(n, ALPHA_TRUE, B_TRUE)

            # Variational estimator
            z_hat_v = variational_estimate(
                crimes, ALPHA_TRUE, BETA_V, B_TRUE
            )
            errs_v.append(np.linalg.norm(z_hat_v - Z_STAR))

            # Rossmo estimator (only for n <= 50)
            if n <= 50:
                z_hat_r = rossmo_estimate(crimes, grid)
                errs_r.append(np.linalg.norm(z_hat_r - Z_STAR))

        rmse_v = np.sqrt(np.mean(np.array(errs_v) ** 2))
        se_v   = rmse_v / np.sqrt(2 * R)
        results_var[n] = rmse_v

        if n <= 50:
            rmse_r = np.sqrt(np.mean(np.array(errs_r) ** 2))
            results_ross[n] = rmse_r
            factor   = rmse_r / rmse_v
            elapsed  = time.time() - t0
            print(f"{n:>6}  {rmse_v:>10.4f}  {rmse_r:>11.4f}  "
                  f"{n**-0.5:>8.4f}  {factor:>8.2f}  {se_v:>9.5f}  [{elapsed:.1f}s]")
        else:
            elapsed = time.time() - t0
            print(f"{n:>6}  {rmse_v:>10.4f}  {'---':>11}  "
                  f"{n**-0.5:>8.4f}  {'---':>8}  {se_v:>9.5f}  [{elapsed:.1f}s]")

    # Log-log slope
    log_ns   = np.log(ns)
    log_rmse = np.log([results_var[n] for n in ns])
    slope, _ = np.polyfit(log_ns, log_rmse, 1)
    print(f"\nEmpirical log-log slope: {slope:.4f}  (theoretical: -0.500)")

    return results_var, results_ross, slope


# ---------------------------------------------------------------
# Plot (reproduces Figure 1B)
# ---------------------------------------------------------------

def plot_convergence(results_var, results_ross, slope, outfile='figure1B.pdf'):
    ns_var  = sorted(results_var.keys())
    ns_ross = sorted(results_ross.keys())
    rmse_v  = [results_var[n]  for n in ns_var]
    rmse_r  = [results_ross[n] for n in ns_ross]
    ns_arr  = np.array(ns_var, dtype=float)
    ref     = 0.55 * ns_arr ** (-0.5)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.loglog(ns_var,  rmse_v, 'bo-', ms=7, lw=1.8,
              label=f'Variational (slope {slope:.3f})')
    ax.loglog(ns_ross, rmse_r, 'rs--', ms=7, lw=1.8,
              label="Rossmo's CGT")
    ax.loglog(ns_arr,  ref,   'k:',   lw=1.5,
              label='$O(n^{-1/2})$ reference')
    ax.set_xlabel('Number of crime sites $n$', fontsize=10)
    ax.set_ylabel('RMSE (km)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.25)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight', dpi=200)
    print(f"Figure saved to {outfile}")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == '__main__':
    results_var, results_ross, slope = run_simulation()
    plot_convergence(results_var, results_ross, slope)
