"""
yorkshire_ripper.py
===================
Case study replication for:
  "Statistical foundations of Rossmo's formula with an application
   to the Yorkshire Ripper case"
  Manuel Ubeda-Flores, University of Almeria

Reproduces Table 2, Figure 1A, and Figure 2 of the paper.

Dependencies: numpy, scipy, matplotlib
Usage:        python yorkshire_ripper.py
"""

import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------
# Crime-site data  (Table 1 of the paper)
# Local Cartesian coordinates (km) centred on Bradford city centre
# (53.796 N, 1.759 W).  Positive x = east, positive y = north.
# Source: Byford Report (Home Office, 1981) and Bilton (2003).
# ---------------------------------------------------------------

VICTIMS = [
    "Wilma McCann",       "Emily Jackson",     "Irene Richardson",
    "Patricia Atkinson",  "Jayne McDonald",    "Jean Jordan",
    "Yvonne Pearson",     "Helen Rytka",       "Vera Millward",
    "Josephine Whittaker","Barbara Leach",     "Marguerite Walls",
    "Jacqueline Hill",
]

CRIMES = np.array([
    [+15.9, +2.9],   # McCann       — Leeds (Potternewton)
    [+13.7, -1.3],   # Jackson      — Leeds (Holbeck)
    [+17.1, +4.3],   # Richardson   — Leeds (Roundhay)
    [ -0.3, +0.9],   # Atkinson     — Bradford (Manningham)
    [+14.9, +1.7],   # McDonald     — Leeds (Chapeltown)
    [-32.4,-40.2],   # Jordan       — Manchester (S.)
    [ +0.7, -0.6],   # Pearson      — Bradford (C.)
    [ -1.5,-16.0],   # Rytka        — Huddersfield
    [-30.8,-37.0],   # Millward     — Manchester (C.)
2    [ -7.3, -9.0],   # Whittaker    — Halifax
    [ -0.1,  0.0],   # Leach        — Bradford (U.)
    [ +5.9, +1.7],   # Walls        — Leeds (Farsley)
    [+12.0, +2.6],   # Hill         — Leeds (Headingley)
])

# Benchmark anchor point: 6 Garden Lane, Heaton, Bradford BD9 5QJ
ANCHOR_HEATON  = np.array([-2.3, +1.8])
# Secondary anchor: Tanton Crescent, Clayton (first 5 murders)
ANCHOR_CLAYTON = np.array([-3.7, -1.7])

# ---------------------------------------------------------------
# Estimator parameters
# ---------------------------------------------------------------
ALPHA = 8.0   # distance-decay scale (km)
BETA  = 3.0   # buffer softness
B     = 2.0   # buffer radius (km)
F_R   = 1.2   # Rossmo decay exponent f
G_R   = 1.2   # Rossmo decay exponent g

# Grid for Rossmo and HS% computation
NX, NY   = 150, 150
N_STARTS = 50   # Nelder-Mead restarts for variational estimator


# ---------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------

def variational_energy(z, crimes, alpha, beta, B):
    d = np.linalg.norm(crimes - z, axis=1)
    return np.mean(d / alpha + beta * np.maximum(B - d, 0) ** 2)


def variational_estimate(crimes, alpha, beta, B, n_starts=N_STARTS, seed=42):
    rng = np.random.default_rng(seed)
    xmin, xmax = crimes[:,0].min()-5, crimes[:,0].max()+5
    ymin, ymax = crimes[:,1].min()-5, crimes[:,1].max()+5
    best_val, best_z = np.inf, None
    for _ in range(n_starts):
        z0  = rng.uniform([xmin, ymin], [xmax, ymax])
        res = minimize(variational_energy, z0,
                       args=(crimes, alpha, beta, B),
                       method='Nelder-Mead',
                       options={'xatol':1e-10,'fatol':1e-12,'maxiter':200000})
        if res.fun < best_val:
            best_val, best_z = res.fun, res.x.copy()
    return best_z


def rossmo_grid(grid, crimes, f=F_R, g=G_R, B=B):
    scores = np.zeros(len(grid))
    for xi in crimes:
        d   = np.linalg.norm(grid - xi, axis=1)
        out = d > B
        ins = (~out) & (d > 0)
        scores[out] += 1.0 / d[out] ** f
        denom = 2*B - d[ins]
        good  = denom > 1e-10
        tmp   = np.zeros(ins.sum())
        tmp[good] = B**(g-f) / denom[good]**g
        scores[ins] += tmp
    return scores


def hit_score_pct(score_flat, anchor, xs, ys, NX):
    """Proportion of grid cells with score >= anchor cell's score."""
    ix = np.argmin(np.abs(xs - anchor[0]))
    iy = np.argmin(np.abs(ys - anchor[1]))
    anchor_score = score_flat[iy * NX + ix]
    rank = (score_flat >= anchor_score).sum()
    return rank / len(score_flat) * 100.0


def confidence_radius(alpha, B, n, coverage=0.95):
    """95% joint confidence radius for the anchor-point estimate."""
    from scipy.stats import chi2
    q = chi2.ppf(coverage, df=2)
    return np.sqrt(2 * q) * (alpha + B) / np.sqrt(n)


# ---------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------

def run_case_study():
    n = len(CRIMES)
    xmin, xmax = CRIMES[:,0].min()-5, CRIMES[:,0].max()+5
    ymin, ymax = CRIMES[:,1].min()-5, CRIMES[:,1].max()+5
    xs = np.linspace(xmin, xmax, NX)
    ys = np.linspace(ymin, ymax, NY)
    GX, GY = np.meshgrid(xs, ys)
    grid_flat = np.stack([GX.ravel(), GY.ravel()], axis=1)

    # 1. Centroid
    centroid = CRIMES.mean(axis=0)

    # 2. Rossmo estimate
    print("Computing Rossmo surface...")
    ross_surf = rossmo_grid(grid_flat, CRIMES).reshape(NY, NX)
    idx       = ross_surf.argmax()
    iy0, ix0  = np.unravel_index(idx, (NY, NX))
    rossmo_est = np.array([xs[ix0], ys[iy0]])

    # 3. Variational estimate
    print("Optimising variational estimator (50 random starts)...")
    var_est = variational_estimate(CRIMES, ALPHA, BETA, B)

    # 4. Error distances (from Heaton anchor)
    err_rossmo   = np.linalg.norm(rossmo_est  - ANCHOR_HEATON)
    err_var      = np.linalg.norm(var_est     - ANCHOR_HEATON)
    err_centroid = np.linalg.norm(centroid    - ANCHOR_HEATON)

    # 5. Hit score percentages
    ross_flat = ross_surf.ravel()
    var_surf = np.array([
    -variational_energy(pt, CRIMES, ALPHA, BETA, B)
    for pt in grid_flat
    ])
    # Compute variational surface on grid
    var_surf = np.array([
        -variational_energy(pt, CRIMES, ALPHA, BETA, B)
        for pt in grid_flat
    ])
    cent_surf = -np.linalg.norm(grid_flat - centroid, axis=1)

    hs_rossmo   = hit_score_pct(ross_flat,  ANCHOR_HEATON, xs, ys, NX)
    hs_var      = hit_score_pct(var_surf,   ANCHOR_HEATON, xs, ys, NX)
    hs_centroid = hit_score_pct(cent_surf,  ANCHOR_HEATON, xs, ys, NX)

    # 6. Confidence radius
    r95 = confidence_radius(ALPHA, B, n)

    # 7. Anchor sensitivity
    err_rossmo_c   = np.linalg.norm(rossmo_est  - ANCHOR_CLAYTON)
    err_var_c      = np.linalg.norm(var_est     - ANCHOR_CLAYTON)
    err_centroid_c = np.linalg.norm(centroid    - ANCHOR_CLAYTON)

    # Print results
    print("\n" + "="*65)
    print("TABLE 2: Anchor-point estimation results (Heaton anchor)")
    print("="*65)
    print(f"{'Method':<25} {'Error (km)':>11} {'HS%':>8}")
    print("-"*45)
    print(f"{'Rossmo CGT':<25} {err_rossmo:>11.1f} {hs_rossmo:>8.1f}")
    print(f"{'Variational':<25} {err_var:>11.1f} {hs_var:>8.1f}")
    print(f"{'Centroid':<25} {err_centroid:>11.1f} {hs_centroid:>8.1f}")
    print(f"\n95% confidence radius: {r95:.1f} km")
    print(f"True anchor within CI: {err_var < r95}")
    print(f"\nAnchor sensitivity (Clayton):")
    print(f"  Rossmo:       {err_rossmo_c:.1f} km")
    print(f"  Variational:  {err_var_c:.1f} km")
    print(f"  Centroid:     {err_centroid_c:.1f} km")

    return {
        'rossmo_est': rossmo_est, 'var_est': var_est, 'centroid': centroid,
        'err_rossmo': err_rossmo, 'err_var': err_var, 'err_centroid': err_centroid,
        'hs_rossmo': hs_rossmo, 'hs_var': hs_var, 'hs_centroid': hs_centroid,
        'ross_surf': ross_surf, 'var_surf': var_surf.reshape(NY, NX),
        'xs': xs, 'ys': ys, 'GX': GX, 'GY': GY, 'r95': r95,
    }


def make_figures(res):
    """Reproduce Figure 1A and Figure 2 of the paper."""
    import scipy.stats as spstats

    # ---- Figure 1A ----
    cmap = LinearSegmentedColormap.from_list('heatmap',
        ['#ffffff','#fffacd','#ffd700','#ff8c00','#dc143c','#8b0000'])
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.contourf(res['GX'], res['GY'], res['ross_surf'],
                levels=50, cmap=cmap, alpha=0.85)
    ax.scatter(CRIMES[:,0], CRIMES[:,1], c='navy', s=60, zorder=5,
               label='Crime sites')
    for i,(x,y) in enumerate(CRIMES):
        ax.annotate(VICTIMS[i].split()[1], (x,y),
                    textcoords='offset points', xytext=(3,3),
                    fontsize=6, color='navy')
    ax.scatter(*ANCHOR_HEATON,  c='green',     s=220, marker='*', zorder=7,
               label='True anchor (Heaton)')
    ax.scatter(*ANCHOR_CLAYTON, c='darkgreen', s=110, marker='P', zorder=7,
               label='Clayton anchor')
    ax.scatter(*res['rossmo_est'], c='red',    s=130, marker='X', zorder=8,
               label=f"Rossmo ({res['err_rossmo']:.1f} km)")
    ax.scatter(*res['var_est'],    c='blue',   s=130, marker='+',
               linewidths=2.5, zorder=8,
               label=f"Variational ({res['err_var']:.1f} km)")
    ax.scatter(*res['centroid'],   c='orange', s=130, marker='D', zorder=8,
               label=f"Centroid ({res['err_centroid']:.1f} km)")
    circle = plt.Circle(res['var_est'], res['r95'], color='blue',
                        fill=False, linestyle='--', lw=1.3, alpha=0.8,
                        label=f"95% CI (r={res['r95']:.1f} km)")
    ax.add_patch(circle)
    ax.set_xlabel('Easting from Bradford centre (km)', fontsize=9)
    ax.set_ylabel('Northing from Bradford centre (km)', fontsize=9)
    ax.text(0.02, 0.98, 'A', transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')
    ax.legend(fontsize=6.5, loc='upper left', framealpha=0.85)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig('figure1A.pdf', bbox_inches='tight', dpi=200)
    plt.close()
    print("Saved figure1A.pdf")

    # ---- Figure 2 (confidence + anchors) ----
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    x2 = np.linspace(-20, 25, 200)
    y2 = np.linspace(-25, 12, 200)
    GX2, GY2 = np.meshgrid(x2, y2)
    grid2 = np.stack([GX2.ravel(), GY2.ravel()], axis=1)
    e2 = np.array([variational_energy(pt, CRIMES, ALPHA, BETA, B)
                   for pt in grid2]).reshape(GX2.shape)
    ax2.contourf(GX2, GY2, -e2, levels=40, cmap='Blues', alpha=0.35)
    ax2.scatter(CRIMES[:,0], CRIMES[:,1], c='navy', s=60, zorder=5,
                label='Crime sites')
    for i,(x,y) in enumerate(CRIMES):
        ax2.annotate(VICTIMS[i].split()[1], (x,y),
                     textcoords='offset points', xytext=(3,3),
                     fontsize=7, color='navy')
    circle2 = plt.Circle(res['var_est'], res['r95'],
                         color='steelblue', fill=False,
                         linestyle='--', lw=2, zorder=7,
                         label=f"95% CI (r={res['r95']:.1f} km)")
    ax2.add_patch(circle2)
    ax2.scatter(*res['var_est'],    c='blue',      s=200, marker='+',
                linewidths=3, zorder=9,
                label=f"Variational ({res['err_var']:.1f} km from Heaton)")
    ax2.scatter(*res['centroid'],   c='orange',    s=150, marker='D',
                zorder=9,
                label=f"Centroid ({res['err_centroid']:.1f} km from Heaton)")
    ax2.scatter(*res['rossmo_est'], c='red',       s=180, marker='X',
                zorder=9,
                label=f"Rossmo ({res['err_rossmo']:.1f} km from Heaton)")
    ax2.scatter(*ANCHOR_HEATON,  c='green',     s=280, marker='*', zorder=10,
                label='Heaton anchor (6 Garden Lane)')
    ax2.scatter(*ANCHOR_CLAYTON, c='limegreen', s=200, marker='P', zorder=10,
                label='Clayton anchor (Tanton Crescent)')
    ax2.plot([ANCHOR_HEATON[0], ANCHOR_CLAYTON[0]],
             [ANCHOR_HEATON[1], ANCHOR_CLAYTON[1]],
             'g:', lw=1.5, alpha=0.7)
    d_anchors = np.linalg.norm(ANCHOR_HEATON - ANCHOR_CLAYTON)
    mid = ((ANCHOR_HEATON+ANCHOR_CLAYTON)/2 + np.array([0.3, 0.2]))
    ax2.annotate(f'{d_anchors:.1f} km', xy=mid,
                 fontsize=8, color='darkgreen')
    for est, col in [(res['var_est'],'blue'),
                     (res['centroid'],'orange'),
                     (res['rossmo_est'],'red')]:
        ax2.plot([est[0], ANCHOR_HEATON[0]], [est[1], ANCHOR_HEATON[1]],
                 color=col, lw=0.8, linestyle=':', alpha=0.5)
    ax2.set_xlabel('Easting from Bradford centre (km)', fontsize=10)
    ax2.set_ylabel('Northing from Bradford centre (km)', fontsize=10)
    ax2.legend(fontsize=8, loc='upper left', framealpha=0.9)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(-20, 25)
    ax2.set_ylim(-25, 12)
    plt.tight_layout()
    plt.savefig('figure2.pdf', bbox_inches='tight', dpi=200)
    plt.close()
    print("Saved figure2.pdf")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == '__main__':
    results = run_case_study()
    make_figures(results)
