#!/usr/bin/env python
# coding: utf-8
# ==============================================================================
# make_paper_figures.py
#
# Lightweight (numpy + matplotlib only; no torch) figure builder.
# Reads the variables saved by kae_figure_data.npz and
# produces:
#
#   fig_main.png / .pdf : 2x2 panel
#       (a) dt scan (sigma_g=0):  KAE / PCA-2, error vs A_ideal (log)
#       (b) B/A sweep,  error vs A_beta^true   (phase-label quality)
#       (c) D_r sweep,  error vs A_beta^true
#       (d) beta sweep, error vs A_beta^true   (control)
#
#
# Edit STYLE / labels / layout here freely; no retraining needed.
# ==============================================================================

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

NPZ_PATH = './kae_figure_data.npz'
SHOW_ERRORBANDS = True      # mean +/- std across seeds on the sweep panels

# ------------------------------------------------------------------ style
mpl.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.titlesize': 12,
    'axes.labelsize': 18,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 18,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.2,
    'lines.markersize': 7,
})

STYLE = {
    'true': dict(marker='o', color='black',     label='true asymptotic phase'),
    'KAE':  dict(marker='o', color='#1b7837',   label='KAE'),
    'PCA2': dict(marker='^', color='#762a83',   label='PCA-2'),
}

D = np.load(NPZ_PATH)


def _get(key):
    return D[key] if key in D.files else None

EST_ERR_MAX = 1.0   # per-seed relative-L2 cap; seeds above this are outliers
                    # (phase identification broke down). Applied per method.


def robust_mean_std(raw_row):
    """raw_row: 1D array of per-seed errors for one grid point (may contain NaN
       from the KAE gate). Drop NaN and catastrophic outliers (> EST_ERR_MAX),
       then return (mean, std, n_used)."""
    v = np.asarray(raw_row, dtype=float)
    v = v[np.isfinite(v)]
    v = v[v <= EST_ERR_MAX]
    if v.size == 0:
        return np.nan, np.nan, 0
    return v.mean(), v.std(), v.size

def plot_sweep(ax, tag, ref, xlabel, methods):
    """Plot one sweep panel for a given reference ('ideal' or 'true').
       Uses per-seed raw data if present (re-aggregating with outlier removal
       via robust_mean_std); falls back to the pre-aggregated mean/std."""
    x = D[f'{tag}_x']
    for m in methods:
        raw = _get(f'{tag}_{ref}_{m}_raw')
        if raw is not None:
            stats = [robust_mean_std(raw[i]) for i in range(raw.shape[0])]
            mean = np.array([s[0] for s in stats])
            sd   = np.array([s[1] for s in stats])
        else:
            mean = _get(f'{tag}_{ref}_{m}_mean')
            sd   = _get(f'{tag}_{ref}_{m}_std')
            if mean is None:
                continue
        st = STYLE[m]
        ax.plot(x, mean, marker=st['marker'], color=st['color'], label=st['label'])
        if SHOW_ERRORBANDS and sd is not None and np.any(sd > 0):
            if ax.get_yscale() == 'log':
                lo = np.clip(mean - sd, 1e-15, None)
            else:
                lo = mean - sd
            ax.fill_between(x, lo, mean + sd, color=st['color'], alpha=0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'relative $L^2$ error')



def panel_label(ax, s):
    ax.text(-0.18, 1.04, s, transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left')


# ============================================================ MAIN 2x2 FIGURE
fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))

# (a) dt scan vs A_ideal (log y).
ax = axes[0, 0]
dt_eff = D['dt_eff']
for m in ['KAE', 'PCA2']:
    y = _get(f'dt_{m}')
    if y is not None:
        st = STYLE[m]
        ax.semilogy(dt_eff, y, marker=st['marker'], color=st['color'], label=st['label'])
        if SHOW_ERRORBANDS:
            sd = _get(f'dt_{m}_std')
            if sd is not None and np.any(sd > 0):
                lo = np.clip(y - sd, 1e-15, None)
                ax.fill_between(dt_eff, lo, y + sd, color=st['color'], alpha=0.15)
ax.set_xlabel(r'$\Delta t$')
ax.set_ylabel(r'relative $L^2$ error')
ax.grid(True, which='both', alpha=0.3)
ax.legend(loc='upper left')

# (b) B/A sweep vs true-phase averaging.
ax = axes[0, 1]
plot_sweep(ax, 'BA', 'true',
           r'$A_2/A_1$ ',
           ['KAE', 'PCA2'])
ax.set_yscale('log')
ax.grid(True, which='both', alpha=0.3)
ax.legend(loc='upper left')


# (c) D_r sweep vs true-phase averaging.
ax = axes[1, 0]
plot_sweep(ax, 'Dr', 'true',
           r'$D_r$',
           ['KAE', 'PCA2'])
ax.legend(loc='upper left')


# (d) beta sweep vs true-phase averaging (control).
ax = axes[1, 1]
plot_sweep(ax, 'beta', 'true',
           r'$b$',
           ['KAE', 'PCA2'])
ax.legend(loc='upper left')


fig.tight_layout()
fig.savefig('summary_mech.png', dpi=300, bbox_inches='tight')
fig.savefig('summary_mech.eps', bbox_inches='tight')
print("Saved summary_mech.png / summary_mech.eps")


plt.show()