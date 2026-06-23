#!/usr/bin/env python
"""
Reproduce Figure 1 (summary_KAE_geometry) and Figure 2 (summary_PSF_comparison)
from pre-trained Koopman autoencoder results.

Requirements
------------
numpy, torch, matplotlib, scikit-learn

Input files (expected in the same directory)
--------------------------------------------
koopman_model.pth   -- model state_dict, architecture config, normalization
koopman_data.npz    -- time series t, observation matrix X, generation params

Output files
------------
summary_KAE_geometry_rep.{png,eps}
summary_PSF_comparison_rep.{png,eps}
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


# ============================================================
# 1. Model architecture (inference only; no training code)
# ============================================================
class KoopmanRobustAE(nn.Module):
    """Koopman autoencoder with block-diagonal linear dynamics."""

    def __init__(self, input_dim, dt, latent_dim=2, use_decay_2x2=False):
        super().__init__()
        self.dt = dt
        self.latent_dim = latent_dim
        self.use_decay_2x2 = use_decay_2x2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.Tanh(),
            nn.Linear(32, 16),        nn.Tanh(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.Tanh(),
            nn.Linear(16, 32),         nn.Tanh(),
            nn.Linear(32, input_dim),
        )

        self.omega_param = nn.Parameter(torch.tensor(2.0))
        if self.use_decay_2x2:
            self.decay_param = nn.Parameter(torch.tensor(0.01))
        else:
            self.register_buffer('decay_param', torch.tensor(0.0))
        if self.latent_dim == 3:
            self.lambda_param = nn.Parameter(torch.tensor(1.0))


# ============================================================
# 2. Geometric utilities
# ============================================================
def angles_to_coords(angles, L_total=1.0):
    """Tangent-angle sequence -> planar polyline coordinates."""
    N = len(angles)
    ds = L_total / (N - 1)
    x = np.zeros(N)
    y = np.zeros(N)
    x[1:] = np.cumsum(ds * np.cos(angles[:-1]))
    y[1:] = np.cumsum(ds * np.sin(angles[:-1]))
    return x, y


def compute_decoder_jacobian_batch(model, Z_grid, mean, std):
    """
    Batch Jacobian  J = d(X)/d(z)  of the decoder.
    Z_grid : (N, latent_dim) tensor
    Returns: (N, input_dim, latent_dim) numpy array
    """
    model.eval()
    N, latent_dim = Z_grid.shape
    Z = Z_grid.clone().detach().requires_grad_(True)
    X_norm = model.decoder(Z)
    input_dim = X_norm.shape[1]

    J_all = torch.zeros(N, input_dim, latent_dim)
    for a in range(input_dim):
        g = torch.autograd.grad(X_norm[:, a].sum(), Z, retain_graph=True)[0]
        J_all[:, a, :] = g

    # Undo normalisation scale: dX/dz = std * dX_norm/dz
    J_all = std.unsqueeze(0).unsqueeze(2) * J_all
    return J_all.detach().numpy()


def compute_metric_on_grid(model, mean, std, grid_res=60, r_max=1.8):
    """
    Pull-back metric  g_{ij} = J^T J  on a square grid in latent space.
    Returns: Z1, Z2 (meshgrid arrays), G (grid_res, grid_res, 2, 2)
    """
    z1 = np.linspace(-r_max, r_max, grid_res)
    z2 = np.linspace(-r_max, r_max, grid_res)
    Z1, Z2 = np.meshgrid(z1, z2)

    Z_flat = torch.FloatTensor(np.column_stack([Z1.ravel(), Z2.ravel()]))
    J = compute_decoder_jacobian_batch(model, Z_flat, mean, std)
    G = np.einsum('nai,naj->nij', J, J).reshape(grid_res, grid_res, 2, 2)
    return Z1, Z2, G


# ============================================================
# 3. Phase sensitivity functions (PSF)
# ============================================================
def compute_psf_analytical(theta_vals, wave_params):
    """Exact PSF from the known waveform parameterisation."""
    k   = wave_params['k']
    A   = wave_params['A']
    B   = wave_params['B']
    Nf  = wave_params.get('Nfreq', 2.0)
    phi = wave_params.get('phi', np.pi / 4)
    num_angles = wave_params['num_angles']
    L_total    = wave_params['L_total']

    s  = np.linspace(0, L_total, num_angles)
    ds = L_total / (num_angles - 1)

    Z = np.zeros((len(theta_vals), num_angles))
    for m, th in enumerate(theta_vals):
        psi = k * s - th
        dX_dth = -A * np.cos(psi) - Nf * B * np.cos(Nf * psi + phi)
        Z[m] = dX_dth / (np.sum(dX_dth**2) * ds)
    return Z


def compute_psf_kae(model, mean, std, theta_vals, X_norm, wave_params):
    """PSF via the KAE decoder Jacobian on the learned limit cycle."""
    model.eval()
    z_data = model.encoder(X_norm).detach().numpy()
    r_lc = np.sqrt(z_data[:, 0]**2 + z_data[:, 1]**2).mean()

    std_np    = std.numpy()
    input_dim = mean.shape[0]
    ds = wave_params['L_total'] / (input_dim - 1)

    Z = np.zeros((len(theta_vals), input_dim))
    for m, th in enumerate(theta_vals):
        z = torch.FloatTensor([r_lc * np.cos(th), r_lc * np.sin(th)])
        z.requires_grad_(True)
        x_norm = model.decoder(z)

        J = torch.zeros(input_dim, 2)
        for a in range(input_dim):
            g = torch.autograd.grad(x_norm[a], z, retain_graph=True)[0]
            J[a, :] = g
        J_np = J.detach().numpy()

        dz_dth = r_lc * np.array([-np.sin(th), np.cos(th)])
        dX_dth = std_np * (J_np @ dz_dth)
        Z[m] = dX_dth / (np.sum(dX_dth**2) * ds)
    return Z


def compute_psf_pca(pca_model, mean, std, theta_vals, X_norm, wave_params,
                    flip=False):
    """PSF via the PCA linear decoder."""
    std_np    = std.numpy()
    input_dim = mean.shape[0]
    ds = wave_params['L_total'] / (input_dim - 1)

    z_pca = pca_model.transform(X_norm.numpy())
    r_mean = np.sqrt(z_pca[:, 0]**2 + z_pca[:, 1]**2).mean()
    V = pca_model.components_.T          # (input_dim, 2)
    sign = -1.0 if flip else 1.0

    Z = np.zeros((len(theta_vals), input_dim))
    for m, th in enumerate(theta_vals):
        dz_dth = sign * r_mean * np.array([-np.sin(th), np.cos(th)])
        dX_dth = std_np * (V @ dz_dth)
        Z[m] = dX_dth / (np.sum(dX_dth**2) * ds)
    return Z


# ---- phase-alignment helpers ----
def estimate_phase_offset(theta_est, theta_ref):
    diff = theta_est - theta_ref
    return np.arctan2(np.mean(np.sin(diff)), np.mean(np.cos(diff)))


def estimate_phase_offset_with_flip(theta_est, theta_ref):
    """Try both orientations and pick the better one."""
    diff_no   = theta_est - theta_ref
    off_no    = np.arctan2(np.mean(np.sin(diff_no)),  np.mean(np.cos(diff_no)))
    res_no    = 1.0 - np.mean(np.cos(theta_est - theta_ref - off_no))

    diff_flip = -theta_est - theta_ref
    off_flip  = np.arctan2(np.mean(np.sin(diff_flip)), np.mean(np.cos(diff_flip)))
    res_flip  = 1.0 - np.mean(np.cos(-theta_est - theta_ref - off_flip))

    if res_no <= res_flip:
        return off_no, False
    return off_flip, True


# ============================================================
# 4. Load pre-trained model and data
# ============================================================
def load_model_and_data(model_path='koopman_model.pth',
                        data_path='koopman_data.npz'):
    # ---- data ----
    raw = np.load(data_path, allow_pickle=True)
    t = raw['t']
    X = raw['X']
    wave_params = raw['params'].item()

    # ---- model ----
    ckpt   = torch.load(model_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    model  = KoopmanRobustAE(**config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    mean = torch.FloatTensor(ckpt['mean'])
    std  = torch.FloatTensor(ckpt['std'])

    return model, mean, std, t, X, wave_params


# ============================================================
# 5. Main: compute all derived quantities, then plot
# ============================================================
if __name__ == '__main__':

    save_dir = './'

    # ---- load ----
    model, mean, std, t, X, wave_params = load_model_and_data()
    dt    = wave_params['dt']
    beta  = wave_params['beta']
    delta = wave_params['delta']
    omega = wave_params['omega']
    D_r     = wave_params['D_r']
    D_theta = wave_params['D_theta']

    # ---- normalise ----
    X_tensor = torch.FloatTensor(X)
    X_norm   = (X_tensor - mean) / std

    # ---- KAE encode / decode ----
    with torch.no_grad():
        z_all        = model.encoder(X_norm).numpy()
        X_recon_norm = model.decoder(torch.FloatTensor(z_all)).numpy()
    X_true  = X_tensor.numpy()
    X_recon = X_recon_norm * std.numpy() + mean.numpy()
    z_kae   = z_all

    # ---- PCA (2 components) ----
    pca = PCA(n_components=2)
    X_norm_np    = X_norm.numpy()
    X_pca_latent = pca.fit_transform(X_norm_np)
    X_pca_recon  = pca.inverse_transform(X_pca_latent) * std.numpy() + mean.numpy()

    # ---- phase alignment for PSF ----
    steps       = len(t)
    r_true      = wave_params['r_sde'].copy()
    theta_sde   = wave_params['theta_sde']
    theta_true  = theta_sde + (beta / delta) * np.log(r_true)

    np.random.seed(42)
    for i in range(steps - 1):
        dW_r     = np.random.normal(0, np.sqrt(dt))
        dW_theta = np.random.normal(0, np.sqrt(dt))
        dr     = delta * r_true[i] * (1.0 - r_true[i]**2) * dt + r_true[i] * D_r * dW_r
        dtheta = (omega + beta * r_true[i]**2) * dt + D_theta * dW_theta
        r_true[i+1]     = r_true[i] + dr
        theta_true[i+1] = theta_true[i] + dtheta

    theta_kae_data = np.arctan2(z_kae[:, 1], z_kae[:, 0])
    theta_pca_data = np.arctan2(X_pca_latent[:, 1], X_pca_latent[:, 0])

    offset_kae            = estimate_phase_offset(theta_kae_data, theta_true)
    offset_pca, flip_pca  = estimate_phase_offset_with_flip(theta_pca_data, theta_true)

    # ---- PSF computation ----
    num_theta  = 8
    theta_eval = np.linspace(0, 2 * np.pi, num_theta, endpoint=False)

    Z_analytic = compute_psf_analytical(theta_eval, wave_params)
    Z_kae_psf  = compute_psf_kae(model, mean, std,
                                 theta_eval + offset_kae, X_norm, wave_params)

    if flip_pca:
        theta_pca_aligned = -theta_eval + offset_pca
    else:
        theta_pca_aligned = theta_eval + offset_pca
    Z_pca_psf = compute_psf_pca(pca, mean, std,
                                theta_pca_aligned, X_norm, wave_params,
                                flip=flip_pca)
    if flip_pca:
        Z_pca_psf = -Z_pca_psf

    # ==================================================================
    # Figure 1 : summary_KAE_geometry
    #   upper row (a-c): waveform reconstruction comparison
    #   lower row (d-f): latent spaces and pull-back metric
    #   1-column width  (88 mm ≈ 3.46 in)
    # ==================================================================
    plt.rcParams.update({
        'font.size': 6,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.labelsize': 6,
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
        'legend.fontsize': 5,
        'axes.linewidth': 0.3,
        'xtick.major.width': 0.3,
        'ytick.major.width': 0.3,
        'xtick.major.size': 1.8,
        'ytick.major.size': 1.8,
    })

    fig_w, fig_h = 3.46, 1.80

    left, right = 0.09, 0.97
    top, bottom = 0.96, 0.15

    cbar_w, cbar_gap, cbar_label_pad = 0.016, 0.010, 0.025
    cbar_total = cbar_gap + cbar_w + cbar_label_pad

    wgap_bot     = 0.09
    avail_w_bot  = (right - left) - cbar_total
    panel_w_bot  = (avail_w_bot - 2 * wgap_bot) / 3
    panel_h_bot  = panel_w_bot * fig_w / fig_h

    y_bot  = bottom
    x_bot0 = left
    x_bot1 = left + panel_w_bot + wgap_bot
    x_bot2 = left + 2 * (panel_w_bot + wgap_bot)
    cbar_x = x_bot2 + panel_w_bot + cbar_gap

    panel_h_top = panel_h_bot * 0.50
    wgap_top    = 0.025
    avail_w_top = right - left
    panel_w_top = (avail_w_top - 2 * wgap_top) / 3

    y_top  = top - panel_h_top
    x_top0 = left
    x_top1 = left + panel_w_top + wgap_top
    x_top2 = left + 2 * (panel_w_top + wgap_top)

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_a = fig.add_axes([x_top0, y_top, panel_w_top, panel_h_top])
    ax_b = fig.add_axes([x_top1, y_top, panel_w_top, panel_h_top])
    ax_c = fig.add_axes([x_top2, y_top, panel_w_top, panel_h_top])
    ax_d = fig.add_axes([x_bot0, y_bot, panel_w_bot, panel_h_bot])
    ax_e = fig.add_axes([x_bot1, y_bot, panel_w_bot, panel_h_bot])
    ax_f = fig.add_axes([x_bot2, y_bot, panel_w_bot, panel_h_bot])

    # -- upper row: waveform snapshots --
    num_samples   = 10
    step_interval = len(t) // (num_samples + 1)
    indices       = [i * step_interval for i in range(num_samples)]
    colors_top    = cm.viridis(np.linspace(0, 0.85, num_samples))

    panels_top = [(ax_a, X_true), (ax_b, X_recon), (ax_c, X_pca_recon)]

    all_cx, all_cy = [], []
    for _, arr in panels_top:
        for ti in indices:
            wx, wy = angles_to_coords(arr[ti], L_total=wave_params['L_total'])
            all_cx.append(wx); all_cy.append(wy)
    all_x = np.concatenate(all_cx)
    all_y = np.concatenate(all_cy)
    xpad = 0.05 * (all_x.max() - all_x.min())
    ypad = 0.05 * (all_y.max() - all_y.min())

    for k, (ax, arr) in enumerate(panels_top):
        for i, ti in enumerate(indices):
            wx, wy = angles_to_coords(arr[ti], L_total=wave_params['L_total'])
            ax.plot(wx, wy, color=colors_top[i], lw=0.45, alpha=0.85)
        ax.set_xlabel(r'$x$', labelpad=1)
        ax.set_xlim(all_x.min() - xpad, all_x.max() + xpad)
        ax.set_ylim(all_y.min() - ypad, all_y.max() + ypad)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
        ax.set_yticks([-0.2, 0.0, 0.2])
        ax.tick_params(pad=1.5)
        ax.grid(True, alpha=0.3, lw=0.25)
        if k == 0:
            ax.set_ylabel(r'$y$', labelpad=1)
        else:
            ax.set_yticklabels([])

    # -- lower row (d): KAE latent space --
    ax_d.plot(z_kae[:, 0], z_kae[:, 1], color='gray', lw=0.3, alpha=0.3)
    ax_d.scatter(z_kae[:, 0], z_kae[:, 1], c=t, cmap='plasma',
                 s=0.8, alpha=0.7, rasterized=True, linewidths=0)
    ax_d.set_xlabel(r'$z_1$', labelpad=1)
    ax_d.set_ylabel(r'$z_2$', labelpad=1)
    ax_d.set_aspect('equal', adjustable='box')
    ax_d.tick_params(pad=1.5)
    ax_d.grid(True, alpha=0.3, lw=0.25)

    # -- lower row (e): PCA latent space --
    ax_e.plot(X_pca_latent[:, 0], X_pca_latent[:, 1],
              color='gray', lw=0.3, alpha=0.3)
    ax_e.scatter(X_pca_latent[:, 0], X_pca_latent[:, 1], c=t, cmap='plasma',
                 s=0.8, alpha=0.7, rasterized=True, linewidths=0)
    ax_e.set_xlabel(r'PC$_1$', labelpad=1)
    ax_e.set_ylabel(r'PC$_2$', labelpad=1)
    ax_e.set_aspect('equal', adjustable='box')
    ax_e.tick_params(pad=1.5)
    ax_e.grid(True, alpha=0.3, lw=0.25)

    # -- lower row (f): metric ellipses + volume element --
    grid_res_fig = 60
    Z1_f, Z2_f, G_f = compute_metric_on_grid(model, mean, std,
                                              grid_res=grid_res_fig, r_max=1.8)

    eig_vals_f = np.zeros((grid_res_fig, grid_res_fig, 2))
    eig_vecs_f = np.zeros((grid_res_fig, grid_res_fig, 2, 2))
    for i in range(grid_res_fig):
        for j in range(grid_res_fig):
            vals, vecs = np.linalg.eigh(G_f[i, j])
            eig_vals_f[i, j] = vals
            eig_vecs_f[i, j] = vecs

    det_G_f = G_f[:, :, 0, 0] * G_f[:, :, 1, 1] - G_f[:, :, 0, 1] * G_f[:, :, 1, 0]
    sqrt_det_G_f = np.sqrt(np.maximum(det_G_f, 0))

    pcm = ax_f.pcolormesh(Z1_f, Z2_f, sqrt_det_G_f, cmap='inferno',
                          shading='auto', rasterized=True)

    skip_f = 6
    dz_f = Z1_f[0, 1] - Z1_f[0, 0]
    ell_scale = dz_f * skip_f * 0.14

    ellipses_f = []
    for i in range(0, grid_res_fig, skip_f):
        for j in range(0, grid_res_fig, skip_f):
            vals = eig_vals_f[i, j]
            if vals[0] < 1e-12:
                continue
            vecs = eig_vecs_f[i, j]
            a_len = ell_scale * np.sqrt(vals[1])
            b_len = ell_scale * np.sqrt(vals[0])
            angle_deg = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
            ellipses_f.append(Ellipse(xy=(Z1_f[i, j], Z2_f[i, j]),
                                      width=2*a_len, height=2*b_len,
                                      angle=angle_deg))

    pc_f = PatchCollection(ellipses_f, facecolor='white', alpha=0.55,
                           edgecolors='white', linewidths=0.15)
    ax_f.add_collection(pc_f)
    ax_f.plot(z_all[:, 0], z_all[:, 1], color='cyan', lw=0.3, alpha=0.7)

    ax_f.set_xlabel(r'$z_1$', labelpad=1)
    ax_f.set_ylabel(r'$z_2$', labelpad=1)
    ax_f.set_xlim(Z1_f.min(), Z1_f.max())
    ax_f.set_ylim(Z2_f.min(), Z2_f.max())
    ax_f.set_aspect('equal', adjustable='box')
    ax_f.tick_params(pad=1.5)

    cax = fig.add_axes([cbar_x, y_bot, cbar_w, panel_h_bot])
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.ax.tick_params(labelsize=5, pad=1.5, width=0.3, size=1.8)
    cbar.outline.set_linewidth(0.3)

    fig.savefig(save_dir + 'summary_KAE_geometry_rep.eps')
    fig.savefig(save_dir + 'summary_KAE_geometry_rep.png', dpi=600)
    plt.close(fig)
    print('Saved: summary_KAE_geometry_rep.{eps,png}')

    # ==================================================================
    # Figure 2 : summary_PSF_comparison
    #   2×4 grid of PSF curves (analytical / KAE / PCA)
    #   1-column width  (88 mm ≈ 3.46 in)
    # ==================================================================
    plt.rcParams.update({
        'font.size': 6,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.labelsize': 6,
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
        'legend.fontsize': 5,
        'axes.linewidth': 0.3,
        'xtick.major.width': 0.3,
        'ytick.major.width': 0.3,
        'xtick.major.size': 1.8,
        'ytick.major.size': 1.8,
    })

    s_plot = np.linspace(0, wave_params['L_total'], wave_params['num_angles'])

    y_all = np.concatenate([Z_analytic.ravel(),
                            Z_kae_psf.ravel(),
                            Z_pca_psf.ravel()])
    y_margin = 0.08 * (y_all.max() - y_all.min())
    y_lo = y_all.min() - y_margin
    y_hi = y_all.max() + y_margin

    fig2 = plt.figure(figsize=(3.46, 2.0))
    gs2  = GridSpec(2, 4, figure=fig2,
                    hspace=0.18, wspace=0.18,
                    left=0.10, right=0.98, top=0.95, bottom=0.20)

    for m in range(num_theta):
        row, col = m // 4, m % 4
        ax = fig2.add_subplot(gs2[row, col])
        th = theta_eval[m]

        ax.plot(s_plot, Z_analytic[m], color='black',   ls='-',  lw=0.5)
        ax.plot(s_plot, Z_kae_psf[m],  color='#d62728', ls='--', lw=0.4)
        ax.plot(s_plot, Z_pca_psf[m],  color='#1f77b4', ls=':',  lw=0.5)

        ax.set_ylim(y_lo, y_hi)
        ax.tick_params(pad=1.5)
        ax.grid(True, alpha=0.3, lw=0.25)

        ax.text(0.96, 0.96,
                rf'$\Theta = {np.degrees(th):.0f}^\circ$',
                transform=ax.transAxes, ha='right', va='top', fontsize=5)

        if col == 0:
            ax.set_ylabel(r'$Z_1$', labelpad=1)
        else:
            ax.set_yticklabels([])
        if row == 1:
            ax.set_xlabel(r'$s$', labelpad=1)
        else:
            ax.set_xticklabels([])

    legend_elements = [
        Line2D([0], [0], color='black',   ls='-',  lw=0.5, label='Analytical'),
        Line2D([0], [0], color='#d62728', ls='--', lw=0.4, label='KAE'),
        Line2D([0], [0], color='#1f77b4', ls=':',  lw=0.5, label='PCA'),
    ]
    fig2.legend(handles=legend_elements,
                loc='lower center', bbox_to_anchor=(0.5, 0.0),
                ncol=3, frameon=False,
                handlelength=2.0, columnspacing=1.5, fontsize=6)

    fig2.savefig(save_dir + 'summary_PSF_comparison_rep.eps')
    fig2.savefig(save_dir + 'summary_PSF_comparison_rep.png', dpi=600)
    plt.close(fig2)
    print('Saved: summary_PSF_comparison_rep.{eps,png}')