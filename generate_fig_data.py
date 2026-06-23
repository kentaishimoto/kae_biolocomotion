#!/usr/bin/env python
# coding: utf-8

# %%
import matplotlib
matplotlib.use('MacOSX')  # macOS標準バックエンド
import matplotlib.pyplot as plt
#plt.ion()  # インタラクティブモード：plt.show() がブロックしなくなる
%matplotlib inline

# %%
save_dir = './'
print(f"保存先フォルダの準備が完了しました: {save_dir}")


# %%
# %%
# ==============================================================================
# Combined 3-dataset summary grids — per-component centering
#
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MaxNLocator, ScalarFormatter

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.linewidth": 1.0,        # ← 追加 (0.8 → 1.3 程度)
})

# ====================================================================
# 設定
# ====================================================================
dataset_order = ['Zebrafish', 'Bull', 'CelegansCrawling']


# ====================================================================
# 全 dataset のデータを読み込み
# ====================================================================
all_data = {}
for dl in dataset_order:
    in_path = save_dir + f'summary_data_{dl}.npz'
    print(f"loading: {in_path}")
    all_data[dl] = np.load(in_path, allow_pickle=True)


# ====================================================================
# (1) abc グリッド (v3 と同じ)
# ====================================================================
fig_abc = plt.figure(figsize=(18, 14))
gs_abc = GridSpec(3, 3,
                  hspace=0.35, wspace=0.30,
                  left=0.06, right=0.99,
                  top=0.97, bottom=0.05)

for row, dl in enumerate(dataset_order):
    data = all_data[dl]
    L = float(data['L'])
    N_s_angles = int(data['N_s_angles'])
    n_cycles_obs = int(data['n_cycles_obs'])

    waveforms_body = data['waveforms_body']
    z_all_xy = data['z_all_xy']
    per_cycle_arr = data['per_cycle_bodyframe_norm']
    g_kae_xy_norm = data['g_kae_xy_norm']
    g_kae_theta = data['g_kae_theta']
    kae_holonomy_norm = data['kae_holonomy_norm']

    # (a)
    ax_a = fig_abc.add_subplot(gs_abc[row, 0])
    n_frames_total = waveforms_body.shape[0]
    n_frames_show = 20
    show_idx = np.linspace(0, n_frames_total - 1, n_frames_show, dtype=int)
    for i, fi in enumerate(show_idx):
        color = plt.cm.viridis(i / n_frames_show)
        ax_a.plot(waveforms_body[fi, :, 0], waveforms_body[fi, :, 1],
                  color=color, alpha=0.8, linewidth=1.4)
    ax_a.set_xlabel(r"$x'/L$", fontsize=24)
    ax_a.set_ylabel(r"$y'/L$", fontsize=24)
    ax_a.set_aspect('equal', adjustable='datalim')
    ax_a.grid(True, alpha=0.3)
    ax_a.tick_params(labelsize=24)

    # (b)
    ax_b = fig_abc.add_subplot(gs_abc[row, 1])
    t_for_color = np.arange(z_all_xy.shape[0])
    # 背景: 時系列の結線 (もっとはっきり)
    ax_b.plot(z_all_xy[:, 0], z_all_xy[:, 1],
          color='gray', lw=0.8, alpha=0.5)   # 0.4/0.3 → 0.8/0.5
    # メイン: 散布図 (少しはっきり)
    ax_b.scatter(z_all_xy[:, 0], z_all_xy[:, 1],
             c=t_for_color, cmap='plasma', s=14, alpha=0.85)  # 8/0.6 → 14/0.85
    #ax_b.scatter(z_all_xy[:, 0], z_all_xy[:, 1],
    #             c=t_for_color, cmap='plasma', s=8, alpha=0.6)
    #ax_b.plot(z_all_xy[:, 0], z_all_xy[:, 1],
    #          color='gray', lw=0.4, alpha=0.3)
    ax_b.set_xlabel(r"$z_1$", fontsize=24)
    ax_b.set_ylabel(r"$z_2$", fontsize=24)
    ax_b.set_aspect('equal', adjustable='datalim')
    ax_b.grid(True, alpha=0.3)
    ax_b.tick_params(labelsize=24)

    # (c)
    ax_c = fig_abc.add_subplot(gs_abc[row, 2])
    for k in range(n_cycles_obs):
        color = plt.cm.viridis(k / max(n_cycles_obs - 1, 1))
        pc = per_cycle_arr[k]
        ax_c.plot(pc[:, 0], pc[:, 1], color=color, lw=1.0, alpha=0.6)  # 0.6/0.4 → 1.0/0.6

    ax_c.plot(g_kae_xy_norm[:, 0], g_kae_xy_norm[:, 1],
              'r-', lw=2.8, label='Trajectory on limit cycle')

    n_arrows = 8
    n_pts = len(g_kae_xy_norm)
    arrow_indices = np.linspace(0, n_pts - 2, n_arrows + 2, dtype=int)
    traj_range = max(g_kae_xy_norm[:, 0].max() - g_kae_xy_norm[:, 0].min(),
                     g_kae_xy_norm[:, 1].max() - g_kae_xy_norm[:, 1].min())
    arrow_length = 0.10 * traj_range
    arrow_head_size = 0.055 * traj_range

    for idx in arrow_indices:
        x0, y0 = g_kae_xy_norm[idx, 0], g_kae_xy_norm[idx, 1]
        theta_body_at = g_kae_theta[idx] + np.pi
        dx = arrow_length * np.cos(theta_body_at)
        dy = arrow_length * np.sin(theta_body_at)
        ax_c.arrow(x0, y0, dx, dy,
                   head_width=arrow_head_size,
                   head_length=arrow_head_size,
                   fc='red', ec='red', alpha=0.95,
                   length_includes_head=True, lw=1.2, zorder=5)

    ax_c.plot(kae_holonomy_norm[0], kae_holonomy_norm[1],
              'r*', markersize=20, markeredgecolor='k',
              label='Geometric phase')
    ax_c.plot(0, 0, 'yo', markersize=12, markeredgecolor='k',
              label=r'$\Theta=0$')
    ax_c.set_xlabel(r'$x/L$', fontsize=24)
    ax_c.set_ylabel(r'$y/L$', fontsize=24)
    ax_c.yaxis.set_label_coords(-0.16, 0.55)   # 第1引数を増減して左右調整
    ax_c.set_aspect('equal', adjustable='datalim')
    ax_c.grid(True, alpha=0.3)
    #ax_c.legend(fontsize=18, loc='best')
    ax_c.tick_params(labelsize=24)
    ax_c.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_c.yaxis.set_major_locator(MaxNLocator(nbins=3))

out_path_abc = save_dir + 'summary_abc_grid.png'
fig_abc.savefig(out_path_abc, dpi=160, bbox_inches='tight')
print(f"\nsaved: {out_path_abc}")
plt.show()


# ====================================================================
# (2) d グリッド: 各 component を個別中心化 + 行間圧縮
# ====================================================================

# まず各 component を個別中心化, 行内の ylim 範囲を共通化
xmin_global, xmax_global = np.inf, -np.inf
ylim_per_row = []
shapes_centered_per_comp = {}    # dl -> array (3, N_s_fine+1)

for dl in dataset_order:
    data = all_data[dl]
    N_s_fine = int(data['N_s_fine'])
    sens_x_fil_fine_norm = data['sens_x_fil_fine_norm']
    sens_y_fil_fine_norm = data['sens_y_fil_fine_norm']
    sens_X_pt_fine = data['sens_X_pt_fine']
    sens_W_at_phi_fine = data['sens_W_at_phi_fine']
    sens_arrow_scale_norm = data['sens_arrow_scale_norm']

    component_labels_W_title = [r'$Z_W^x$', r'$Z_W^y$', r'$Z_W^\theta / L$']
    centered_y = np.zeros_like(sens_y_fil_fine_norm)
    pts_x_row, pts_y_row = [], []

    for comp in range(3):
        x_fil = sens_x_fil_fine_norm[comp]
        # 各 component を個別中心化
        y_mean_comp = sens_y_fil_fine_norm[comp].mean()
        y_fil_c = sens_y_fil_fine_norm[comp] - y_mean_comp
        centered_y[comp] = y_fil_c

        X_pt = sens_X_pt_fine[comp]
        W_at_phi = sens_W_at_phi_fine[comp]
        arrow_scale_norm = float(sens_arrow_scale_norm[comp])

        pts_x_row.extend(x_fil)
        pts_y_row.extend(y_fil_c)
        for i in range(N_s_fine):
            nx = -np.sin(X_pt[i])
            ny =  np.cos(X_pt[i])
            xp_end = x_fil[i] + nx * W_at_phi[i] * arrow_scale_norm
            yp_end = y_fil_c[i] + ny * W_at_phi[i] * arrow_scale_norm
            pts_x_row.append(xp_end)
            pts_y_row.append(yp_end)
 

    # 行内の ylim 範囲
    margin_x = 0.04 * (max(pts_x_row) - min(pts_x_row))
    margin_y = 0.06 * (max(pts_y_row) - min(pts_y_row))
    xmin_row = min(pts_x_row) - margin_x
    xmax_row = max(pts_x_row) + margin_x
    ymin_row = min(min(pts_y_row) - margin_y, -0.25)
    ymax_row = max(max(pts_y_row) + margin_y, 0.25)
    ylim_per_row.append((ymin_row, ymax_row))
    xmin_global = min(xmin_global, xmin_row)
    xmax_global = max(xmax_global, xmax_row)

    shapes_centered_per_comp[dl] = centered_y

xlim_common = (xmin_global, xmax_global)
height_ratios = [(y[1] - y[0]) for y in ylim_per_row]

# 横サイズに対する height_ratios の比率から縦サイズを決定
# xlim 幅 vs 各 ylim 幅の比でアスペクト比を計算
xlim_width = xlim_common[1] - xlim_common[0]
fig_width_inch = 18
ax_width_inch = fig_width_inch / 3 * 0.95    # 3 列, 多少のマージン込み
ax_height_per_row = [ax_width_inch * hr / xlim_width for hr in height_ratios]
fig_height_inch = sum(ax_height_per_row) * 1.10    # hspace 分を 10% 追加

print(f"  ax_width_inch       = {ax_width_inch:.2f}")
print(f"  ax_height_per_row   = {[f'{h:.2f}' for h in ax_height_per_row]}")
print(f"  fig_height_inch     = {fig_height_inch:.2f}")

fig_d = plt.figure(figsize=(fig_width_inch, fig_height_inch))
gs_d = GridSpec(3, 3,
                height_ratios=height_ratios,
                hspace=0.10,     # 行間圧縮
                wspace=0.08,
                left=0.06, right=0.99,
                top=0.99, bottom=0.06)

comp_names = ['x', 'y', r'\theta']

for row, dl in enumerate(dataset_order):
    data = all_data[dl]
    L = float(data['L'])
    N_s_fine = int(data['N_s_fine'])

    sens_phi0_star = data['sens_phi0_star']
    sens_s_value_fine_star = data['sens_s_value_fine_star']
    sens_s_star_fine_idx = data['sens_s_star_fine_idx']
    sens_x_fil_fine_norm = data['sens_x_fil_fine_norm']
    sens_X_pt_fine = data['sens_X_pt_fine']
    sens_W_at_phi_fine = data['sens_W_at_phi_fine']
    sens_arrow_scale_norm = data['sens_arrow_scale_norm']

    centered_y_row = shapes_centered_per_comp[dl]

    for comp in range(3):
        ax_d = fig_d.add_subplot(gs_d[row, comp])
        x_fil = sens_x_fil_fine_norm[comp]
        y_fil = centered_y_row[comp]
        X_pt = sens_X_pt_fine[comp]
        W_at_phi = sens_W_at_phi_fine[comp]
        arrow_scale_norm = float(sens_arrow_scale_norm[comp])
        phi0 = float(sens_phi0_star[comp])
        s_value_fine = float(sens_s_value_fine_star[comp])
        s_star_idx = int(sens_s_star_fine_idx[comp])

        ax_d.plot(x_fil, y_fil, 'k-', lw=3.0)
        ax_d.plot(x_fil[0], y_fil[0], 'ko', markersize=7)

        n_arrows_d = 22
        step = max(1, N_s_fine // n_arrows_d)
        for i in range(0, N_s_fine, step):
            nx = -np.sin(X_pt[i])
            ny =  np.cos(X_pt[i])
            xp, yp = x_fil[i], y_fil[i]
            dx = nx * W_at_phi[i] * arrow_scale_norm
            dy = ny * W_at_phi[i] * arrow_scale_norm
            color = 'r' if W_at_phi[i] > 0 else 'b'
            ax_d.arrow(xp, yp, dx, dy,
                       head_width=0.022, head_length=0.022,
                       fc=color, ec=color, alpha=0.80,
                       length_includes_head=True, lw=0.9)

        s_star_idx_clip = min(s_star_idx, N_s_fine)
        ax_d.plot(x_fil[s_star_idx_clip], y_fil[s_star_idx_clip],
                  'g*', markersize=22, markeredgecolor='k',
                  markeredgewidth=0.6, zorder=10)

        ax_d.set_xlim(xlim_common)
        ax_d.set_ylim(ylim_per_row[row])
        ax_d.set_aspect('equal')

        info_text = (
            #rf'$Z_W^{{{comp_names[comp]}}}$, '
            rf'$\Theta_\ast={phi0/np.pi:.2f}\pi$, '
            rf'$s/L={s_value_fine/L:.2f}$'
        )
        ax_d.text(0.03, 0.95, info_text,
                  transform=ax_d.transAxes,
                  fontsize=24, va='top', ha='left',
                  bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white', alpha=0.85,
                            edgecolor='gray', linewidth=0.5))
        if row == 0:
            ax_d.set_title(component_labels_W_title[comp], fontsize=28)   
        if row == 2:
            ax_d.set_xlabel(r'$x/L$', fontsize=24)
        else:
            ax_d.set_xticklabels([])
        if comp == 0:
            ax_d.set_ylabel(r'$y/L$', fontsize=24)
        else:
            ax_d.set_yticklabels([])
        ax_d.tick_params(labelsize=18)
        ax_d.grid(alpha=0.3)
        ax_d.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax_d.yaxis.set_major_locator(MaxNLocator(nbins=4))

out_path_d = save_dir + 'summary_d_grid.png'
fig_d.savefig(out_path_d, dpi=160, bbox_inches='tight')
print(f"saved: {out_path_d}")
plt.show()

# %%

# %%

# %%
