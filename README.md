# Data-driven geometric phase in biolocomotion

Code and data to reproduce the figures in:

> ** Data-driven geometric phase in biolocomotion **
> Pyae Hein Htet and Kenta Ishimoto
> Preprint: [arXiv:2606.22440](https://arxiv.org/abs/2606.22440)

## Overview

This repository provides pre-computed data files and plotting scripts that
reproduce the main figures of the paper.  All scripts read `.npz` or `.pth`
files and produce publication-ready EPS figures; no model training or heavy
computation is required.

## Repository structure

```
.
├── generate_fig_kae.py                # Figs. 2, 5
│   ├── koopman_model.pth              #   trained KAE weights & config
│   └── koopman_data.npz               #   synthetic waveform data
│
├── generate_fig_mech.py               # Fig. 6
│   └── kae_figure_data.npz            #   parameter-sweep results
│
├── generate_fig_data.py               # Figs. 3, 4
│   ├── summary_data_Bull.npz          #   bull sperm dataset
│   ├── summary_data_Zebrafish.npz     #   zebrafish sperm dataset
│   └── summary_data_CelegansCrawling.npz  # C. elegans crawling dataset
│
├── README.md
└── LICENSE
```

## Figure–script correspondence

| Figure | Script | Output file(s) | Content |
|--------|--------|-----------------|---------|
| Fig. 2 | `generate_fig_kae.py` | `summary_KAE_geometry_rep.eps` | Waveform reconstruction (KAE vs PCA) and pull-back metric |
| Fig. 3 | `generate_fig_data.py` | `summary_abc_grid.eps` | Experimental waveforms, KAE latent trajectories, and geometric phase extraction |
| Fig. 4 | `generate_fig_data.py` | `summary_d_grid.eps` | Geometric phase sensitivity function |
| Fig. 5 | `generate_fig_kae.py` | `summary_PSF_comparison_rep.eps` | Phase sensitivity function comparison (analytical / KAE / PCA) |
| Fig. 6 | `generate_fig_mech.py` | `summary_fig_mech.eps` | Geometric phase reconstruction accuracy across parameter regimes |

## Requirements

Python ≥ 3.9 with the following packages:

```
numpy
torch
matplotlib
scikit-learn
```

Install all dependencies at once:

```bash
pip install numpy torch matplotlib scikit-learn
```

## Usage

Clone the repository and run each script from the project root:

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>

python generate_fig_kae.py     # → summary_KAE_geometry_rep.eps, summary_PSF_comparison_rep.eps
python generate_fig_mech.py    # → summary_fig_mech.eps
python generate_fig_data.py    # → summary_abc_grid.eps, summary_d_grid.eps
```

All output files are written to the working directory.

## License

This project is released under the [MIT License](LICENSE).
