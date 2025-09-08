import numpy as np
import os
import json
from pathlib import Path
import bgp_qnm_fits as bgp
from matplotlib import pyplot as plt
import seaborn as sns
import time

ID = '0001'
DATA_TYPE = 'news'
T = 120

sim = bgp.SXS_CCE(ID, type=DATA_TYPE, lev="Lev5", radius="R2")
Mf = sim.Mf
chif = sim.chif_mag

SPHERICAL_MODES = [(2,2)]
tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[ID]
THRESHOLD = 0.9
N_MAX = 6

t0_vals = np.arange(0, 60.1, 2)

mode_lists = [] 
times_list = [] 

log_threshold = np.log(THRESHOLD)
initial_modes = [(*s, 0, 1) for s in SPHERICAL_MODES] + [(3,2,0,1)]
candidate_modes = ( [(2,2,n,1) for n in range(0, 7)]) 

for t0 in t0_vals:
    print(f'Fitting from {t0=}')

    select_object = bgp.BGP_select(sim.times, 
            sim.h, 
            initial_modes, 
            sim.Mf, 
            sim.chif_mag, 
            tuned_param_dict_GP, 
            bgp.kernel_GP, 
            t0=t0, 
            candidate_modes=candidate_modes,
            log_threshold=log_threshold,
            T= T - t0, 
            spherical_modes=SPHERICAL_MODES,
            include_chif=False,
            include_Mf=False,
            data_type=DATA_TYPE
            )
    
    modes = select_object.full_modes
    mode_lists.append(modes) 
    output_path = Path(f'mode_content_lists_{ID}/t0_{t0}_simple.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(modes, f)

"""
full_mode_list = [(3,2,0,1)] + [(2,2,n,1) for n in range(7)]

for mode in full_mode_list:
    plt.figure(figsize=(8, 6))  # Create a new figure for each mode
    for i, t0 in enumerate(t0_vals):
        if mode not in mode_lists[i]:
            continue
        else:
            idx = mode_lists[i].index(mode)
        re = fits[i].fit["samples"][:, 2*idx]
        im = fits[i].fit["samples"][:, 2*idx+1]

        # Create a contour plot for this t0
        sns.kdeplot(
            x=re,
            y=im,
            levels=[0.68, 0.95],
            fill=True,
            alpha=0.5,
            label=f't0={t0}'
        )

    # Add labels, legend, and title
    plt.xlabel(r'$Real Part$', fontsize=12)
    plt.ylabel(r'$Imag Part$', fontsize=12)
    plt.title(f'Contour Plot for Mode {mode[0]}_{mode[1]}_{mode[2]}', fontsize=14)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)  # Horizontal line at y=0
    plt.axvline(0, color='black', linestyle='--', linewidth=1)  # Vertical line at x=0
    plt.legend(title="t0 values", fontsize=10)
    plt.tight_layout()

    # Save the contour plot
    output_plot_path = Path(f'plots/contour_plot_mode_{mode[0]}_{mode[1]}_{mode[2]}.png')
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path)
    plt.close()
""" 