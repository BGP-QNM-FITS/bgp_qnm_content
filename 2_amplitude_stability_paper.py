import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.ticker import FixedLocator
from plot_config import PlotConfig
import json
import bgp_qnm_fits as bgp

# Configuration
config = PlotConfig()
config.apply_style()

special_color_1 = to_hex("#C26C88")

DATA_TYPE = 'news'
T = 100
INCLUDE_CHIF = False
INCLUDE_MF = False
PVAL_THRESHOLD = 0.9

L_GROUPS = [2, 3, 4, 5, 6]
custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", config.colors)
l_to_color = {
    l: custom_cmap(i / (len(L_GROUPS) - 1))
    for i, l in enumerate(L_GROUPS)
}

nonlinear_modes = {
    (2,2,0,1,2,2,0,1): r'$(2,2,0,+)^2$',
    (2,2,0,1,3,3,0,1): r'$(2,2,0,+) \times (3,3,0,+)$',
    (3,3,0,1,3,3,0,1): r'$(3,3,0,+)^2$',
    (2,2,0,1,4,4,0,1): r'$(2,2,0,+) \times (4,4,0,+)$',
    (2,2,0,1,2,2,0,1,2,2,0,1): r'$(2,2,0,+)^3$',
    (2,-2,0,-1,2,-2,0,-1): r'$(2,-2,0,-)^2$',
    (2,-2,0,-1,3,-3,0,-1): r'$(2,-2,0,-) \times (3,-3,0,-)$',
    (3,-3,0,-1,3,-3,0,-1): r'$(3,-3,0,-)^2$',
    (2,-2,0,-1,4,-4,0,-1): r'$(2,-2,0,-) \times (4,-4,0,-)$',
    (2,-2,0,-1,2,-2,0,-1,2,-2,0,-1): r'$(2,-2,0,-)^3$'
}

nonlinear_linestyle = {
            (2,2,0,1,2,2,0,1): '-',
            (2,2,0,1,3,3,0,1): '-',
            (3,3,0,1,3,3,0,1): '-',
            (2,2,0,1,4,4,0,1): ':',
            (2,2,0,1,2,2,0,1,2,2,0,1): '--'
        }


def get_fits(sim_id, mode_content_data_dict, t0_vals, full_modes_list, spherical_modes): 
    sim = bgp.SXS_CCE(sim_id, type=DATA_TYPE, lev="Lev5", radius="R2")
    tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[sim_id]
    Mf, chif = sim.Mf, sim.chif_mag

    full_modes_list = [list(map(tuple, inner_list)) for inner_list in mode_content_data_dict["modes"]]

    fits = [] 

    for i, t0 in enumerate(t0_vals): 

        print(f'Fitting from {t0=}')

        select_modes = full_modes_list[i]
        fits.append(bgp.BGP_fit(sim.times, 
                                    sim.h, 
                                    select_modes, 
                                    sim.Mf, 
                                    sim.chif_mag, 
                                    tuned_param_dict_GP, 
                                    bgp.kernel_GP, 
                                    t0=t0, 
                                    T=T, 
                                    decay_corrected=True,
                                    strain_parameters=True,
                                    spherical_modes = spherical_modes,
                                    include_chif=INCLUDE_CHIF,
                                    include_Mf=INCLUDE_MF,
                                    data_type=DATA_TYPE)
                    )
    
    return fits 


def masks(mode, t0_vals, full_modes_list):
    mask = np.array([mode in full_modes_list[i] for i in range(len(t0_vals))])
    changes = np.diff(mask.astype(int))
    start_indices = np.where(changes == 1)[0] + 1
    end_indices = np.where(changes == -1)[0] + 1

    if mask[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if mask[-1]:
        end_indices = np.append(end_indices, len(mask))

    return [np.arange(start, end) for start, end in zip(start_indices, end_indices)]

def get_amplitude_stability_plot(sim_id, mode_content_data_dict, spherical_modes, t0_vals, l_max):

    full_modes_list = [list(map(tuple, inner_list)) for inner_list in mode_content_data_dict["modes"]]
    unique_modes = list(set(mode for modes in full_modes_list for mode in modes))
    fits = get_fits(sim_id, mode_content_data_dict, t0_vals, full_modes_list, spherical_modes)
    p_values = mode_content_data_dict["p_values"]

    groups_to_plot = [(2,2), (4,4), (6,6)]
    fig, axes = plt.subplots(1, 3, figsize=(config.fig_width_2, config.fig_height), sharey=True, dpi=300)

    for ax, (l, m) in zip(axes, groups_to_plot):
        possible_modes_for_plot = [mode for mode in unique_modes if (len(mode) == 4 or len(mode) == 2) and mode[0] == l and mode[1] == m] + \
                                [mode for mode in unique_modes if len(mode) == 8 and (mode[0] + mode[4] == l) and (mode[1] + mode[5] == m)] + \
                                [mode for mode in unique_modes if len(mode) == 12 and (mode[0] + mode[4] + mode[8] == l) and (mode[1] + mode[5] + mode[9] == m)]

        if not possible_modes_for_plot:
            continue

        base_color = l_to_color.get(l, "#888888")

        for mode in possible_modes_for_plot:
            label = nonlinear_modes[mode] if mode in nonlinear_modes else ''
            if len(mode) == 4:
                _, em, n, p = mode
                if np.sign(em) != np.sign(p):
                    continue
                alpha = 1-0.1*n
                color = base_color 
                ls = '-'
                lw = 2-0.25*n
            elif len(mode) == 2:
                p = 1
                n = 0
                alpha = 1.0 
                color = special_color_1
                ls = nonlinear_linestyle.get(mode, '-')
                lw = 1.5
            elif len(mode) == 8 or len(mode) == 12:
                p = 1
                n = 0
                alpha = 0.7
                color = special_color_1
                ls = nonlinear_linestyle.get(mode, '-')
                lw = 1.2

            runs = masks(mode, t0_vals, full_modes_list)
            for run in runs:
                temp_t0_vals = t0_vals[run]
                amps = np.zeros_like(temp_t0_vals)
                lowers = np.zeros_like(temp_t0_vals)
                uppers = np.zeros_like(temp_t0_vals)
                for i, t0 in enumerate(temp_t0_vals):
                    tidx = t0_vals.tolist().index(t0)
                    idx = full_modes_list[tidx].index(mode)
                    amps[i] = np.median(fits[tidx].fit["sample_amplitudes"][:, idx])
                    lowers[i] = np.percentile(fits[tidx].fit["sample_amplitudes"][:, idx], 5)
                    uppers[i] = np.percentile(fits[tidx].fit["sample_amplitudes"][:, idx], 95)
                if len(temp_t0_vals) > 1:
                    ax.plot(temp_t0_vals - 0.5, amps,
                            color=color, alpha=alpha, lw=lw,
                            label=label if run[0] == runs[0][0] else "",
                            ls=ls)
                    ax.fill_between(temp_t0_vals - 0.5, lowers, uppers, color=color, alpha=0.4, linewidth=0)
                else:
                    ax.plot([temp_t0_vals[0] - 0.5, temp_t0_vals[0] + 0.5], [amps[0], amps[0]],
                            color=color, alpha=alpha, lw=lw,
                            label=label if run[0] == runs[0][0] else "",
                            ls=ls)
                    ax.fill_between([temp_t0_vals[0] - 0.5, temp_t0_vals[0] + 0.5], [lowers[0], lowers[0]], [uppers[0], uppers[0]], color=color, alpha=0.4, linewidth=0)

        threshold_idx = next((i for i in reversed(range(len(p_values))) if p_values[i] > PVAL_THRESHOLD), None)
        if threshold_idx is not None:
            threshold_idx += 1
            ax.axvspan(0, t0_vals[threshold_idx] - np.median(np.diff(t0_vals))/2, color='grey', alpha=0.2, zorder=0)
        else:
            print(f"No threshold index found for simulation {sim_id}.")

        ax.set_xlim([t0_vals[0], t0_vals[-1]])
        ax.set_xlabel(r"Start time $t_0 \, [M]$")
        ax.set_title(fr"$\ell = m = {m}$")
        ax.set_yscale('log')
        ax.legend(loc='upper right', frameon=False, fontsize=4.5, handlelength=3)

    axes[0].set_ylabel(r"$|\hat{C}_{\alpha}|$")
    axes[0].set_ylim([5e-3, 400])
    plt.subplots_adjust(wspace=0.05) 
    plt.tight_layout()
    plt.savefig(f"paper_figures/amplitude_stability_{sim_id}.pdf", bbox_inches="tight")
    plt.close()

def __main__():
    sim_ids = ["0010"] 
    for sim_id in sim_ids:

        with open(f'mode_content_files/mode_content_data_{sim_id}.json', 'r') as f:
            mode_content_data_dict = json.load(f)

        t0_vals = np.array(mode_content_data_dict['times'])
        spherical_modes = [tuple(mode) for mode in mode_content_data_dict['spherical_modes']]
        candidate_modes = [tuple(mode) for mode in mode_content_data_dict['candidate_modes']]

        l_max = max(mode[0] for mode in candidate_modes if len(mode) == 4)

        get_amplitude_stability_plot(sim_id, mode_content_data_dict, spherical_modes, t0_vals, l_max)

if __name__ == "__main__":
    __main__()  