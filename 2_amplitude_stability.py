import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.ticker import FixedLocator
from plot_config import PlotConfig
import json
import bgp_qnm_fits as bgp
import seaborn as sns 

# Configuration
config = PlotConfig()
config.apply_style()

special_color_1 = to_hex("#C26C88")

DATA_TYPE = 'news'
T = 100
INCLUDE_CHIF = False
INCLUDE_MF = False

NUM_SAMPLES = 1000

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
        (3,3,0,1,3,3,0,1): ':',
        (2,2,0,1,4,4,0,1): '--',
        (2,2,0,1,2,2,0,1,2,2,0,1): '-',
        (2,-2,0,-1,2,-2,0,-1): '-',
        (2,-2,0,-1,3,-3,0,-1): '-',
        (3,-3,0,-1,3,-3,0,-1): ':',
        (2,-2,0,-1,4,-4,0,-1): '-',
        (2,-2,0,-1,2,-2,0,-1,2,-2,0,-1): '-'
    }

def get_fits(sim_id, mode_content_data_dict, t0_vals, full_modes_list, spherical_modes): 
    sim = bgp.SXS_CCE(sim_id, type=DATA_TYPE, lev="Lev5", radius="R2")
    tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[sim_id]

    full_modes_list = [list(map(tuple, inner_list)) for inner_list in mode_content_data_dict["modes"]]
    unique_modes = list({mode for modes in full_modes_list for mode in modes})

    fits = [] 

    fits_full = bgp.BGP_fit(sim.times, 
                                sim.h, 
                                unique_modes, 
                                sim.Mf, 
                                sim.chif_mag, 
                                tuned_param_dict_GP, 
                                bgp.kernel_GP, 
                                t0=t0_vals, 
                                T=T, 
                                decay_corrected=True,
                                strain_parameters=True,
                                num_samples=NUM_SAMPLES,
                                spherical_modes = spherical_modes,
                                include_chif=INCLUDE_CHIF,
                                include_Mf=INCLUDE_MF,
                                data_type=DATA_TYPE)

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
                                    num_samples=NUM_SAMPLES,
                                    spherical_modes = spherical_modes,
                                    include_chif=INCLUDE_CHIF,
                                    include_Mf=INCLUDE_MF,
                                    data_type=DATA_TYPE)
                    )
                        
    
    return fits, fits_full


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


def get_amplitude_stability_plot(fits, sim_id, mode_content_data_dict, spherical_modes, t0_vals, l_max):

    full_modes_list = [list(map(tuple, inner_list)) for inner_list in mode_content_data_dict["modes"]]
    unique_modes = list(set(mode for modes in full_modes_list for mode in modes))
    p_values = mode_content_data_dict["p_values"]

    for l in range(2, l_max+1):
        for m in range(-l, l+1):
            possible_modes_for_plot = [mode for mode in unique_modes if (len(mode) == 4 or len(mode) == 2) and mode[0] == l and mode[1] == m] + \
                        [mode for mode in unique_modes if len(mode) == 8 and (mode[0] + mode[4] == l) and (mode[1] + mode[5] == m)] + \
                        [mode for mode in unique_modes if len(mode) == 12 and (mode[0] + mode[4] + mode[8] == l) and (mode[1] + mode[5] + mode[9] == m)]
            
            if possible_modes_for_plot == []:
                continue

            base_color = l_to_color.get(l, "#888888")

            fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height), dpi=300)

            for mode in possible_modes_for_plot:
                label=f'{mode}'
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
                    alpha = 1.0 
                    color = special_color_1
                    ls = nonlinear_linestyle.get(mode, '-')
                    lw = 1.5
                print(f'Plotting {mode=}') 
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
                        ax.plot(temp_t0_vals - 1, amps,
                                color=color, alpha=alpha, lw=lw,
                                label=label if run[0] == runs[0][0] else "",
                                ls=ls)
                        ax.fill_between(temp_t0_vals - 1, lowers, uppers, color=color, alpha=0.4, linewidth=0)
                    else:
                        ax.plot([temp_t0_vals[0] - 1, temp_t0_vals[0] + 1], [amps[0], amps[0]],
                                color=color, alpha=alpha, lw=lw,
                                label=label if run[0] == runs[0][0] else "",
                                ls=ls)
                        ax.fill_between([temp_t0_vals[0] - 1, temp_t0_vals[0] + 1], [lowers[0], lowers[0]], [uppers[0], uppers[0]], color=color, alpha=0.4, linewidth=0)

            threshold_idx = next((i for i, p in enumerate(p_values) if p < 0.7), None)
            if threshold_idx is not None:
                ax.axvspan(0, t0_vals[threshold_idx], color='grey', alpha=0.2, zorder=0)

            ax.set_xlim([t0_vals[0], t0_vals[-1]])
            ax.set_xlabel(r"Start time $t_0 \, [M]$")
            ax.set_title(fr"$\ell = m = {m}$")
            ax.set_yscale('log')
            ax.legend(loc='upper right', frameon=False, fontsize=4.5)

            ax.set_ylabel(r"$|\hat{C}_{\alpha}|$")
            plt.subplots_adjust(wspace=0.05) 
            plt.tight_layout()
            outdir = f"docs/figures/{sim_id}/amplitude_stability"
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(f"{outdir}/amplitude_stability_{l}{m}.png", bbox_inches="tight")
            plt.close()


def get_epsilon(fits, fits_full, sim_id, mode_content_data_dict, Mf_ref, chif_ref, t0_vals, spherical_modes, indices):
    """
    Generate the amplitude stability plot and add contour plots for specified indices using seaborn kdeplot.

    Parameters:
    -----------
    sim_id : str
        Simulation ID.
    mode_content_data_dict : dict
        Mode content data.
    Mf_ref, chif_ref : float
        Reference values for Mf and chif.
    t0_vals : array
        Array of t0 values.
    spherical_modes : list
        List of spherical modes.
    indices : list of int
        Indices of t0 values for which to generate contour plots.
    """

    colors = LinearSegmentedColormap.from_list("custom_colormap2", config.colors)(np.linspace(0, 1, 3))

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[1, 1, 1])

    # Main plot
    ax_main = fig.add_subplot(gs[0, :])

    p_values = mode_content_data_dict["p_values"]
    threshold_idx = next((i for i, p in enumerate(p_values) if p < 0.7), None)
    if threshold_idx is not None:
        ax_main.axvspan(0, t0_vals[threshold_idx], color='grey', alpha=0.5, zorder=0)

    epsilons_adaptive = np.zeros((len(t0_vals), NUM_SAMPLES))
    epsilons_full = np.zeros((len(t0_vals), NUM_SAMPLES))
    epsilons_full_pos = np.zeros((len(t0_vals), NUM_SAMPLES))

    for i, t0 in enumerate(t0_vals):
        Mf_adaptive, chif_adaptive = fits[i].fit["samples"][:, -1], fits[i].fit["samples"][:, -2]
        Mf_full, chif_full = fits_full.fits[i]["samples"][:, -1], fits_full.fits[i]["samples"][:, -2]
        epsilon_adaptive = np.sqrt((Mf_ref - Mf_adaptive)**2 / Mf_ref**2 + (chif_ref - chif_adaptive)**2 / chif_ref**2)
        epsilon_full = np.sqrt((Mf_ref - Mf_full)**2 / Mf_ref**2 + (chif_ref - chif_full)**2 / chif_ref**2)

        epsilons_adaptive[i, :] = epsilon_adaptive
        epsilons_full[i, :] = epsilon_full

    median_adaptive = np.median(epsilons_adaptive, axis=1)
    p16_adaptive = np.percentile(epsilons_adaptive, 16, axis=1)
    p84_adaptive = np.percentile(epsilons_adaptive, 84, axis=1)
    median_full = np.median(epsilons_full, axis=1)
    p16_full = np.percentile(epsilons_full, 16, axis=1)
    p84_full = np.percentile(epsilons_full, 84, axis=1)

    ax_main.plot(t0_vals, median_adaptive, color=colors[0], label="Adaptive model")
    ax_main.fill_between(t0_vals, p16_adaptive, p84_adaptive, color=colors[0], alpha=0.3)
    ax_main.plot(t0_vals, median_full, color=colors[1], label="All modes")
    ax_main.fill_between(t0_vals, p16_full, p84_full, color=colors[1], alpha=0.3)

    ax_main.legend(frameon=False, loc="upper right", fontsize=8)

    ax_main.set_xlim([t0_vals[0], t0_vals[-1]])
    ax_main.set_xlabel(r"$t_0$ [M]")
    ax_main.set_ylabel(r"$\epsilon$")
    ax_main.set_yscale("log")

    # Contour plots for specified indices
    for idx, i in enumerate(indices):
        ax = fig.add_subplot(gs[1, idx])

        Mf_adaptive, chif_adaptive = fits[i].fit["samples"][:, -1], fits[i].fit["samples"][:, -2]
        Mf_full, chif_full = fits_full.fits[i]["samples"][:, -1], fits_full.fits[i]["samples"][:, -2]

        sns.kdeplot(x=Mf_adaptive, y=chif_adaptive, ax=ax, color=colors[0], fill=False, levels=5, linewidths=2, label="Adaptive model")
        sns.kdeplot(x=Mf_full, y=chif_full, ax=ax, color=colors[1], fill=False, levels=5, linewidths=2, label="All modes")

        ax.scatter(Mf_ref, chif_ref, color='red', marker='*', s=120, label="True $(M_f, \chi_f)$", zorder=10)

        ax.set_xlabel("$M_f$")
        if idx == 0:
            ax.set_ylabel("$\chi_f$")
        ax.set_title(f"$t_0 = {t0_vals[i]}$")

    plt.tight_layout()
    outdir = f"docs/figures/{sim_id}/epsilon"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/epsilon.png", bbox_inches="tight")
    plt.close(fig)


def __main__():
    #sim_ids = [f"{i:04}" for i in range(1, 14)]
    sim_ids = ["0010"]
    for sim_id in sim_ids:

        with open(f'mode_content_files/mode_content_data_{sim_id}.json', 'r') as f:
            mode_content_data_dict = json.load(f)

        t0_vals = np.array(mode_content_data_dict['times'])
        spherical_modes = [tuple(mode) for mode in mode_content_data_dict['spherical_modes']]
        candidate_modes = [tuple(mode) for mode in mode_content_data_dict['candidate_modes']]
        full_modes_list = [list(map(tuple, inner_list)) for inner_list in mode_content_data_dict["modes"]]

        corner_indices = np.searchsorted(t0_vals, [10, 20, 30])

        l_max = max(mode[0] for mode in candidate_modes if len(mode) == 4)

        sim = bgp.SXS_CCE(sim_id, type=DATA_TYPE, lev="Lev5", radius="R2")
        Mf_ref, chif_ref = sim.Mf, sim.chif_mag

        fits, fits_full = get_fits(sim_id, mode_content_data_dict, t0_vals, full_modes_list, spherical_modes)

        get_amplitude_stability_plot(fits, sim_id, mode_content_data_dict, spherical_modes, t0_vals, l_max)
        get_epsilon(fits, fits_full, sim_id, mode_content_data_dict, Mf_ref, chif_ref, t0_vals, spherical_modes, indices=corner_indices)

if __name__ == "__main__":
    __main__()  