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

special_color_1 = to_hex("#8B5FBF")
special_color_2 = to_hex("#C26C88")
special_color_3 = to_hex("#DE6A5E")

DATA_TYPE = 'news'
T = 100
INCLUDE_CHIF = False
INCLUDE_MF = False

NUM_SAMPLES = 10000 

def get_fits(sim_id, mode_content_data_dict, t0_vals, full_modes_list, spherical_modes): 
    sim = bgp.SXS_CCE(sim_id, type=DATA_TYPE, lev="Lev5", radius="R2")
    tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[sim_id]

    full_modes_list = [list(map(tuple, inner_list)) for inner_list in mode_content_data_dict["modes"]]
    unique_modes = {mode for modes in full_modes_list for mode in modes}
    unique_modes_pos = {mode for modes in full_modes_list for mode in modes if mode[3] > 0}

    breakpoint() 

    fits = [] 

    fits_full = bgp.BGP_fit_lite(sim.times, 
                                sim.h, 
                                unique_modes, 
                                sim.Mf, 
                                sim.chif_mag, 
                                tuned_param_dict_GP, 
                                bgp.kernel_GP, 
                                t0=t0_vals, 
                                T=T, 
                                num_samples=NUM_SAMPLES,
                                spherical_modes = spherical_modes,
                                include_chif=INCLUDE_CHIF,
                                include_Mf=INCLUDE_MF,
                                data_type=DATA_TYPE)

    for i, t0 in enumerate(t0_vals): 

        print(f'Fitting from {t0=}')

        select_modes = full_modes_list[i]
        fits.append(bgp.BGP_fit_lite(sim.times, 
                                    sim.h, 
                                    select_modes, 
                                    sim.Mf, 
                                    sim.chif_mag, 
                                    tuned_param_dict_GP, 
                                    bgp.kernel_GP, 
                                    t0=t0, 
                                    T=T, 
                                    num_samples=NUM_SAMPLES,
                                    spherical_modes = spherical_modes,
                                    include_chif=INCLUDE_CHIF,
                                    include_Mf=INCLUDE_MF,
                                    data_type=DATA_TYPE)
                    )
                        
    
    return fits, fits_full


def get_amplitude_stability_plot(sim_id, mode_content_data_dict, Mf_ref, chif_ref, t0_vals, spherical_modes, indices):
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

    full_modes_list = [list(map(tuple, inner_list)) for inner_list in mode_content_data_dict["modes"]]
    colors = LinearSegmentedColormap.from_list("custom_colormap2", config.colors)(np.linspace(0, 1, 3))

    fits, fits_full = get_fits(sim_id, mode_content_data_dict, t0_vals, full_modes_list, spherical_modes)

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
    plt.savefig(f"figures/{sim_id}/epsilon.png", bbox_inches="tight")
    plt.close(fig)

def __main__():
    #sim_ids = [f"{i:04}" for i in range(1, 14)]
    sim_ids = ["0001"]
    for sim_id in sim_ids:

        with open(f'mode_content_files/mode_content_data_{sim_id}.json', 'r') as f:
            mode_content_data_dict = json.load(f)

        t0_vals = np.array(mode_content_data_dict['times'])
        spherical_modes = [tuple(mode) for mode in mode_content_data_dict['spherical_modes']]

        t0_vals_new = np.arange(t0_vals[0], t0_vals[-1] + 1, 5) # TODO to speed things up 
        indices = np.searchsorted(t0_vals, t0_vals_new)
        mode_content_data_dict["modes"] = [mode_content_data_dict["modes"][i] for i in indices]
        mode_content_data_dict["p_values"] = [mode_content_data_dict["p_values"][i] for i in indices]

        sim = bgp.SXS_CCE(sim_id, type=DATA_TYPE, lev="Lev5", radius="R2")
        Mf_ref, chif_ref = sim.Mf, sim.chif_mag
        get_amplitude_stability_plot(sim_id, mode_content_data_dict, Mf_ref, chif_ref, t0_vals_new, spherical_modes)

if __name__ == "__main__":
    __main__()