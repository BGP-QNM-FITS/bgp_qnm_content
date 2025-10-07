import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.ticker import FixedLocator
from plot_config import PlotConfig
import json
import bgp_qnm_fits as bgp
import seaborn as sns
from matplotlib.lines import Line2D

# Configuration
config = PlotConfig()
config.apply_style()

special_color_1 = to_hex("#8B5FBF")
special_color_2 = to_hex("#C26C88")
special_color_3 = to_hex("#DE6A5E")

DATA_TYPE = 'news'
T = 100
INCLUDE_CHIF = True
INCLUDE_MF = True
PVAL_THRESHOLD = 0.9

NUM_SAMPLES = 1000

def get_fits(sim_id, mode_content_data_dict, t0_vals, full_modes_list, spherical_modes): 
    sim = bgp.SXS_CCE(sim_id, type=DATA_TYPE, lev="Lev5", radius="R2")
    tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[sim_id]

    full_modes_list = [list(map(tuple, inner_list)) for inner_list in mode_content_data_dict["modes"]]
    unique_modes = {mode for modes in full_modes_list for mode in modes}

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

def plot_epsilon_main(sim_id, mode_content_data_dict, Mf_ref, chif_ref, t0_vals, fits, fits_full):
    colors = LinearSegmentedColormap.from_list("custom_colormap2", config.colors2)(np.linspace(0, 1, 3))
    fig, ax_main = plt.subplots(figsize=(config.fig_width, config.fig_height), dpi=300)

    p_values = mode_content_data_dict['p_values']

    threshold_idx = next((i for i in reversed(range(len(p_values))) if p_values[i] > PVAL_THRESHOLD), None)
    if threshold_idx is not None and threshold_idx + 1 < len(t0_vals):
        threshold_idx += 1
        ax_main.axvspan(0, t0_vals[threshold_idx] - np.median(np.diff(t0_vals))/2, color='grey', alpha=0.2, zorder=0)
    else:
        print(f"No threshold index found for simulation {sim_id}.")

    epsilons_adaptive = np.zeros((len(t0_vals), NUM_SAMPLES))
    epsilons_full = np.zeros((len(t0_vals), NUM_SAMPLES))

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

    outdir = f"docs/figures/{sim_id}/epsilon"
    os.makedirs(outdir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/epsilon.png", bbox_inches="tight")
    plt.close(fig)

def plot_epsilon_corners(sim_id, Mf_ref, chif_ref, t0_vals, fits, fits_full):
    colors = LinearSegmentedColormap.from_list("custom_colormap2", config.colors2)(np.linspace(0, 1, 3))
    outdir = f"docs/figures/{sim_id}/epsilon"
    os.makedirs(outdir, exist_ok=True)
    for idx in range(0, len(t0_vals), 5):  # Loop through every fifth t0 value
        fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_width), dpi=300)
        Mf_adaptive, chif_adaptive = fits[idx].fit["samples"][:, -1], fits[idx].fit["samples"][:, -2]
        Mf_full, chif_full = fits_full.fits[idx]["samples"][:, -1], fits_full.fits[idx]["samples"][:, -2]
        sns.kdeplot(
            x=Mf_adaptive, y=chif_adaptive, ax=ax,
            color=colors[0], fill=False, levels=[0.5, 0.9], linewidths=2
        )
        sns.kdeplot(
            x=Mf_full, y=chif_full, ax=ax,
            color=colors[1], fill=False, levels=[0.5, 0.9], linewidths=2
        )
        ax.scatter(Mf_ref, chif_ref, color='red', marker='*', s=120, zorder=10)
        ax.set_xlabel("$M_f$")
        ax.set_ylabel("$\chi_f$")
        ax.set_title(f"$t_0 = {t0_vals[idx]}$")
        handles = [
            Line2D([0], [0], color=colors[0], lw=2, ls='-', label="Adaptive model"),
            Line2D([0], [0], color=colors[1], lw=2, ls='-', label="All modes"),
        ]
        labels = ["Adaptive model", "All modes"]
        ax.legend(handles, labels, frameon=False, loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{outdir}/posterior_{t0_vals[idx]:.1f}.png", bbox_inches="tight")
        plt.close(fig)

def __main__():
    #sim_ids = [f"{i:04}" for i in range(1, 14)]
    sim_ids = ["0012"]
    for sim_id in sim_ids:
        with open(f'mode_content_files/mode_content_data_{sim_id}.json', 'r') as f:
            mode_content_data_dict = json.load(f)
        t0_vals = np.array(mode_content_data_dict['times'])
        spherical_modes = [tuple(mode) for mode in mode_content_data_dict['spherical_modes']]
        sim = bgp.SXS_CCE(sim_id, type=DATA_TYPE, lev="Lev5", radius="R2")
        Mf_ref, chif_ref = sim.Mf, sim.chif_mag
        full_modes_list = [list(map(tuple, inner_list)) for inner_list in mode_content_data_dict["modes"]]
        fits, fits_full = get_fits(sim_id, mode_content_data_dict, t0_vals, full_modes_list, spherical_modes)
        plot_epsilon_main(sim_id, mode_content_data_dict, Mf_ref, chif_ref, t0_vals, fits, fits_full)
        plot_epsilon_corners(sim_id, Mf_ref, chif_ref, t0_vals, fits, fits_full)

if __name__ == "__main__":
    __main__()