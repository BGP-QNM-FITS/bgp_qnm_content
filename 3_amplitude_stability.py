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

special_color_1 = to_hex("#8B5FBF")
special_color_2 = to_hex("#C26C88")
special_color_3 = to_hex("#DE6A5E")

DATA_TYPE = 'news'
T = 100
INCLUDE_CHIF = False
INCLUDE_MF = False

def get_fits(sim_id, mode_content_data_dict, t0_vals, full_modes_list, spherical_modes): 
    sim = bgp.SXS_CCE(sim_id, type=DATA_TYPE, lev="Lev5", radius="R2")
    tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[sim_id]
    Mf, chif = sim.Mf, sim.chif_mag

    full_modes_list = [list(map(tuple, inner_list)) for inner_list in mode_content_data_dict["modes"]]

    fits = [] 

    #fits_full = bgp.BGP_fit(sim.times, 
    #                        sim.h, 
    #                        list(set(initial_modes + candidate_modes)), 
    #                        sim.Mf, 
    #                        sim.chif_mag, 
    #                        tuned_param_dict_GP, 
    #                        bgp.kernel_GP, 
    #                        t0=t0_vals, 
    #                        T=T, 
    #                        decay_corrected=True,
    #                        spherical_modes = spherical_modes,
    #                        include_chif=INCLUDE_CHIF,
    #                        include_Mf=INCLUDE_MF,
    #                        data_type=DATA_TYPE)

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
    colors = LinearSegmentedColormap.from_list("custom_colormap", config.colors)(np.linspace(0, 1, l_max+1))

    for l in range(2, l_max+1):
        for m in range(-l, l+1):
            possible_modes_for_plot = [mode for mode in unique_modes if (len(mode) == 4 or len(mode) == 2) and mode[0] == l and mode[1] == m] + \
                        [mode for mode in unique_modes if len(mode) == 8 and (mode[0] + mode[4] == l) and (mode[1] + mode[5] == m)] + \
                        [mode for mode in unique_modes if len(mode) == 12 and (mode[0] + mode[4] + mode[8] == l) and (mode[1] + mode[5] + mode[9] == m)]
            
            if possible_modes_for_plot == []:
                continue

            fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height), dpi=300)

            for mode in possible_modes_for_plot:
                label=f'{mode}'
                if len(mode) == 4:
                    _, _, n, p = mode
                    color = colors[l-2]
                elif len(mode) == 2:
                    p = 1
                    n = 0
                    color = special_color_3
                elif len(mode) == 8:
                    p = 1
                    n = 0
                    color = special_color_1
                elif len(mode) == 12:
                    p = 1
                    n = 0 
                    color = special_color_2
                print(f'Plotting {mode=}') 
                runs = masks(mode, t0_vals, full_modes_list)
                for j, run in enumerate(runs):
                    temp_t0_vals = t0_vals[run]
                    amps = np.zeros_like(temp_t0_vals)
                    for i, t0 in enumerate(temp_t0_vals):
                        tidx = t0_vals.tolist().index(t0)
                        idx = full_modes_list[tidx].index(mode)
                        amps[i] = np.median(fits[tidx].fit["sample_amplitudes"][:, idx]) 
                    ax.plot(temp_t0_vals, amps,
                            color=color, alpha=1-0.1*n, lw=2.5-0.25*n,
                            label=label if j==0 and p==1 else None,
                            ls = '-' if p==1 else '--'
                            )

            ax.set_xlim([t0_vals[0], t0_vals[-1]])
            ax.set_xlabel(r"$t_0$ [M]")
            ax.set_ylabel("Amplitude")
            ax.set_yscale('log')

            # Adjust the second legend to appear inside the axis at the top-right
            linestyle_legend = [plt.Line2D([0], [0], color='black', linestyle='-', label='Prograde'),
                                plt.Line2D([0], [0], color='black', linestyle='--', label='Retrograde')]

            legend2 = ax.legend(handles=linestyle_legend, loc='upper right', frameon=False, fontsize=8)
            ax.add_artist(legend2)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False, fontsize=8)  # Re-add the first legend

            plt.savefig(f"figures/{sim_id}/2_amplitude_stability_{sim_id}_{l}{m}.pdf", bbox_inches="tight")
            plt.close()

def __main__():
    #sim_ids = [f"{i:04}" for i in range(1, 14)]
    sim_ids = ["0001"]
    for sim_id in sim_ids:

        with open(f'mode_content_files/mode_content_data_{sim_id}_TEST.json', 'r') as f:
            mode_content_data_dict = json.load(f)

        t0_vals = np.array(mode_content_data_dict['times'])
        spherical_modes = [tuple(mode) for mode in mode_content_data_dict['spherical_modes']]
        initial_modes = [tuple(mode) for mode in mode_content_data_dict['initial_modes']]
        candidate_modes = [tuple(mode) for mode in mode_content_data_dict['candidate_modes']]

        l_max = max(mode[0] for mode in candidate_modes if len(mode) == 4)
        n_max = max(mode[2] for mode in candidate_modes if len(mode) == 4)

        get_amplitude_stability_plot(sim_id, mode_content_data_dict, spherical_modes, t0_vals, l_max)

if __name__ == "__main__":
    __main__()  