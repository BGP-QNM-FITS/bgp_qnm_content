import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.ticker import FixedLocator
from plot_config import PlotConfig
import json
import bgp_qnm_fits as bgp
import seaborn as sns
import os 
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

L_GROUPS = [2, 3, 4, 5, 6]
custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", config.colors)
l_to_color = {
    l: custom_cmap(i / (len(L_GROUPS) - 1))
    for i, l in enumerate(L_GROUPS)
}

SPHERICAL_MODES_PES = [(2, 2), (3, 2), (4, 2)]

TARGET_MODES_PES = [(2, 2), (3, 2)] 

SPHERICAL_MODES_PS = [(2, 2), (3, 2),
                     (3, 3), (4, 3)] 

TARGET_MODES_PS = [(2, 2), (3, 3)]

SPHERICAL_MODES_ALLS = SPHERICAL_MODES_PS + [(l, -m) for l, m in SPHERICAL_MODES_PS]

TARGET_MODES_ALLS = [(2, 2), (3, 3), (2, -2), (3, -3)]

SPHERICAL_MODES_ES = SPHERICAL_MODES_PES + [(l, -m) for l, m in SPHERICAL_MODES_PES]

TARGET_MODES_ES = [(2, 2), (3, 2), (2, -2), (3, -2)]

SPHERICAL_MODES_P = [(2, 2), (3, 2),
                     (3, 3), (4, 3), 
                     (4, 4), (5, 4),
                     (5, 5), (6, 5),
                     (6, 6), (7, 6)] 

TARGET_MODES_P = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]

SPHERICAL_MODES_ALL = SPHERICAL_MODES_P + [(l, -m) for l, m in SPHERICAL_MODES_P]

TARGET_MODES_ALL = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (2, -2), (3, -3), (4, -4), (5, -5), (6, -6)]

SPH_MODE_RULES = {
    "0001": "PES",
    "0002": "PES",
    "0003": "PES",
    "0004": "PES",
    "0005": "PS",
    "0006": "PS",
    "0007": "PS",
    "0008": "ALLS",
    "0009": "ES",
    "0010": "P",
    "0011": "P",
    "0012": "P",
    "0013": "ALL",
}


def get_fits(sim_id, mode_content_data_dict, t0_vals, t0, full_modes_list, spherical_modes): 
    sim = bgp.SXS_CCE(sim_id, type=DATA_TYPE, lev="Lev5", radius="R2")
    tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[sim_id]

    full_modes_list = [list(map(tuple, inner_list)) for inner_list in mode_content_data_dict["modes"]]
    unique_modes = {mode for modes in full_modes_list for mode in modes}

    tdx = np.searchsorted(t0_vals, t0)
    select_modes = full_modes_list[tdx]

    fit = bgp.BGP_fit_lite(sim.times, 
                                sim.h, 
                                select_modes, 
                                sim.Mf, 
                                sim.chif_mag, 
                                tuned_param_dict_GP, 
                                bgp.kernel_GP, 
                                t0=t0, 
                                T=T, 
                                num_samples=NUM_SAMPLES,
                                spherical_modes=spherical_modes,
                                include_chif=INCLUDE_CHIF,
                                include_Mf=INCLUDE_MF,
                                data_type=DATA_TYPE)

    return fit



def get_model_linear(constant_term, mean_vector, ref_params, model_terms):
    return constant_term + np.einsum("p,stp->st", mean_vector - ref_params, model_terms)

def get_models_residuals(spherical_modes, masked_data_array, constant_term, ref_params, model_terms, mean_vector, covariance_matrix, num_draws):
        samples = np.random.multivariate_normal(mean_vector, covariance_matrix, int(num_draws))
        residuals = np.zeros((num_draws, len(spherical_modes), len(masked_data_array[0])), dtype=np.complex128)
        models = np.zeros((num_draws, len(spherical_modes), len(masked_data_array[0])), dtype=np.complex128)
        for j in range(num_draws):
            theta_j = samples[j, :]
            sample_model = get_model_linear(constant_term, theta_j, ref_params, model_terms)
            residual = masked_data_array - sample_model
            residuals[j, :, :] = residual
            models[j, :, :] = sample_model
        return residuals, models


def plot_residuals(sim_id, times, data, models, residuals, spherical_modes, plotting_modes, t0_choice):
    """
    For each spherical mode, plot:
      - Top axis: data (over times), model median, shaded region for model spread
      - Bottom axis: residuals (data - model median), shaded region for residual spread
    """
    n_modes = len(spherical_modes)

    for mode in plotting_modes:
        fig, (ax_data, ax_resid) = plt.subplots(2, 1, figsize=(config.fig_width, config.fig_height * 1.5), 
                                                sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        i = spherical_modes.index(mode)

        # Data shape: (n_modes, n_times)
        # Models/Residuals shape: (n_samples, n_modes, n_times)
        data_mode_real = np.abs(np.real(data[i, :])) 
        data_mode_imag = np.abs(np.imag(data[i, :]))
        models_mode_real = np.abs(np.real(models[:, i, :]))  # shape: (n_samples, n_times)
        residuals_mode_real = np.abs(np.real(residuals[:, i, :]))  # shape: (n_samples, n_times)
        models_mode_imag = np.abs(np.imag(models[:, i, :]))  # shape: (n_samples, n_times)
        residuals_mode_imag = np.abs(np.imag(residuals[:, i, :]))  # shape: (n_samples, n_times)

        # Model statistics
        median_model = np.median(models_mode_real, axis=0)
        p5_model = np.percentile(models_mode_real, 5, axis=0)
        p90_model = np.percentile(models_mode_real, 90, axis=0)

        median_model_imag = np.median(models_mode_imag, axis=0)
        p5_model_imag = np.percentile(models_mode_imag, 5, axis=0)
        p90_model_imag = np.percentile(models_mode_imag, 90, axis=0)

        # Residual statistics
        median_resid = np.median(residuals_mode_real, axis=0)
        p5_resid = np.percentile(residuals_mode_real, 5, axis=0)
        p90_resid = np.percentile(residuals_mode_real, 90, axis=0)

        median_resid_imag = np.median(residuals_mode_imag, axis=0)
        p5_resid_imag = np.percentile(residuals_mode_imag, 5, axis=0)
        p90_resid_imag = np.percentile(residuals_mode_imag, 90, axis=0)

        # Top axis: data and model
        ax_data.plot(times, data_mode_real, color=special_color_1, alpha = 0.5, label='Data')
        ax_data.plot(times, data_mode_imag, color=special_color_2, alpha = 0.5)

        ax_data.plot(times, median_model, color='k', label='Model', linestyle='--')
        ax_data.plot(times, median_model_imag, color='k', linestyle=':')

        ax_data.fill_between(times, p5_model, p90_model, color=special_color_1, alpha=0.3)
        ax_data.fill_between(times, p5_model_imag, p90_model_imag, color=special_color_2, alpha=0.3)

        ax_data.set_title(fr"$\ell = {mode[0]}, m = {mode[1]}$ at $t_0 = {t0_choice} \, [M]$")
        ax_data.set_yscale("log")
        ax_data.set_ylabel("Amplitude")
        ax_data.legend() 

        # Bottom axis: residuals
        ax_resid.plot(times, median_resid, color='k', label='Real', linestyle='--')
        ax_resid.plot(times, median_resid_imag, color='k', label='Imag', linestyle=':')

        ax_resid.fill_between(times, p5_resid, p90_resid, color='k', alpha=0.3)
        ax_resid.fill_between(times, p5_resid_imag, p90_resid_imag, color='k', alpha=0.3)

        ax_resid.set_ylabel("Residual")
        ax_resid.set_xlabel(r"Time $t \, [M]$")
        ax_resid.set_yscale("log")
        ax_resid.legend(loc='upper right', fontsize=7)

        ax_data.set_xlim([times[0], times[-1]])
        ax_resid.set_xlim([times[0], times[-1]])

        # Add legends for color and linestyle

        legend_elements = [
            Line2D([0], [0], color=special_color_1, alpha=0.5, label='Data (Real)'),
            Line2D([0], [0], color=special_color_2, alpha=0.5, label='Data (Imag)'),
            Line2D([0], [0], color='k', linestyle='--', label='Model (Real)'),
            Line2D([0], [0], color='k', linestyle=':', label='Model (Imag)'),
        ]

        ax_data.legend(handles=legend_elements, loc='lower left', fontsize=7, ncol=2)

        plt.tight_layout()
        outdir = f"docs/figures/{sim_id}/fits"
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f"{outdir}/fits_{mode[0]}{mode[1]}.png", bbox_inches="tight")
        plt.close()

def __main__():
    sim_ids = [f"{i:04}" for i in range(1, 13)]
    for sim_id in sim_ids:

        sim = bgp.SXS_CCE(sim_id, type=DATA_TYPE, lev="Lev5", radius="R2")

        with open(f'mode_content_files/mode_content_data_{sim_id}.json', 'r') as f:
            mode_content_data_dict = json.load(f)

        t0_vals = np.array(mode_content_data_dict['times'])
        spherical_modes = [tuple(mode) for mode in mode_content_data_dict['spherical_modes']]

        ppc_vals = mode_content_data_dict['p_values']
        below_threshold_idx = next((i for i, val in enumerate(ppc_vals) if val < PVAL_THRESHOLD), None)
        full_modes_list = [list(map(tuple, inner_list)) for inner_list in mode_content_data_dict["modes"]]

        t0_choice = t0_vals[below_threshold_idx]

        fit = get_fits(sim_id, 
                       mode_content_data_dict, 
                       t0_vals, 
                       t0_choice, 
                       full_modes_list, 
                       spherical_modes)

        times = fit.fit['analysis_times']    
        data = fit.fit['data_array_masked']

        residuals, models = get_models_residuals(
            spherical_modes, 
            data, 
            fit.fit['constant_term'], 
            fit.fit['ref_params'], 
            fit.fit['model_terms'], 
            fit.fit['mean'], 
            fit.fit['covariance'], 
            num_draws=NUM_SAMPLES
        )

        mode_rules_map = {
            "PES": (SPHERICAL_MODES_PES, TARGET_MODES_PES),
            "PS": (SPHERICAL_MODES_PS, TARGET_MODES_PS),
            "ES": (SPHERICAL_MODES_ES, TARGET_MODES_ES),
            "ALLS": (SPHERICAL_MODES_ALLS, TARGET_MODES_ALLS),
            "P": (SPHERICAL_MODES_P, TARGET_MODES_P),
            "ALL": (SPHERICAL_MODES_ALL, TARGET_MODES_ALL),
        }

        spherical_modes, plotting_modes = mode_rules_map[SPH_MODE_RULES[sim_id]]
        plot_residuals(sim_id, times, data, models, residuals, spherical_modes, plotting_modes, t0_choice)

if __name__ == "__main__":
    __main__()