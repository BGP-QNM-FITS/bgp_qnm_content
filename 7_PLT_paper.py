import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.ticker import FixedLocator
from plot_config import PlotConfig
import json
import bgp_qnm_fits as bgp
import seaborn as sns
import os 
import corner
from matplotlib.lines import Line2D

config = PlotConfig()
config.apply_style()

sim_id = "0010"
DATA_TYPE = 'news'
t0 = 50 
T = 100
N_MAX = 6
INCLUDE_CHIF = False
INCLUDE_MF = False

NSTEPS = 5000
NWALKERS = 50
A_PLT_PRIOR = (-0.1, 0.1)
T_PLT_PRIOR = (t0-100, t0-10)
LAM_PLT_PRIOR = (1, 15)

sim = bgp.SXS_CCE(sim_id, type=DATA_TYPE, lev="Lev5", radius="R2")
tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[sim_id]

with open(f'mode_content_files/mode_content_data_{sim_id}.json', 'r') as f:
            mode_content_data_dict = json.load(f)

t0_vals = np.array(mode_content_data_dict['times'])
spherical_modes = [tuple(mode) for mode in mode_content_data_dict['spherical_modes']]
full_modes_list = [list(map(tuple, inner)) for inner in mode_content_data_dict["modes"]]
target_sph_modes = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]

t0_idx = np.argmin(np.abs(t0_vals - t0))
modes = full_modes_list[t0_idx]

param_names = [r"$\mathfrak{Re}A^{\beta}$", r"$\mathfrak{Im}A^{\beta}$", r"$t_{\beta}$"]

results = {} 

results = {
      "(2,2)": {
            "autocorrelation_times": [17.24593489, 19.75596196, 62.74109633],
            "percentile_90": 1.639791355225988e-05,
            "n_iid_samples": 1394,
            "n_steps": NSTEPS,
            "n_walkers": NWALKERS,
            "A_PLT_prior": A_PLT_PRIOR,
            "T_PLT_prior": T_PLT_PRIOR,
            "lam_PLT_prior": LAM_PLT_PRIOR
      },
      "(3,3)": {
            "autocorrelation_times": [14.97008348, 12.86091514, 53.58425786],
            "percentile_90": 1.2035153614096133e-05,
            "n_iid_samples": 1632,
            "n_steps": NSTEPS,
            "n_walkers": NWALKERS,
            "A_PLT_prior": A_PLT_PRIOR,
            "T_PLT_prior": T_PLT_PRIOR,
            "lam_PLT_prior": LAM_PLT_PRIOR
      },
      "(4,4)": {
            "autocorrelation_times": [14.18369853, 13.84920344, 45.70641062],
            "percentile_90": 7.735402931452636e-06,
            "n_iid_samples": 1914,
            "n_steps": NSTEPS,
            "n_walkers": NWALKERS,
            "A_PLT_prior": A_PLT_PRIOR,
            "T_PLT_prior": T_PLT_PRIOR,
            "lam_PLT_prior": LAM_PLT_PRIOR
      },
      "(5,5)": {
            "autocorrelation_times": [12.22357828, 14.22925906, 51.8574474],
            "percentile_90": 5.415274857745418e-06,
            "n_iid_samples": 1687,
            "n_steps": NSTEPS,
            "n_walkers": NWALKERS,
            "A_PLT_prior": A_PLT_PRIOR,
            "T_PLT_prior": T_PLT_PRIOR,
            "lam_PLT_prior": LAM_PLT_PRIOR
      },
      "(6,6)": {
            "autocorrelation_times": [12.94084326, 11.65149118, 45.5239344],
            "percentile_90": 3.623374132951909e-06,
            "n_iid_samples": 1922,
            "n_steps": NSTEPS,
            "n_walkers": NWALKERS,
            "A_PLT_prior": A_PLT_PRIOR,
            "T_PLT_prior": T_PLT_PRIOR,
            "lam_PLT_prior": LAM_PLT_PRIOR
      }
}

for PLT_mode in target_sph_modes:

    m = PLT_mode[1]

    PLT_modes = [PLT_mode]
    lams = [2 * l + 2 for (l, _) in PLT_modes]

    fit_sph_modes = [s for s in spherical_modes if s[1] == m]

    fit_modes = [c for c in modes
                 if (len(c) == 4 and c[1] == m) or
                    (len(c) == 8 and c[1] + c[5] == m) or
                    (len(c) == 12 and c[1] + c[5] + c[9] == m)
                ]
    
    fit = bgp.PLT_BGP_fit(
        sim.times,
        sim.h,
        fit_modes,
        sim.Mf,
        sim.chif_mag,
        tuned_param_dict_GP,
        bgp.kernel_GP,
        t0=t0,
        PLT_modes=PLT_modes,
        lam_PLT_val = lams,
        A_PLT_prior = A_PLT_PRIOR,
        t_PLT_prior = T_PLT_PRIOR,
        lam_PLT_prior = LAM_PLT_PRIOR,
        nsteps=NSTEPS,
        nwalkers=NWALKERS,
        T=T,
        spherical_modes=fit_sph_modes,
        include_chif=INCLUDE_CHIF,
        include_Mf=INCLUDE_MF,
        data_type=DATA_TYPE
    )

    samples = fit.sampler.get_chain()
    thin = 10  # or higher
    samples_thin = samples[::thin]

    for i in range(3):
        plt.figure()
        for walker in range(samples_thin.shape[1]):
            plt.plot(samples_thin[:, walker, i], alpha=0.3)
        plt.title(f"Trace plot for parameter {i} ({param_names[i]})")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.tight_layout()
        outdir = f"docs/figures/{sim_id}/PLTs"
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f"{outdir}/PLT_trace_{i}_{PLT_modes[0][0]}{PLT_modes[0][1]}.png", bbox_inches="tight")
        plt.close()

    autocorrelation_time = fit.sampler.get_autocorr_time(tol=10)

    burn_in = int(0.3 * samples.shape[0])
    flat_samples = fit.sampler.get_chain(discard=burn_in, flat=True)  
    corner.corner(flat_samples, labels=param_names)
    plt.tight_layout()
    plt.savefig(f"{outdir}/PLT_corner_{PLT_modes[0][0]}{PLT_modes[0][1]}.png", bbox_inches="tight")
    plt.close()

    dists = np.sqrt(flat_samples[:,0]**2 + flat_samples[:,1]**2)
    percentile_90 = np.percentile(dists, 90)

    plt.hist(dists, bins=30, density=True, alpha=0.5, color='b')
    plt.xlabel(r'$\sqrt{(\mathfrak{Re}A^{\beta})^2 + (\mathfrak{Im}A^{\beta})^2}$')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(f"{outdir}/PLT_histogram_{PLT_modes[0][0]}{PLT_modes[0][1]}.png", bbox_inches="tight")
    plt.close()

    print(f"Results for PLT mode {PLT_modes[0]}:")
    print(f"Autocorrelation times for PLT mode: {autocorrelation_time}")
    print(f"Value at which 90% of the area is contained: {percentile_90}")

    n_iid_samples = int((samples.shape[0] - burn_in) * samples.shape[1] / (2 * np.max(autocorrelation_time)))
    print(f"Estimated number of independent and identically distributed (iid) samples: {n_iid_samples}")

    results[str(PLT_modes[0])] = {
        "autocorrelation_times": autocorrelation_time.tolist(),
        "percentile_90": percentile_90,
        "n_iid_samples": n_iid_samples,
        "n_steps": NSTEPS,
        "n_walkers": NWALKERS,
        "A_PLT_prior": A_PLT_PRIOR,
        "T_PLT_prior": T_PLT_PRIOR,
        "lam_PLT_prior": LAM_PLT_PRIOR
    }

with open(f"PLT_results.json", "w") as f:
    json.dump(results, f, indent=4) 

