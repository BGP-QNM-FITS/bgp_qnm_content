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
        A_PLT_prior = (0, 0.1),
        t_PLT_prior = (t0-100, t0-10),
        lam_PLT_prior = (1, 14),
        nsteps=3000,
        nwalkers=20,
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
        plt.savefig(f"{outdir}/PLT_trace_{i}_{PLT_modes[0]}.png", bbox_inches="tight")
        plt.close()

    autocorrelation_time = fit.sampler.get_autocorr_time(tol=10)

    burn_in = int(0.3 * samples.shape[0])
    flat_samples = fit.sampler.get_chain(discard=burn_in, flat=True)  
    corner.corner(flat_samples, labels=param_names)
    plt.tight_layout()
    plt.savefig(f"{outdir}/PLT_corner_{PLT_modes[0]}.png", bbox_inches="tight")
    plt.close()

    dists = np.sqrt(flat_samples[:,0]**2 + flat_samples[:,1]**2)
    percentile_90 = np.percentile(dists, 90)

    plt.hist(dists, bins=30, density=True, alpha=0.5, color='b')
    plt.tight_layout()
    plt.savefig(f"{outdir}/PLT_histogram_{PLT_modes[0]}.png", bbox_inches="tight")
    plt.close()

    print(f"Results for PLT mode {PLT_modes[0]}:")
    print(f"Autocorrelation times for PLT mode: {autocorrelation_time}")
    print(f"Value at which 90% of the area is contained: {percentile_90}")

    n_iid_samples = int((samples.shape[0] - burn_in) * samples.shape[1] / (2 * np.max(autocorrelation_time)))
    print(f"Estimated number of independent and identically distributed (iid) samples: {n_iid_samples}")

    results[str(PLT_modes[0])] = {
        "autocorrelation_times": autocorrelation_time.tolist(),
        "percentile_90": percentile_90,
        "n_iid_samples": n_iid_samples
    }

with open(f"{outdir}/PLT_results.json", "w") as f:
    json.dump(results, f, indent=4) 

