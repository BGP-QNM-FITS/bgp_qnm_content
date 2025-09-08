import numpy as np
from matplotlib import pyplot as plt
from plot_config import PlotConfig
import bgp_qnm_fits as bgp
from matplotlib.colors import LinearSegmentedColormap

# Constants
ID = '0001'
DATA_TYPE = 'news'
SPHERICAL_MODES = [(2, 2)]
THRESHOLD = 0.9
N_MAX = 6
T = 100
N_DRAWS = 1000
INCLUDE_CHIF = True
INCLUDE_MF = True
t0_vals = np.arange(0, 50.1, 4)

sim = bgp.SXS_CCE(ID, type=DATA_TYPE, lev="Lev5", radius="R2")
tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[ID]
Mf, chif = sim.Mf, sim.chif_mag

log_threshold = np.log(THRESHOLD)
initial_modes = [(*s, 0, 1) for s in SPHERICAL_MODES] + [(3, 2, 0, 1)]
candidate_modes = [(2, 2, n, 1) for n in range(7)]

p_values_adaptive = np.zeros((N_DRAWS, len(t0_vals)))
p_value_adaptive_mean = np.zeros(len(t0_vals))

r_squareds_adaptive = np.zeros((N_DRAWS, len(t0_vals)))
r_squared_adaptive_mean = np.zeros(len(t0_vals))

config = PlotConfig()
config.apply_style()

colors = config.colors
custom_colormap = LinearSegmentedColormap.from_list("custom_colormap", colors)
colors = custom_colormap(np.linspace(0, 1, N_MAX+1))

for i, t0 in enumerate(t0_vals):
    print(f'Fitting from {t0=}')
    select_object = bgp.BGP_select(
        sim.times,
        sim.h,
        initial_modes,
        Mf,
        chif,
        tuned_param_dict_GP,
        bgp.kernel_GP,
        t0=t0,
        candidate_modes=candidate_modes,
        log_threshold=log_threshold,
        num_draws=N_DRAWS,
        T=T,
        spherical_modes=SPHERICAL_MODES,
        include_chif=INCLUDE_CHIF,
        include_Mf=INCLUDE_MF,
        data_type=DATA_TYPE
    )

    r_squareds_adaptive[:, i] = select_object.r_squareds
    r_squared_adaptive_mean[i] = select_object.r_squared_mean

    p_values_adaptive[:, i] = select_object.p_values
    p_value_adaptive_mean[i] = select_object.p_value_mean

p_values_fixed = np.zeros((N_DRAWS, len(t0_vals), len(np.arange(0, N_MAX+1))))
p_value_fixed_mean = np.zeros((len(t0_vals), len(np.arange(0, N_MAX+1))))
r_squareds_fixed = np.zeros((N_DRAWS, len(t0_vals), len(np.arange(0, N_MAX+1))))
r_squared_fixed_mean = np.zeros((len(t0_vals), len(np.arange(0, N_MAX+1))))

for i, N in enumerate(np.arange(1, N_MAX+2)):

    candidate_modes2 = [(2, 2, n, 1) for n in np.arange(N)]
    print(candidate_modes2)

    full_fit = bgp.BGP_fit_lite(
        sim.times,
        sim.h,
        candidate_modes2 + [(3, 2, 0, 1)],
        Mf,
        chif,
        tuned_param_dict_GP,
        bgp.kernel_GP,
        t0=t0_vals,
        num_samples=N_DRAWS,
        t0_method="geq",
        T=T,
        spherical_modes=SPHERICAL_MODES,
        include_chif=INCLUDE_CHIF,
        include_Mf=INCLUDE_MF,
        strain_parameters=False,
        data_type=DATA_TYPE
    )
    r_squareds_fixed[:, :, i] = np.array([fit["r_squareds"] for fit in full_fit.fits]).T
    r_squared_fixed_mean[:, i] = [fit["r_squared_mean"] for fit in full_fit.fits]

    p_values_fixed[:, :, i] = np.array([fit["p_values"] for fit in full_fit.fits]).T
    p_value_fixed_mean[:, i] = [fit["p_value_mean"] for fit in full_fit.fits]

#p_values_adaptive_mean = np.mean(p_values_adaptive, axis=0)
#r_squareds_adaptive_mean = np.mean(r_squareds_adaptive, axis=0)

p_values_adaptive_mean, p_values_adaptive_95, p_values_adaptive_5 = np.median(p_values_adaptive, axis=0), np.percentile(p_values_adaptive, axis=0, q=95), np.percentile(p_values_adaptive, axis=0, q=5)
r_squareds_adaptive_mean, r_squareds_adaptive_95, r_squareds_adaptive_5 = np.median(r_squareds_adaptive, axis=0), np.percentile(r_squareds_adaptive, axis=0, q=95), np.percentile(r_squareds_adaptive, axis=0, q=5)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(config.fig_width, config.fig_height * 2), gridspec_kw={'height_ratios': [1, 1]})

for i, N in enumerate(np.arange(1, N_MAX+2)):
    p_values_fixed_mean, p_values_fixed_95, p_values_fixed_5 = np.mean(p_values_fixed[:, :, i], axis=0), np.percentile(p_values_fixed[:, :, i], axis=0, q=95), np.percentile(p_values_fixed[:, :, i], axis=0, q=5)
    r_squareds_fixed_mean, r_squareds_fixed_95, r_squareds_fixed_5 = np.mean(r_squareds_fixed[:, :, i], axis=0), np.percentile(r_squareds_fixed[:, :, i], axis=0, q=95), np.percentile(r_squareds_fixed[:, :, i], axis=0, q=5)
    #p_values_fixed_mean = np.percentile(p_values_fixed[:, :, i], axis=0, q=98)
    #r_squareds_fixed_mean = np.percentile(r_squareds_fixed[:, :, i], axis=0, q=98)
    ax1.plot(t0_vals, r_squareds_fixed_mean, color=colors[i], linestyle="--", label=f"N={i}", lw=1)
    #ax1.plot(t0_vals, r_squared_fixed_mean[:, i], color=colors[i], linestyle=":", lw = 1.5)
    #ax1.fill_between(t0_vals, r_squareds_fixed_5, r_squareds_fixed_95, color=colors[i], alpha=0.1)
    ax2.plot(t0_vals, p_values_fixed_mean, color=colors[i], linestyle="--", lw=1)
    ax2.plot(t0_vals, p_value_fixed_mean[:, i], color=colors[i], linestyle=":", label=f"N={i}", lw=1.5)
    #ax2.fill_between(t0_vals, p_values_fixed_5, p_values_fixed_95, color=colors[i], alpha=0.1)

    #ax1.fill_between(t0_vals, ppcs_basic_lower, ppcs_basic_upper, alpha=0.1, color=colors[N])

#ax1.plot(t0_vals, r_squareds_adaptive_mean, color="k", linestyle="-", lw=1)
#ax1.plot(t0_vals, r_squared_adaptive_mean, color="k", linestyle=":", lw=1.5)
#ax1.fill_between(t0_vals, r_squareds_adaptive_5, r_squareds_adaptive_95, color="k", alpha=0.1)
ax2.plot(t0_vals, p_values_adaptive_mean, color="k", linestyle="-", lw=1)
ax2.plot(t0_vals, p_value_adaptive_mean, color="k", linestyle=":", lw=1.5)
#ax2.fill_between(t0_vals, p_values_adaptive_5, p_values_adaptive_95, color="k", alpha=0.1)
#ax1.fill_between(t0_vals, ppcs_lower, ppcs_upper, color="k", alpha=0.1)
ax1.set_xlim(t0_vals[0], t0_vals[-1])
ax2.set_xlim(t0_vals[0], t0_vals[-1])
#ax1.set_ylim(-0.1, 1.1)
#ax1.set_ylabel(r"$\mathrm{CDF}$")
ax2.set_xlabel(r"$t_0 [M]$")
ax1.set_yscale('log')
ax1.legend(loc='upper right', fontsize=5)

fig.savefig("outputs/ppc_test.pdf", bbox_inches="tight")
plt.show()
