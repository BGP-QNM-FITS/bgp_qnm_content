import numpy as np
from matplotlib import pyplot as plt
from plot_config import PlotConfig
import bgp_qnm_fits as bgp

# Constants
ID = '0001'
DATA_TYPE = 'news'
SPHERICAL_MODES = [(2, 2)]
THRESHOLD = 0.9
N_MAX = 6
T = 100
N_DRAWS = 10000
INCLUDE_CHIF = True
INCLUDE_MF = True
t0_vals = np.arange(0, 40.1, 2)

sim = bgp.SXS_CCE(ID, type=DATA_TYPE, lev="Lev5", radius="R2")
tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[ID]
Mf, chif = sim.Mf, sim.chif_mag

log_threshold = np.log(THRESHOLD)
initial_modes = [(*s, 0, 1) for s in SPHERICAL_MODES] + [(3, 2, 0, 1)]
candidate_modes = [(2, 2, n, 1) for n in range(7)]

mode_lists = []
ppcs = np.zeros((N_DRAWS, len(t0_vals)))

config = PlotConfig()
config.apply_style()

####################################################################################################

candidate_modes2 = [(2, 2, n, 1) for n in range(7)]

full_fit_6 = bgp.BGP_fit(
    sim.times,
    sim.h,
    candidate_modes2 + [(3, 2, 0, 1)],
    Mf,
    chif,
    tuned_param_dict_GP,
    bgp.kernel_GP,
    t0=30,
    use_nonlinear_params=False,
    num_samples=N_DRAWS,
    t0_method="geq",
    T=T,
    spherical_modes=SPHERICAL_MODES,
    include_chif=INCLUDE_CHIF,
    include_Mf=INCLUDE_MF,
    strain_parameters=False,
    data_type=DATA_TYPE
)
model_6 = full_fit_6.fit["model_array_linear"]
data = full_fit_6.fit["data_array_masked"]


candidate_modes2 = [(2, 2, n, 1) for n in range(1)]

full_fit_1 = bgp.BGP_fit(
    sim.times,
    sim.h,
    candidate_modes2 + [(3, 2, 0, 1)],
    Mf,
    chif,
    tuned_param_dict_GP,
    bgp.kernel_GP,
    t0=30,
    use_nonlinear_params=False,
    num_samples=N_DRAWS,
    t0_method="geq",
    T=T,
    spherical_modes=SPHERICAL_MODES,
    include_chif=INCLUDE_CHIF,
    include_Mf=INCLUDE_MF,
    strain_parameters=False,
    data_type=DATA_TYPE
)
model_1 = full_fit_1.fit["model_array_linear"]



fig, ax1 = plt.subplots(figsize=(config.fig_width, config.fig_height))

ax1.plot(t0_vals, ppcs_median, color="k", linestyle="-")

ax1.set_xlim(t0_vals[0], t0_vals[-1])
ax1.set_xlabel(r"$t_0 [M]$")
ax1.set_yscale('log')
ax1.legend()

fig.savefig("outputs/ppc.pdf", bbox_inches="tight")
plt.show()