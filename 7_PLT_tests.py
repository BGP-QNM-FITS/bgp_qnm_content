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

# Configuration
config = PlotConfig()
config.apply_style()

special_color_1 = to_hex("#8B5FBF")
special_color_2 = to_hex("#C26C88")
special_color_3 = to_hex("#DE6A5E")

sim_id = "0010"
DATA_TYPE = 'news'
t0 = 50 
T = 100
INCLUDE_CHIF = False
INCLUDE_MF = False

modes = [(2,2,0,1)] 
sph_modes = [(2,2)]

PLT_modes = [(2,2)]

sim = bgp.SXS_CCE(sim_id, type=DATA_TYPE, lev="Lev5", radius="R2")
tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[sim_id]

# Artifically inject a tail from 40 M into (2,2)

A_PLT_real = 0.0015
A_PLT_imag = 0.0025

t_PLT = 16
lam_PLT = 1.8

threshold = 30
PLT_term = np.where(sim.times > threshold, (A_PLT_real + 1j * A_PLT_imag) * ((sim.times - t_PLT)/(t0 - t_PLT)) ** (-lam_PLT), 0)
#PLT_term = (A_PLT_real + 1j * A_PLT_imag) * ((sim.times - t_PLT)/(t0 - t_PLT)) ** (-lam_PLT)

sim_data_artificial = {mode: sim.h[mode] for mode in sim.h}
sim_data_artificial[(2,2)] = sim.h[(2,2)] + PLT_term

fit = bgp.PLT_BGP_fit(
    sim.times,
    sim_data_artificial,
    modes,
    sim.Mf,
    sim.chif_mag,
    tuned_param_dict_GP,
    bgp.kernel_GP,
    t0=t0,
    PLT_modes=PLT_modes,
    t_PLT_val = [t_PLT],
    nsteps=1000,
    nwalkers=20,
    T=T,
    spherical_modes=sph_modes,
    include_chif=INCLUDE_CHIF,
    include_Mf=INCLUDE_MF,
    data_type=DATA_TYPE
)

# mcmc_samples: shape (nsteps, nwalkers, ndim)
samples = fit.sampler.get_chain()  # shape: (nsteps, nwalkers, ndim)
param_names = ["A_PLT_real", "A_PLT_imag", "t_PLT", "lam_PLT"]   # Adjust as needed

for i in range(1):
    plt.figure()
    for walker in range(samples.shape[1]):
        plt.plot(samples[:, walker, i], alpha=0.3)
    plt.title(f"Trace plot for parameter {i} ({param_names[i]})")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.show()

burn_in = int(0.3 * samples.shape[0])
flat_samples = fit.sampler.get_chain(discard=burn_in, flat=True)  

tau = fit.sampler.get_autocorr_time()
print("Autocorrelation times:", tau)
# Recommended: nsteps > 50 * max(tau)

corner.corner(flat_samples, labels=param_names)
plt.show()