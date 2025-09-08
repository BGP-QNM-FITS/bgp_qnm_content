import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.ticker import FixedLocator
from plot_config import PlotConfig
import bgp_qnm_fits as bgp

# Configuration
config = PlotConfig()
config.apply_style()

special_color_1 = to_hex("#395470")
special_color_2 = to_hex("#65858c")

ID = '0001'
DATA_TYPE = 'news'
SPHERICAL_MODES = [(2, 2), (3, 2), (4, 2), (5, 2)] #(5, 2)] # (6, 2) #(4, 4), (5, 4), (6, 4)]

THRESHOLD = 0.9
N_MAX = 6
T = 100
N_DRAWS = 1000
INCLUDE_CHIF = False
INCLUDE_MF = False

t0_vals = np.arange(0, 50.1, 2)

# Simulation and parameter setup
sim = bgp.SXS_CCE(ID, type=DATA_TYPE, lev="Lev5", radius="R2")
tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[ID]
Mf, chif = sim.Mf, sim.chif_mag

log_threshold = np.log(THRESHOLD)
initial_modes = [] #[(2,2,0,1), (2,2,1,1), (3,2,0,1)] #[(*s, 0, -1 if s[1] < 0 else 1) for s in SPHERICAL_MODES]
candidate_modes = [
   (*s, n, -1 if s[1] < 0 else 1) for s in SPHERICAL_MODES for n in range(0, N_MAX + 1)
]

#+ [(2, 2, 0, 1, 2, 2, 0, 1)]

# Fit modes
full_modes_list = []
for t0 in t0_vals:
    print(f'Fitting from t0={t0}')
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
    full_modes_list.append(select_object.full_modes)

# Helper function for masks
def masks(mode, full_modes_list):
    mask = np.array([mode in full_modes_list[i] for i in range(len(t0_vals))])
    changes = np.diff(mask.astype(int))
    start_indices = np.where(changes == 1)[0] + 1
    end_indices = np.where(changes == -1)[0] + 1

    if mask[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if mask[-1]:
        end_indices = np.append(end_indices, len(mask))

    return [np.arange(start, end) for start, end in zip(start_indices, end_indices)]

# Plotting
fig, ax = plt.subplots(figsize=(8, 3), dpi=300)

y_tick_len = len(SPHERICAL_MODES)
colors = LinearSegmentedColormap.from_list("custom_colormap", config.colors)(np.linspace(0, 1, y_tick_len))
lm_y_pos = [-i * 10 for i in range(y_tick_len)]
eps = 0.25

for mode in set(candidate_modes + initial_modes):
    if len(mode) == 4:
        l, m, n, _ = mode
        try:
            y = lm_y_pos[SPHERICAL_MODES.index((l, m))] - n
            color = colors[SPHERICAL_MODES.index((l, m))]
        except ValueError:
            continue
    elif len(mode) == 8:
        y, color = 8, special_color_1
    elif len(mode) == 12:
        y, color = 4, special_color_2

    for run in masks(mode, full_modes_list):
        ax.fill_between(
            t0_vals[run],
            y - eps,
            y - 1 + eps,
            color=color,
            alpha=1 - 0.1 * n if len(mode) == 4 else 1,
        )

# Axis labels and ticks
ax.set_xlabel(r"$t_0 [M]$")
ax.set_ylabel(r"$\rm Mode content$")
ax.set_xlim(0, t0_vals[-1])
ax.set_ylim(-(y_tick_len - 1) * 12 - N_MAX - 2, 9)

ticks = [lm_y_pos[i] - 0.5 for i in range(y_tick_len)]
ax.set_yticks(ticks)
ax.set_yticklabels([f"({l},{m})" for l, m in SPHERICAL_MODES])
ax.yaxis.set_minor_locator(FixedLocator([tick - 1 for tick in ticks]))

ax.legend(loc='upper right', frameon=False, fontsize=6, bbox_to_anchor=(1., 1.), ncol=2)

plt.savefig(f"figures/1_mode_content_{ID}.png", bbox_inches="tight")
plt.close() 