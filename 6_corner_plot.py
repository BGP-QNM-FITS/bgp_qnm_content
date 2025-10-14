import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from plot_config import PlotConfig
import bgp_qnm_fits as bgp

# Configuration (reuse same style)
config = PlotConfig()
config.apply_style()

DATA_TYPE = 'news'
T = 100
INCLUDE_CHIF = False
INCLUDE_MF = False
NUM_SAMPLES = 2000

OUTDIR_BASE = "docs/front_page_figures"

def run_single_corner(sim_id, t0_val=10.0):
    path = f"mode_content_files/mode_content_data_{sim_id}.json"
    with open(path, "r") as fh:
        mode_content = json.load(fh)

    times = np.array(mode_content["times"])
    modes_by_time = mode_content["modes"]
    spherical_modes = [tuple(m) for m in mode_content.get("spherical_modes", [])]

    idx = int(np.argmin(np.abs(times - t0_val)))
    t0_actual = float(times[idx])
    select_modes = [tuple(m) for m in modes_by_time[idx]]

    sim = bgp.SXS_CCE(sim_id, type=DATA_TYPE, lev="Lev5", radius="R2")
    tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE).get(sim_id)

    fit_obj = bgp.BGP_fit(
        sim.times,
        sim.h,
        select_modes,
        sim.Mf,
        sim.chif_mag,
        tuned_param_dict_GP,
        bgp.kernel_GP,
        t0=t0_actual,
        T=T,
        num_samples=NUM_SAMPLES,
        spherical_modes=spherical_modes,
        include_chif=INCLUDE_CHIF,
        include_Mf=INCLUDE_MF,
        data_type=DATA_TYPE,
    )

    posterior_samples = fit_obj.fit["samples"]

    amp_indices = []
    amp_labels = []
    for i, m in enumerate(select_modes):
        if len(m) >= 4 and m[0] == 2 and m[1] == 2 and int(m[3]) == 1:
            amp_indices.append(i)
            amp_labels.append(m[2])

    n_params = posterior_samples.shape[1]
    df = pd.DataFrame()

    for idx_mode, n in zip(amp_indices, amp_labels):
        col_re = 2 * idx_mode
        col_im = col_re + 1
        re_label = rf"$\mathrm{{Re}}\,A_{{2,2,{n},+}}$"
        im_label = rf"$\mathrm{{Im}}\,A_{{2,2,{n},+}}$"
        df[re_label] = posterior_samples[:, col_re]
        df[im_label] = posterior_samples[:, col_im]

    os.makedirs(OUTDIR_BASE, exist_ok=True)

    contour_color = "C0"
    scatter_kwargs = dict(s=8, alpha=0.15, color=contour_color, linewidth=0)
    kde_kwargs = dict(levels=4, thresh=0.05, colors=[contour_color], linewidths=1.0)

    grid = sns.PairGrid(df, corner=True)
    grid.map_lower(sns.scatterplot, **scatter_kwargs)
    grid.map_lower(sns.kdeplot, **kde_kwargs)
    grid.map_diag(sns.kdeplot, fill=False, color=contour_color, alpha=0.5)

    # adjust aesthetics
    for ax in grid.axes.flatten():
        if ax is None:
            continue
        ax.tick_params(axis='both', which='major', labelsize=8)

    out_path = os.path.join(OUTDIR_BASE, f"corner_plot.png")
    grid.tight_layout()
    grid.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Wrote corner plot to {out_path}")

if __name__ == "__main__":
    run_single_corner("0010", t0_val=10.0)