import numpy as np
from matplotlib import pyplot as plt
import json
import os
from plot_config import PlotConfig

config = PlotConfig()
config.apply_style()

def plot_total_mode_count(sim_id, mode_content_data_dict, t0_vals):
    """
    Plot the total number of modes (QNMs, QQNMs, CQNMs combined) as a function of start time.

    Parameters:
    -----------
    sim_id : str
        Simulation ID used for saving figures
    mode_content_data_dict : dict
        Dictionary containing mode content data
    t0_vals : array
        Array of time values
    """

    fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height), dpi=300)

    full_modes_list = [list(map(tuple, inner)) for inner in mode_content_data_dict["modes"]]

    total_counts = [len(modes_at_t0) for modes_at_t0 in full_modes_list]

    # Plot the total counts
    ax.plot(t0_vals, total_counts, linestyle="-", color="k")

    ax.set_xlim([t0_vals[0], t0_vals[-1]])
    ax.set_xlabel("$t_0$ [M]")
    ax.set_ylabel("Number of modes")
    fig.tight_layout()

    # Save the plot
    output_dir = f"figures/{sim_id}"
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f"{output_dir}/total_mode_count_{sim_id}.pdf", bbox_inches="tight")
    plt.close(fig)

def __main__():
    sim_ids = ["0001"]
    for sim_id in sim_ids:
        with open(f'mode_content_files/mode_content_data_{sim_id}.json', 'r') as f:
            mode_content_data_dict = json.load(f)

        t0_vals = np.array(mode_content_data_dict['times'])

        output_dir = f"figures/{sim_id}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plot_total_mode_count(sim_id, mode_content_data_dict, t0_vals)

if __name__ == "__main__":
    __main__()