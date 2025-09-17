import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.ticker import FixedLocator
from plot_config import PlotConfig
import bgp_qnm_fits as bgp
import json 
import os

config = PlotConfig()
config.apply_style()

special_color_1 = to_hex("#8B5FBF")
special_color_2 = to_hex("#C26C88")
special_color_3 = to_hex("#DE6A5E")

SPH_MODE_RULES = {
    "0001": "PE",
    "0002": "PE",
    "0003": "PE",
    "0004": "PE",
    "0005": "P",
    "0006": "P",
    "0007": "P",
    "0008": "ALL",
    "0009": "E",
    "0010": "P",
    "0011": "P",
    "0012": "P",
    "0013": "ALL",
}

SPHERICAL_MODES_PE = [(2, 2),
                      (4, 4), 
                      (6, 6)]


SPHERICAL_MODES_P = [(2, 2),
                    (3, 3),
                    (4, 4),
                    (5, 5),
                    (6, 6),
                    (7, 7),]

def group_sort_key(item):
        group = item[0]
        if group == ('QQNM',):
            return (0,)  
        elif group == ('CQNM',):
            return (1,)  
        else:
            return (2, -group[0], -group[1])

def plot_mode_content_testing(sim_id, mode_content_data_dict, t0_vals, spherical_modes):

    full_modes_list = [list(map(tuple, inner_list)) for inner_list in mode_content_data_dict["modes"]]
    p_values = mode_content_data_dict["p_values"]
    threshold = 0.99  # Define the threshold value

    # Create figure
    fig, ax = plt.subplots(figsize=(config.fig_width_2, config.fig_height_2), dpi=300)

    below_threshold_indices = next((i for i, p in enumerate(p_values) if p < threshold), None)
    t0_below_threshold = t0_vals[below_threshold_indices]

    ax.axvspan(
        0, 
        t0_below_threshold, 
        color='grey', 
        alpha=0.1, 
        zorder=0
    )

    lm_groups = {}
    for modes_at_t0 in full_modes_list:
        for mode in modes_at_t0:
            if len(mode) == 4:
                l, m, n, p = mode
                if (l, m) in spherical_modes:
                    lm_groups.setdefault((l, m), set()).add((n, mode))
            elif len(mode) == 8:
                lm_groups.setdefault(('QQNM',), set()).add((0, mode))
            elif len(mode) == 12:
                lm_groups.setdefault(('CQNM',), set()).add((0, mode))
    
    for group in lm_groups:
        lm_groups[group] = sorted(list(lm_groups[group]), reverse=True)

    sorted_groups = sorted(lm_groups.items(), key=group_sort_key)
    
    colors = LinearSegmentedColormap.from_list("custom_colormap", config.colors)(
        np.linspace(0, 1, len(lm_groups))
    ) 
    
    dt = np.median(np.diff(t0_vals)) 
    
    # Position variables
    y_pos = 0
    y_ticks = []
    y_labels = []
    y_positions = {}
    bar_height = 0.2
    
    for (group, modes) in sorted_groups:
        group_start = y_pos
        
        # Assign positions for all modes in this group
        for n, mode in modes:
            y_positions[mode] = y_pos
            y_pos += 0.3
        
        # Add group label
        y_ticks.append((group_start + y_pos - 0.3) / 2)
        y_labels.append('QQNM' if group == ('QQNM',) else 
                       'CQNM' if group == ('CQNM',) else 
                       f"({group[0]},{group[1]})")
        
        y_pos += 0.6  # Space between groups
    
    # Plot for each t0
    for t_idx, (t0, modes_at_t0) in enumerate(zip(t0_vals, full_modes_list)):
        width = (t0_vals[t_idx + 1] - t0) if t_idx < len(t0_vals) - 1 else dt
        
        for mode in modes_at_t0:
            if mode in y_positions:
                y = y_positions[mode]
                
                if len(mode) == 4:
                    l, m, n, _ = mode
                    color_idx = [g[0] for g in sorted_groups].index((l, m))
                    color = colors[color_idx]
                    alpha = 1.0 - 0.1 * n
                elif len(mode) == 8:
                    color, alpha = special_color_1, 1.0
                elif len(mode) == 12:
                    color, alpha = special_color_2, 1.0
                
                # Draw rectangle
                ax.broken_barh(
                    [(t0 - width/2, width)],
                    (y - bar_height/2, bar_height),
                    facecolors=color,
                    alpha=alpha,
                    edgecolor='none'
                )
    
    x_label_pos = t0_vals[-1] + dt/2
    
    for mode, y in y_positions.items():
        label = mode 
            
        ax.text(
            x_label_pos, 
            y, 
            label, 
            fontsize=3, 
            va='center',
            ha='left'
        )
    
    ax.set_xlabel(r"$t_0 [M]$")
    ax.set_ylabel(r"Mode content")
    ax.set_xlim(t0_vals[0], t0_vals[-1] + dt*1.5)
    ax.set_ylim(-0.5, y_pos - 0.5)
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"figures/{sim_id}/1_mode_content_{sim_id}.png", bbox_inches="tight")
    plt.close()


def classify_mode(mode):
    """Classify a mode as QNM, QQNM, CQNM, or other."""
    if len(mode) == 4:  # Regular QNM
        l, m, n, p = mode
        return "qnm", (l, m), n, p==1
    elif len(mode) == 8:  # QQNM
        l1, m1, n1, s1, l2, m2, n2, s2 = mode
        l_combined, m_combined = l1 + l2, m1 + m2
        return "qqnm", (l_combined, m_combined), max(n1, n2), True
    elif len(mode) == 12:  # CQNM
        l1, m1, n1, s1, l2, m2, n2, s2, l3, m3, n3, s3 = mode
        l_combined, m_combined = l1 + l2 + l3, m1 + m2 + m3
        return "cqnm", (l_combined, m_combined), max(n1, n2, n3), True
    
def plot_mode_content_production(sim_id, mode_content_data_dict, t0_vals, spherical_modes, 
                                modes_to_plot=None):
    """Create a plot showing QNM mode content across different start times.
    
    Parameters:
    -----------
    sim_id : str
        Simulation ID used for saving figures
    mode_content_data_dict : dict
        Dictionary containing mode content data
    t0_vals : array
        Array of time values
    spherical_modes : list
        List of (l,m) tuples representing all possible spherical harmonic modes
    modes_to_plot : list, optional
        List of (l,m) tuples specifying which spherical harmonic mode groups to include.
        If None, all spherical modes found in the data will be plotted.
    """
    # Setup and data preparation
    full_modes_list = [list(map(tuple, inner)) for inner in mode_content_data_dict["modes"]]
    p_values = mode_content_data_dict["p_values"]
    dt = np.median(np.diff(t0_vals))
    bar_height = 0.2
    
    # Create figure
    fig, ax = plt.subplots(figsize=(config.fig_width_2, config.fig_height), dpi=300)
    
    # If modes_to_plot is specified, filter the spherical modes
    spherical_modes_filtered = set(spherical_modes)
    if modes_to_plot is not None:
        spherical_modes_filtered = set(modes_to_plot)

    lm_groups = {}
    all_modes = set(mode for modes in full_modes_list for mode in modes)

    for mode in all_modes:
        _, group_key, sort_key, prograde = classify_mode(mode)
        if group_key in spherical_modes_filtered and prograde: 
            lm_groups.setdefault(group_key, set()).add((sort_key, mode))

    # Sort entries within groups
    for group in lm_groups:
        lm_groups[group] = sorted(list(lm_groups[group]), reverse=True)
    
    # Sort the groups
    sorted_groups = sorted(lm_groups.items(), key=lambda x: (2, -x[0][0], -x[0][1]) 
                          if isinstance(x[0], tuple) and len(x[0]) == 2 
                          else (0,) if x[0] == ('QQNM_other',) 
                          else (1,))
    
    # Setup colors
    colors = LinearSegmentedColormap.from_list("custom_colormap", config.colors)(
        np.linspace(0, 1, len(lm_groups))
    )
    
    # Calculate positions
    y_pos, y_ticks, y_labels = 0, [], []
    y_positions, key_positions = {}, {}
    
    for group, modes in sorted_groups:
        group_start = y_pos
        # Separate modes by type while preserving order
        for n, mode in modes:
            type, group_key, sort_key, prograde = classify_mode(mode)

            if type == "qnm":
                key_positions[mode] = y_pos
                y_pos += 0.3
            elif type == "qqnm":
                y_positions[mode] = y_pos
                y_pos += 0.3
            elif type == "cqnm":
                y_positions[mode] = y_pos
                y_pos += 0.3

        # Add group label
        y_ticks.append((group_start + y_pos - 0.3) / 2)
        y_labels.append('QQNM' if group == ('QQNM',) else 
                        'CQNM' if group == ('CQNM',) else 
                        f"({group[0]},{group[1]})")
        
        y_pos += 0.6  # Space between groups

    ordered_positions = []
    ordered_labels = []
    
    # Plot for each time
    for t_idx, (t0, modes_at_t0) in enumerate(zip(t0_vals, full_modes_list)):
        width = (t0_vals[t_idx + 1] - t0) if t_idx < len(t0_vals) - 1 else dt

        # Create a lookup for quick access to whether a mode appears at this time
        modes_set = set(modes_at_t0)

        # Process all modes (QNM, QQNM, CQNM) in the same loop
        for group, modes in sorted_groups:
            color_idx = [g[0] for g in sorted_groups].index(group)
            base_color = colors[color_idx]

            for n, mode in modes:
                if mode not in modes_set:
                    continue
                type, group_key, sort_key, prograde = classify_mode(mode)
                if type == "qnm":
                    y = key_positions[mode]
                    l, m, n, _ = mode
                    alpha = 1.0 - 0.1 * n
                    hatch = '///////////' if (mode in modes_set and (l, m, n, -1) in modes_set) else None
                    mode_color = base_color
                    pos = key_positions.get(mode)
                    label = f"({l},{m},{n})" if n == 0 else ""
                    ordered_positions.append(pos)
                    ordered_labels.append(label)
                elif type == "qqnm" or type == "cqnm":
                    y = y_positions[mode] 
                    alpha = 1.0
                    hatch = None
                    mode_color = tuple(np.clip(np.array(base_color) * 0.4, 0, 1))
                    pos = y_positions.get(mode)
                    ordered_positions.append(pos)
                    ordered_labels.append("")
                    l, m, n, p = mode[:4]
                    label = fr"$({l},{m},{n},{p})^2$" if type == "qqnm" else fr"$({l},{m},{n},{p})^3$"
                    ax.text(
                        t0_vals[-1] - 7 * dt,
                        pos,
                        label,
                        fontsize=4,
                        va='center',
                        ha='left'
                    )

                # Draw the bar
                ax.broken_barh(
                    [(t0 - width / 2, width)],
                    (y - bar_height / 2, bar_height),
                    facecolors=mode_color,
                    alpha=alpha,
                    edgecolor='none',
                    hatch=hatch
                )

    ax.set_yticks(ordered_positions)
    ax.set_yticklabels(ordered_labels, fontsize=4)

    # Show threshold region
    threshold_idx = next((i for i, p in enumerate(p_values) if p < 0.7), None)
    if threshold_idx is not None:
        ax.axvspan(0, t0_vals[threshold_idx], color='grey', alpha=0.5, zorder=0)

    ax.set_xlabel(r"$t_0 [M]$")
    ax.set_ylabel(r"Mode content")
    ax.set_xlim(t0_vals[0], t0_vals[-1])
    ax.set_ylim(-0.5, y_pos - 0.3)
    
    plt.tight_layout()
    outdir = f"figures/{sim_id}"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/1_mode_content_{sim_id}.pdf", bbox_inches="tight")
    plt.close()


def __main__():
    #sim_ids = [f"{i:04}" for i in range(1, 14)]
    sim_ids = ["0001"]
    for sim_id in sim_ids:

        with open(f'mode_content_files/mode_content_data_{sim_id}_4test_2.json', 'r') as f:
            mode_content_data_dict = json.load(f)

        t0_vals = np.array(mode_content_data_dict['times'])
        spherical_modes = [tuple(mode) for mode in mode_content_data_dict['spherical_modes']]

        #if SPH_MODE_RULES[sim_id] == "PE":
        #    spherical_modes = SPHERICAL_MODES_PE
        #elif SPH_MODE_RULES[sim_id] == "P":
        #    spherical_modes = SPHERICAL_MODES_P
        #elif SPH_MODE_RULES[sim_id] == "E":
        #    spherical_modes = SPHERICAL_MODES_PE
        #elif SPH_MODE_RULES[sim_id] == "ALL":
        #    spherical_modes = SPHERICAL_MODES_P 

        output_dir = f"figures/{sim_id}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        #plot_mode_content_production(sim_id, mode_content_data_dict, t0_vals, spherical_modes, modes_to_plot=spherical_modes)
        plot_mode_content_testing(sim_id, mode_content_data_dict, t0_vals, spherical_modes)

if __name__ == "__main__":
    __main__()