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

special_color_1 = to_hex("#C26C88")

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

PVAL_THRESHOLD = 0.9

L_GROUPS = [2, 3, 4, 5, 6]
custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", config.colors)
l_to_color = {
    l: custom_cmap(i / (len(L_GROUPS) - 1))
    for i, l in enumerate(L_GROUPS)
}

nonlinear_modes = {
    (2,2,0,1,2,2,0,1): r'$(2,2,0,+)^2$',
    (2,2,0,1,3,3,0,1): r'$(2,2,0,+) \times (3,3,0,+)$',
    (3,3,0,1,3,3,0,1): r'$(3,3,0,+)^2$',
    (2,2,0,1,4,4,0,1): r'$(2,2,0,+) \times (4,4,0,+)$',
    (2,2,0,1,2,2,0,1,2,2,0,1): r'$(2,2,0,+)^3$',
    (2,-2,0,-1,2,-2,0,-1): r'$(2,-2,0,-)^2$',
    (2,-2,0,-1,3,-3,0,-1): r'$(2,-2,0,-) \times (3,-3,0,-)$',
    (3,-3,0,-1,3,-3,0,-1): r'$(3,-3,0,-)^2$',
    (2,-2,0,-1,4,-4,0,-1): r'$(2,-2,0,-) \times (4,-4,0,-)$',
    (2,-2,0,-1,2,-2,0,-1,2,-2,0,-1): r'$(2,-2,0,-)^3$'
}


def classify_mode(mode):
    """Classify a mode as QNM, QQNM, CQNM, or other."""
    if len(mode) == 2:
        l, m = mode
        return "constant", (l, m), 0, True
    elif len(mode) == 4:  # Regular QNM
        l, m, n, p = mode
        return "qnm", (l, m), n, np.sign(p) == np.sign(m)
    elif len(mode) == 8:  # QQNM
        l1, m1, n1, s1, l2, m2, n2, s2 = mode
        l_combined, m_combined = l1 + l2, m1 + m2
        return "qqnm", (l_combined, m_combined), 7, True
    elif len(mode) == 12:  # CQNM
        l1, m1, n1, s1, l2, m2, n2, s2, l3, m3, n3, s3 = mode
        l_combined, m_combined = l1 + l2 + l3, m1 + m2 + m3
        return "cqnm", (l_combined, m_combined), 7, True
    

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

    # Create figure
    fig, ax = plt.subplots(figsize=(config.fig_width_2, config.fig_height_2 * (len(lm_groups) / 7)), dpi=300)
    
    # Sort the groups
    sorted_groups = sorted(lm_groups.items(), key=lambda x: (0,) if x[0] == ('QQNM',) 
                          else (1,) if x[0] == ('CQNM',) 
                          else (2, -x[0][0], 0 if x[0][1] < 0 else 1, abs(x[0][1])) 
                          if isinstance(x[0], tuple) and len(x[0]) == 2 
                          else (3,))
    
    # Calculate positions
    y_pos, y_ticks, y_labels = 0, [], []
    y_positions, key_positions = {}, {}

    for group, modes in sorted_groups:
        group_start = y_pos
        # Separate modes by type while preserving order
        for n, mode in modes:
            type, group_key, sort_key, prograde = classify_mode(mode)

            if type == "constant":
                y_positions[mode] = y_pos
                y_pos += 0.3
            elif type == "qnm":
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

        # Check for prograde and retrograde pairs
        for mode in modes_set:
            if len(mode) == 4: 
                l, m, n, p = mode
                if np.sign(m) != np.sign(p) and (l, m, n, -p) not in modes_set:
                    if (l, m, n, np.sign(m)*1) in key_positions.keys(): 
                        print(sim_id) 
                        print(f"Retrograde mode without prograde counterpart: {mode} at t0={t0}") 
                        y = key_positions[(l, m, n, np.sign(m)*1)]
                        l, m, n, _ = mode
                        alpha = 1.0
                        hatch = '....'
                        mode_color = l_to_color.get(l, "#888888") 
                        pos = key_positions.get((l, m, n, np.sign(m)*1))

        # Process all modes (QNM, QQNM, CQNM) in the same loop
        for group, modes in sorted_groups:
            ell, _ = group
            base_color = l_to_color.get(ell, "#888888") 

            for n, mode in modes:
                if mode not in modes_set:
                    continue
                type, group_key, sort_key, prograde = classify_mode(mode)
                if type == "qnm":
                    y = key_positions[mode]
                    l, m, n, _ = mode
                    alpha = 1.0 - 0.12 * n
                    hatch = '///////////' if (mode in modes_set and (l, m, n, -1 if m>0 else 1) in modes_set) else None
                    mode_color = base_color
                    pos = key_positions.get(mode)
                    label = f"({l},{m},{n})" if n == 0 else ""
                    ordered_positions.append(pos)
                    ordered_labels.append(label)
                elif type == "qqnm" or type == "cqnm":
                    y = y_positions[mode] 
                    alpha = 1.0
                    hatch = None
                    mode_color = special_color_1
                    pos = y_positions.get(mode)
                    ordered_positions.append(pos)
                    ordered_labels.append("")
                    l, m, n, p = mode[:4]
                    label = fr"$({l},{m},{n},{p})^2$" if type == "qqnm" else fr"$({l},{m},{n},{p})^3$"
                elif type == "constant":
                    y = y_positions[mode] 
                    alpha = 1.0
                    hatch = None
                    mode_color = special_color_1
                    pos = y_positions.get(mode)
                    ordered_positions.append(pos)
                    ordered_labels.append("")
                    l, m  = mode
                    label = fr"$({l},{m})$" 
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

    unique_positions = []
    unique_labels = []
    seen_positions = set()
    for label, pos in zip(ordered_labels, ordered_positions):
        if pos not in seen_positions:
            if label == '(2,2,0)':
                seen_positions.add(pos)
                unique_positions.append(pos)
                unique_labels.append("") 
            else:
                unique_positions.append(pos)
                unique_labels.append(label if label else " ") 
                seen_positions.add(pos)

    ax.set_yticks(unique_positions)
    ax.set_yticklabels(unique_labels)


    for label, pos in zip(ordered_labels, ordered_positions):
        if label == '(2,2,0)':
            ax.text(
                ax.get_xlim()[0] + 1.73 * dt,
                pos,                
                label,
                va='bottom',                  
                ha='right'
            )
            break 

    for mode, y in key_positions.items():
        l, m, n, p = mode 
        if (l, m) == (2, 2) and n != 0:
            ax.text(
                t0_vals[0] - 0.5 * dt,  # Place label to the left of the bar
                y,
                fr"$n={n}$",
                fontsize=4.5,
                va='center',
                ha='right'
            )

    for nonlinear_mode in nonlinear_modes.keys():
        # Find the last time index where this mode appears
        last_idx = None
        for t_idx, modes_at_t0 in enumerate(full_modes_list):
            if nonlinear_mode in modes_at_t0:
                last_idx = t_idx
        if last_idx is not None and nonlinear_mode in y_positions:
            pos = y_positions.get(nonlinear_mode)
            x_pos = t0_vals[last_idx]
            if x_pos == 58.0 or x_pos == 60.0: 
                x_pos -= 15
            ax.text(
                x_pos - 2,
                pos + bar_height * 0.9,  # slightly above the bar
                nonlinear_modes[nonlinear_mode],
                fontsize=6,
                va='bottom',
                ha='center'
            )

    #Show threshold region
    threshold_idx = next((i for i, p in enumerate(p_values) if p < PVAL_THRESHOLD), None)
    if threshold_idx is not None:
        ax.axvspan(0, t0_vals[threshold_idx], color='grey', alpha=0.2, zorder=0)

    ax.set_xlabel(r"Start time $t_0 \, [M]$")
    ax.set_xlim(t0_vals[0], t0_vals[-1])
    ax.set_ylim(-0.5, y_pos - 0.3)
    ax.tick_params(axis='y', direction='out', which='both', right=False)

    ax.xaxis.set_minor_locator(FixedLocator(np.arange(t0_vals[0], t0_vals[-1] + dt, 5)))
    
    plt.tight_layout()
    outdir = f"docs/figures/{sim_id}/mode_content"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/mode_content.png", bbox_inches="tight")
    plt.close()


def __main__():
    #sim_ids = [f"{i:04}" for i in range(1, 13)]
    sim_ids = ["0010"]
    for sim_id in sim_ids:

        with open(f'mode_content_files/mode_content_data_{sim_id}.json', 'r') as f:
            mode_content_data_dict = json.load(f)

        t0_vals = np.array(mode_content_data_dict['times'])
        spherical_modes = [tuple(mode) for mode in mode_content_data_dict['spherical_modes']]

        mode_rules_map = {
            "PES": (SPHERICAL_MODES_PES, TARGET_MODES_PES),
            "PS": (SPHERICAL_MODES_PS, TARGET_MODES_PS),
            "ES": (SPHERICAL_MODES_ES, TARGET_MODES_ES),
            "ALLS": (SPHERICAL_MODES_ALLS, TARGET_MODES_ALLS),
            "P": (SPHERICAL_MODES_P, TARGET_MODES_P),
            "ALL": (SPHERICAL_MODES_ALL, TARGET_MODES_ALL),
        }

        spherical_modes, plotting_modes = mode_rules_map[SPH_MODE_RULES[sim_id]]

        plot_mode_content_production(sim_id, mode_content_data_dict, t0_vals, spherical_modes, modes_to_plot=plotting_modes)

if __name__ == "__main__":
    __main__()