import os
import json
import bgp_qnm_fits as bgp

figures_dir = "docs/figures"
rst_dir = "docs"
mode_content_dir = "mode_content_files"

TARGET_MODES_PES = [(2, 2), (3, 2)] 
TARGET_MODES_PS = [(2, 2), (3, 3)]
TARGET_MODES_ALLS = [(2, 2), (3, 3), (2, -2), (3, -3)]
TARGET_MODES_ES = [(2, 2), (3, 2), (2, -2), (3, -2)]
TARGET_MODES_P = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
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

mode_rules_map = {
            "PES": TARGET_MODES_PES,
            "PS": TARGET_MODES_PS,
            "ES": TARGET_MODES_ES,
            "ALLS": TARGET_MODES_ALLS,
            "P": TARGET_MODES_P,
            "ALL": TARGET_MODES_ALL,
        }


def prettify_heading(name):
    return name.replace('_', ' ').title()


sim_metatdata = {
    "0001": (1, "q1_nospin", (0,0,0), (0,0,0)),
    "0002": (1, "q1_aligned_chi0_2", (0,0,0.2), (0,0,0.2)),
    "0003": (1, "q1_aligned_chi0_4", (0,0,0.4), (0,0,0.4)),
    "0004": (1, "q1_aligned_chi0_6", (0,0,0.6), (0,0,0.6)),
    "0005": (1, "q1_antialigned_chi0_2", (0,0,0.2), (0,0,-0.2)),
    "0006": (1, "q1_antialigned_chi0_4", (0,0,0.4), (0,0,-0.4)),
    "0007": (1, "q1_antialigned_chi0_6", (0,0,0.6), (0,0,-0.6)),
    "0008": (1, "q1_precessing", (0.487, 0.125, -0.327), (-0.190, 0.051, -0.227)),
    "0009": (1, "q1_superkick", (0,0,0.6), (-0.6,0,0)),
    "0010": (4, "q4_no_spin", (0,0,0), (0,0,0)),
    "0011": (4, "q4_aligned_chi0_4", (0,0,0.4), (0,0,0.4)),
    "0012": (4, "q4_antialigned_chi0_4", (0,0,0.4), (0,0,-0.4)),
    "0013": (4, "q4_precessing", (0.487, 0.125, -0.327), (-0.190, 0.051, -0.227)),
}


for sim_folder in sorted(os.listdir(figures_dir)):
    sim_path = os.path.join(figures_dir, sim_folder)
    if os.path.isdir(sim_path):
        sim_id = sim_folder
        # Load metadata from JSON
        json_path = os.path.join(mode_content_dir, f"mode_content_data_{sim_id}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f_json:
                mode_content_data_dict = json.load(f_json)
            spherical_modes = [tuple(mode) for mode in mode_content_data_dict['spherical_modes']]
            candidate_modes = [tuple(mode) for mode in mode_content_data_dict['candidate_modes']]
            target_modes = mode_rules_map[SPH_MODE_RULES[sim_id]]
        else:
            spherical_modes = []
            candidate_modes = []
            target_modes = [] 
        # Get Mf and chif from bgp
        try:
            sim = bgp.SXS_CCE(sim_id, type='strain', lev="Lev5", radius="R2")
            Mf = round(sim.Mf, 3)
            chif = round(sim.chif_mag, 3)
        except Exception:
            Mf = "N/A"
            chif = "N/A"
        # Get additional metadata
        mass_ratio, sim_name, spin1, spin2 = sim_metatdata.get(sim_id, ("N/A", "N/A", "N/A"))

        rst_file = os.path.join(rst_dir, f"sim_{sim_folder}.rst")
        with open(rst_file, "w") as f:
            f.write(f"Simulation {sim_folder}\n{'='*27}\n\n")
            # Metadata table (short fields only)
            f.write("+-----------------------+-------------------------+\n")
            f.write("| Metadata Field        | Value                   |\n")
            f.write("+=======================+=========================+\n")
            f.write(f"| Simulation ID         | {sim_id:<23} |\n")
            f.write("+-----------------------+-------------------------+\n")
            f.write(f"| Name                  | {sim_name:<23} |\n")
            f.write("+-----------------------+-------------------------+\n")
            f.write(f"| Mass Ratio            | {mass_ratio:<23} |\n")
            f.write("+-----------------------+-------------------------+\n")
            f.write(f"| Spin 1                | {spin1!s:<23} |\n")
            f.write("+-----------------------+-------------------------+\n")
            f.write(f"| Spin 2                | {spin2!s:<23} |\n")
            f.write("+-----------------------+-------------------------+\n")
            f.write(f"| Final Mass            | {Mf:<23} |\n")
            f.write("+-----------------------+-------------------------+\n")
            f.write(f"| Final Spin            | {chif:<23} |\n")
            f.write("+-----------------------+-------------------------+\n\n")

            # Long fields as code blocks
            f.write("**Spherical harmonics (included in the fits):**\n\n")
            f.write("::\n\n")
            f.write(f"    {str(spherical_modes)}\n\n")
            f.write("**Target harmonics (included in the figures):**\n\n")
            f.write("::\n\n")
            f.write(f"    {str(target_modes)}\n\n")
            f.write("**Candidate modes considered:**\n\n")
            f.write(f"Tuples of length 2 / 4 / 8 / 12 are constant terms / QNMs / quadratic QNMs / cubic QNMs. \n\n")
            f.write("::\n\n")
            f.write(f"    {str(candidate_modes)}\n\n")


            # Find all subfolders
            subfolder_order = ["mode_content", "amplitude_stability", "fits", "epsilon"]
            for subfolder in subfolder_order:
                subfolder_path = os.path.join(sim_path, subfolder)
                if os.path.isdir(subfolder_path):
                    heading = prettify_heading(subfolder)
                    f.write(f"{heading}\n{'-'*len(heading)}\n\n")

                    if subfolder == "mode_content":
                        if sim_id == "0004":
                            f.write("Additional Notes\n----------------\n\n")
                            f.write("This simulation contains two instances of a retrograde mode present without a prograde mode, "
                                    "which are not shown on the mode content plot. These are the (3, 2, 4, -) mode at timesteps"
                                    "t0 = 10.0 [M] and t0 = 12.0 [M]."
                                    "\n\n")
                        elif sim_id == "0009":
                            f.write("Additional Notes\n----------------\n\n")
                            f.write("This simulation contains four instances of a retrograde mode present without a prograde mode, "
                                    "which are not shown on the mode content plot. These are the (2, 2, 6, -) mode at timestep"
                                    "t0 = 18.0 [M], and the (3, 2, 6, -) mode at timesteps t0 = 18.0 [M], t0 = 20.0 [M], and t0 = 28.0 [M]."
                                    "\n\n")
                            
                        elif sim_id == "0013":

                            f.write("Additional Notes\n----------------\n\n")
                            f.write("This simulation contains multiple instances of a retrograde mode present without a prograde mode, "
                                    "which, in this exceptional case, have been shown on a separate plot as dotted regions. Furthermore, "
                                    "this is the only simulation found to contain a constant offset, which is shown in pink on the plots."
                                    "\n\n")

                    for file in sorted(os.listdir(subfolder_path)):
                        if file.endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                            rel_path = os.path.relpath(os.path.join(subfolder_path, file), rst_dir)
                            f.write(f".. image:: {rel_path}\n   :width: 600px\n   :alt: {file}\n\n")

