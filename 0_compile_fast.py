import os
import json
import re
import bgp_qnm_fits as bgp 
import numpy as np 
from collections import defaultdict

input_dir = "0013_mode_content_files"
output_file = "mode_content_files/mode_content_data_0013.json"

# Prepare containers for each t0
t0_data = defaultdict(list)
metadata = {}
metadata["spherical_modes"] = [] 
metadata["initial_modes"] = []
metadata["candidate_modes"] = []
first_file = True
done_m = set()

# Scan all files
for fname in sorted(os.listdir(input_dir)):
    match = re.match(r"mode_content_data_0013_(\d+)_(\d+\.?\d*)\.json", fname)
    if not match:
        continue
    m = int(match.group(1))
    t0 = float(match.group(2))
    with open(os.path.join(input_dir, fname), "r") as f:
        data = json.load(f)
    # Save metadata from the first file
    if first_file:
        metadata["sim_id"] = "0013"
        metadata["times"] = data["times"]
        metadata["threshold"] = data["threshold"]
        metadata["include_chif"] = data["include_chif"]
        metadata["include_Mf"] = data["include_Mf"]
        first_file = False

    if m not in done_m:
        metadata["spherical_modes"].extend(data["spherical_modes"])
        metadata["initial_modes"].extend(data["initial_modes"])
        metadata["candidate_modes"].extend(data["candidate_modes"])
        done_m.add(m)

    t0_data[t0].extend(data["modes"][0])


# Scan all negative m files 
for fname in sorted(os.listdir(input_dir)):
    match = re.match(r"mode_content_data_0013_m(\d+)_(\d+\.?\d*)\.json", fname)
    if not match:
        continue
    m = -int(match.group(1))
    t0 = float(match.group(2))
    with open(os.path.join(input_dir, fname), "r") as f:
        data = json.load(f)
    # Save metadata from the first file
    if first_file:
        metadata["sim_id"] = "0013"
        metadata["times"] = data["times"]
        metadata["threshold"] = data["threshold"]
        metadata["include_chif"] = data["include_chif"]
        metadata["include_Mf"] = data["include_Mf"]
        first_file = False

    if m not in done_m:
        metadata["spherical_modes"].extend(data["spherical_modes"])
        metadata["initial_modes"].extend(data["initial_modes"])
        metadata["candidate_modes"].extend(data["candidate_modes"])
        done_m.add(m)

    t0_data[t0].extend(data["modes"][0])

# Build output lists
compiled_modes = []
for t0 in metadata["times"]:
    modes_at_t0 = t0_data.get(t0, [])
    compiled_modes.append(modes_at_t0)

metadata["modes"] = compiled_modes

sim = bgp.SXS_CCE("0013", type="news", lev="Lev5", radius="R2")
tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type="news")["0013"]

p_values_median = []

full_modes_list = [list(map(tuple, inner_list)) for inner_list in metadata["modes"]]
spherical_modes = [tuple(mode) for mode in metadata["spherical_modes"]]

for i, t0 in enumerate(metadata["times"]): 

    print(f'Fitting from {t0=}')

    select_modes = full_modes_list[i]

    fit = bgp.BGP_fit(sim.times, 
                        sim.h, 
                        select_modes, 
                        sim.Mf, 
                        sim.chif_mag, 
                        tuned_param_dict_GP, 
                        bgp.kernel_GP, 
                        t0=t0, 
                        T=100, 
                        spherical_modes=spherical_modes,
                        include_chif=False,
                        include_Mf=False,
                        data_type="news")

    p_values_median.append(np.median(fit.fit["p_values"]))

metadata["p_values"] = p_values_median

with open(output_file, "w") as f:
    json.dump(metadata, f)

print(f"Compiled file written to {output_file}")