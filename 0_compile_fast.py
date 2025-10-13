import os
import json
import re
import bgp_qnm_fits as bgp 
import numpy as np 
from collections import defaultdict

sim_id = "0010"

input_dir = f"mode_content_fast/{sim_id}_rerun"
output_file = f"mode_content_compiled/mode_content_data_{sim_id}_6.json"

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
    match = re.match(fr"mode_content_data_{sim_id}_(\d+)_(\d+\.?\d*)\.json", fname)
    if not match:
        continue
    m = int(match.group(1))
    t0 = float(match.group(2))
    with open(os.path.join(input_dir, fname), "r") as f:
        data = json.load(f)
    # Save metadata from the first file
    if first_file:
        metadata["sim_id"] = sim_id
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
    match = re.match(fr"mode_content_data_{sim_id}_[-m](\d+)_(\d+\.?\d*)\.json", fname)
    if not match:
        continue
    m = -int(match.group(1))
    t0 = float(match.group(2))
    with open(os.path.join(input_dir, fname), "r") as f:
        data = json.load(f)
    # Save metadata from the first file
    if first_file:
        metadata["sim_id"] = sim_id
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

full_modes_list = [list(map(tuple, inner_list)) for inner_list in metadata["modes"]]
spherical_modes = [tuple(mode) for mode in metadata["spherical_modes"]]

with open(output_file, "w") as f:
    json.dump(metadata, f)

print(f"Compiled file written to {output_file}")