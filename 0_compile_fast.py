import json
import glob
import re

# Find all relevant JSON files
json_files = glob.glob("mode_content_files/mode_content_data_*.json")

compiled = {
    "sim_id": None,
    "times": [],
    "spherical_modes": [],
    "threshold": None,
    "initial_modes": [],
    "candidate_modes": [],
    "include_chif": None,
    "include_Mf": None,
    "modes": [],
    "p_values": [],
    "run_time": 0.0,
}

for fname in json_files:
    with open(fname, "r") as f:
        data = json.load(f)
    # Set shared metadata from the first file
    if compiled["sim_id"] is None:
        compiled["sim_id"] = data["sim_id"]
        compiled["threshold"] = data["threshold"]
        compiled["include_chif"] = data["include_chif"]
        compiled["include_Mf"] = data["include_Mf"]
    # Merge lists
    compiled["spherical_modes"].extend(data["spherical_modes"])
    compiled["initial_modes"].extend(data["initial_modes"])
    compiled["candidate_modes"].extend(data["candidate_modes"])
    # Extract t0 from filename (assuming pattern ..._{sim_id}_{m}_{t0}.json)
    match = re.search(r'_(\d+)_(\d+)_(\d+\.?\d*)\.json$', fname)
    if match:
        t0 = float(match.group(3))
        if t0 not in compiled["times"]:
            compiled["times"].append(t0)
    # Each file contains only one time step
    compiled["modes"].extend(data["modes"])
    compiled["p_values"].extend(data["p_values"])
    compiled["run_time"] += data.get("run_time", 0.0)

# Remove duplicates and sort times
compiled["spherical_modes"] = list({tuple(x) for x in compiled["spherical_modes"]})
compiled["initial_modes"] = list({tuple(x) for x in compiled["initial_modes"]})
compiled["candidate_modes"] = list({tuple(x) for x in compiled["candidate_modes"]})
compiled["times"] = sorted(compiled["times"])

# Save the compiled file
with open(f"mode_content_files/mode_content_data_compiled_{compiled['sim_id']}.json", "w") as f:
    json.dump(compiled, f)