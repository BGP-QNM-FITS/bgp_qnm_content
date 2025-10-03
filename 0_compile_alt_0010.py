import os
import json
import re
import bgp_qnm_fits as bgp 
import numpy as np 
from collections import defaultdict

even_file = "mode_content_files/mode_content_data_0010_even.json"
odd_file = "mode_content_files/mode_content_data_0010_odd.json"
output_file = "mode_content_files/mode_content_data_0010.json"

# Load both files
with open(even_file, "r") as f:
    even_data = json.load(f)
with open(odd_file, "r") as f:
    odd_data = json.load(f)

# Combine times and modes, keeping order
all_times = even_data["times"] + odd_data["times"]
all_modes = even_data["modes"] + odd_data["modes"]
all_p_values = even_data["p_values"] + odd_data["p_values"]

metadata = even_data.copy()

# Sort by t0
sorted_data = sorted(zip(all_times, all_modes, all_p_values), key=lambda x: x[0])
sorted_times, sorted_modes, sorted_p_values = zip(*sorted_data)
metadata["times"], metadata["modes"], metadata["p_values"] = list(sorted_times), list(sorted_modes), list(sorted_p_values)

with open(output_file, "w") as f:
    json.dump(metadata, f)

print(f"Compiled file written to {output_file}")