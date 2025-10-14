import os
import json
from pathlib import Path
import re
import bgp_qnm_fits as bgp
import numpy as np
from collections import defaultdict
from tqdm import tqdm

SIM_ID = "0013"

main_dir = Path("mode_content_files")
path = main_dir / f"mode_content_data_{SIM_ID}.json"
output_file = path

with open(path, "r") as fh:
    metadata = json.load(fh)

# basic validation
times = list(metadata.get("times", []))
modes = list(metadata.get("modes", []))
spherical_modes = [tuple(m) for m in metadata.get("spherical_modes", [])]

if len(times) != len(modes):
    raise RuntimeError("times and modes length mismatch in metadata")

sim = bgp.SXS_CCE(SIM_ID, type="news", lev="Lev5", radius="R2")
tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type="news").get(SIM_ID)

full_modes_list = [list(map(tuple, inner_list)) for inner_list in modes]

p_values_median = []

print(f"Computing p-values for {len(times)} times for sim {SIM_ID}")

for i, t0 in enumerate(tqdm(times, desc="times")):
    select_modes = full_modes_list[i]
    fit = bgp.BGP_fit(
        sim.times,
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
        data_type="news",
    )
    pv = fit.fit.get("p_values", None)
    median_p = float(np.median(pv))
    p_values_median.append(median_p)
    print(median_p)

# write back
metadata["p_values"] = p_values_median

with open(output_file, "w") as fh:
    json.dump(metadata, fh, indent=2)

print(f"Wrote compiled file with updated p_values to {output_file}")