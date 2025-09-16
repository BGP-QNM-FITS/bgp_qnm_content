"""
This script performs the selection algorithm and saves the relevant data as a JSON for 
use in the plotting scripts. 

"""

import numpy as np
import json
import bgp_qnm_fits as bgp
import time

DATA_TYPE = 'news'
L_MAX = 7

SPHERICAL_MODES_PE = [(2, 2),
                      (3, 2), 
                      (4, 4), (4, 2),
                      (5, 4), (5, 2),
                      (6, 6), (6, 4),
                      (7, 6)]


SPHERICAL_MODES_P = [(2, 2), (2, 1),
                    (3, 3), (3, 2),
                    (4, 4), (4, 3),
                    (5, 5), (5, 4),
                    (6, 6), (6, 5),
                    (7, 7), (7, 6)]

SPHERICAL_MODES_E = SPHERICAL_MODES_PE + [(l, -m) for l, m in SPHERICAL_MODES_PE]
SPHERICAL_MODES_ALL = SPHERICAL_MODES_P + [(l, -m) for l, m in SPHERICAL_MODES_P]

SPHERICAL_MODES_EVERYTHING_PE = [(2, 2),
                                (3, 2), 
                                (4, 4), (4, 2),
                                (5, 4), (5, 2),
                                (6, 6), (6, 4), (6, 2),
                                (7, 6), (7, 4), (7, 2)]

SPHERICAL_MODES_EVERYTHING = [(l, m) for l in range(2, L_MAX + 1) for m in range(-l, l + 1)]

THRESHOLD = 0.9999
N_MAX = 6
T = 100
N_DRAWS = 1000
INCLUDE_CHIF = False
INCLUDE_MF = False

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

log_threshold = np.log(THRESHOLD)

FILENAME = f'mode_content_files/mode_content_data'

t0_vals = np.arange(0, 80.1, 1)

def get_mode_list(sim_id, initial_modes, candidate_modes, spherical_modes):
    
    sim = bgp.SXS_CCE(sim_id, type=DATA_TYPE, lev="Lev5", radius="R2")
    tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[sim_id]
    Mf, chif = sim.Mf, sim.chif_mag

    full_modes_list = [] 
    p_values_median = [] 

    for t0 in t0_vals:
            
        print(f'Fitting from t0={t0}')
        
        select_object = bgp.BGP_select(
            sim.times,
            sim.h,
            initial_modes,
            Mf,
            chif,
            tuned_param_dict_GP,
            bgp.kernel_GP,
            t0=t0,
            candidate_modes=candidate_modes,
            log_threshold=log_threshold,
            candidate_type="sequential",
            num_draws=N_DRAWS,
            T=T,
            spherical_modes=spherical_modes,
            include_chif=INCLUDE_CHIF,
            include_Mf=INCLUDE_MF,
            data_type=DATA_TYPE
        )

        full_modes_list.append(select_object.full_modes)
        p_values_median.append(np.median(select_object.p_values)) 

    return full_modes_list, p_values_median


def __main__():
    #sim_ids = [f"{i:04}" for i in range(1, 14)]
    #sim_ids = ["0001"]

    sim_ids = ["0010"] 

    for sim_id in sim_ids:

        if SPH_MODE_RULES[sim_id] == "PE":
            spherical_modes = SPHERICAL_MODES_PE
        elif SPH_MODE_RULES[sim_id] == "P":
            spherical_modes = SPHERICAL_MODES_P
        elif SPH_MODE_RULES[sim_id] == "E":
            spherical_modes = SPHERICAL_MODES_E
        elif SPH_MODE_RULES[sim_id] == "ALL":
            spherical_modes = SPHERICAL_MODES_ALL 

        initial_modes = [(*s, 0, 1) for s in spherical_modes]
        candidate_modes = [(*s, n, 1) for s in spherical_modes for n in range(0, N_MAX + 1)] + \
                        [(*s, n, -1) for s in spherical_modes for n in range(0, N_MAX + 1)] + \
                        spherical_modes + \
                        [
                            (2, 2, 0, 1, 2, 2, 0, 1),
                            (3, 3, 0, 1, 3, 3, 0, 1),
                            (2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1)
                        ]

        mode_selection_data = {
            "sim_id": sim_id,
            "times": t0_vals.tolist(),
            "spherical_modes": spherical_modes,
            "threshold": THRESHOLD,
            "initial_modes": initial_modes,
            "candidate_modes": candidate_modes,
        }
        print(f"Processing simulation ID: {sim_id}")
        start_time = time.time()
        mode_selection_data[f"modes"], mode_selection_data[f"p_values"] = get_mode_list(sim_id, initial_modes, candidate_modes, spherical_modes)
        end_time = time.time()
        mode_selection_data[f"run_time"] = end_time - start_time

        with open(f'{FILENAME}_{sim_id}.json', 'w') as f:
            json.dump(mode_selection_data, f)

if __name__ == "__main__":
    __main__()