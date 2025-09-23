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

SPHERICAL_MODES_PES = [(2, 2), (3, 2), (4, 2)]

SPHERICAL_MODES_PS = [(2, 2), (3, 2),
                     (3, 3), (4, 3)] 

SPHERICAL_MODES_ALLS = SPHERICAL_MODES_PS + [(l, -m) for l, m in SPHERICAL_MODES_PS]

SPHERICAL_MODES_ES = SPHERICAL_MODES_PES + [(l, -m) for l, m in SPHERICAL_MODES_PES]

SPHERICAL_MODES_P = [(2, 2), (3, 2),
                     (3, 3), (4, 3), 
                     (4, 4), (5, 4),
                     (5, 5), (6, 5),
                     (6, 6), (7, 6)] 

SPHERICAL_MODES_ALL = SPHERICAL_MODES_P + [(l, -m) for l, m in SPHERICAL_MODES_P]

THRESHOLD = 0.9999
N_MAX = 6
T = 100
N_DRAWS = 1000
INCLUDE_CHIF = False
INCLUDE_MF = False

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

log_threshold = np.log(THRESHOLD)

FILENAME = f'mode_content_files/0013/mode_content_data'

t0_vals = np.arange(0, 60.1, 2)

def get_mode_list(sim_id, initial_modes, candidate_modes, spherical_modes, t0):
    
    sim = bgp.SXS_CCE(sim_id, type=DATA_TYPE, lev="Lev5", radius="R2")
    tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[sim_id]
    Mf, chif = sim.Mf, sim.chif_mag

    full_modes_list = [] 
    p_values_median = [] 
            
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
        candidate_type="prograde_sequential",
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

    sim_ids = ["0010"]

    for sim_id in sim_ids:

        candidate_mode_extras = [] 

        if SPH_MODE_RULES[sim_id] == "PES":
            spherical_modes = SPHERICAL_MODES_PES

        elif SPH_MODE_RULES[sim_id] == "PS":
            spherical_modes = SPHERICAL_MODES_PS

        elif SPH_MODE_RULES[sim_id] == "ES":
            spherical_modes = SPHERICAL_MODES_ES

        elif SPH_MODE_RULES[sim_id] == "ALLS":
            spherical_modes = SPHERICAL_MODES_ALLS

        elif SPH_MODE_RULES[sim_id] == "P":
            candidate_mode_extras = [(2,2,0,1,2,2,0,1), 
                                     (2,2,0,1,3,3,0,1),
                                     (3,3,0,1,3,3,0,1), 
                                     (2,2,0,1,4,4,0,1),
                                     (2,2,0,1,2,2,0,1,2,2,0,1)]
            spherical_modes = SPHERICAL_MODES_P

        elif SPH_MODE_RULES[sim_id] == "ALL":
            candidate_mode_extras = [(2,2,0,1,2,2,0,1), 
                                     (2,2,0,1,3,3,0,1),
                                     (3,3,0,1,3,3,0,1), 
                                     (2,2,0,1,4,4,0,1),
                                     (2,2,0,1,2,2,0,1,2,2,0,1)] + \
                                    [(2,-2,0,-1,2,-2,0,-1), 
                                     (2,-2,0,-1,3,-3,0,-1),
                                     (3,-3,0,-1,3,-3,0,-1), 
                                     (2,-2,0,-1,4,-4,0,-1),
                                     (2,-2,0,-1,2,-2,0,-1,2,-2,0,-1)]
                                     
            spherical_modes = SPHERICAL_MODES_ALL

        for m in np.arange(2, L_MAX):
            spherical_modes_m = [s for s in spherical_modes if s[1] == m]

            candidate_mode_extras_subset = [
                c for c in candidate_mode_extras
                if (len(c) == 8 and c[1] + c[5] == m) or
                   (len(c) == 12 and c[1] + c[5] + c[9] == m)
            ]

            initial_modes = [(*s, 0, 1 if s[1] > 0 else -1) for s in spherical_modes_m]
            candidate_modes = [(*s, n, 1) for s in spherical_modes_m for n in range(1, N_MAX + 1)] + \
                            [(*s, n, -1) for s in spherical_modes_m for n in range(0, N_MAX + 1)] + \
                            spherical_modes_m + candidate_mode_extras_subset
            
            print(f"Starting mode selection for simulation ID: {sim_id}")
            print(f"Using initial modes: {initial_modes}")
            print(f"Using spherical modes: {spherical_modes_m}")
            print(f"Candidate modes: {candidate_modes}")

            mode_selection_data = {
                "sim_id": sim_id,
                "times": t0_vals.tolist(),
                "spherical_modes": spherical_modes_m,
                "threshold": THRESHOLD,
                "initial_modes": initial_modes,
                "candidate_modes": candidate_modes,
                "include_chif": INCLUDE_CHIF,
                "include_Mf": INCLUDE_MF,
            }

            for t0 in t0_vals:
                mode_selection_data[f"modes"], mode_selection_data[f"p_values"] = get_mode_list(sim_id, initial_modes, candidate_modes, spherical_modes, t0)

                with open(f'{FILENAME}_{sim_id}_{m}_{t0}.json', 'w') as f:
                    json.dump(mode_selection_data, f)

if __name__ == "__main__":
    __main__()