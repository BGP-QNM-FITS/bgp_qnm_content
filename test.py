import numpy as np
from matplotlib import pyplot as plt
import bgp_qnm_fits as bgp
import corner

# Constants
ID = '0013'
DATA_TYPE = 'news'
SPHERICAL_MODES = [(2, 2), (3, 2)]
THRESHOLD = 0.9
N_MAX = 6
t0 = 10
T = 100
N_DRAWS = 10000
INCLUDE_CHIF = False
INCLUDE_MF = False
t0_vals = np.arange(0, 40.1, 2)

sim = bgp.SXS_CCE(ID, type=DATA_TYPE, lev="Lev5", radius="R2")
tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[ID]
Mf, chif = sim.Mf, sim.chif_mag

import json 
sim_id = '0013' 
t0 = 20 

full_modes_list = [] 
p_values_median = [] 

with open(f'mode_content_files/mode_content_data_{sim_id}.json', 'r') as f:
    mode_content_data_dict = json.load(f)

full_modes_list = [list(map(tuple, inner_list)) for inner_list in mode_content_data_dict["modes"]]
unique_modes = list({mode for modes in full_modes_list for mode in modes})
spherical_modes = [tuple(mode) for mode in mode_content_data_dict['spherical_modes']]
t0_vals = np.array(mode_content_data_dict['times'])

fits = [] 

i = np.where(t0_vals == t0)[0][0]

select_modes = full_modes_list[i]
fits.append(bgp.BGP_fit(sim.times, 
                            sim.h, 
                            select_modes, 
                            sim.Mf, 
                            sim.chif_mag, 
                            tuned_param_dict_GP, 
                            bgp.kernel_GP, 
                            t0=t0, 
                            T=T, 
                            decay_corrected=True,
                            strain_parameters=True,
                            num_samples=1000,
                            spherical_modes = spherical_modes,
                            include_chif=INCLUDE_CHIF,
                            include_Mf=INCLUDE_MF,
                            data_type=DATA_TYPE)
            )

i = full_modes_list[i].index((2, 2))

const_amps = fits[0].fit["sample_amplitudes"][:, i]