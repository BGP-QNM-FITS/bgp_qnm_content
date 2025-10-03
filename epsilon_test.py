import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.ticker import FixedLocator
import json
import bgp_qnm_fits as bgp

special_color_1 = to_hex("#8B5FBF")
special_color_2 = to_hex("#C26C88")
special_color_3 = to_hex("#DE6A5E")

DATA_TYPE = 'news'
T = 100
INCLUDE_CHIF = True
INCLUDE_MF = True

NUM_SAMPLES = 1000

sim_id = "0012"

with open(f'mode_content_files/mode_content_data_{sim_id}.json', 'r') as f:
            mode_content_data_dict = json.load(f)

t0_vals = np.array(mode_content_data_dict['times'])
spherical_modes = [tuple(mode) for mode in mode_content_data_dict['spherical_modes']]

sim = bgp.SXS_CCE(sim_id, type=DATA_TYPE, lev="Lev5", radius="R2")
Mf_ref, chif_ref = sim.Mf, sim.chif_mag

full_modes_list = [list(map(tuple, inner_list)) for inner_list in mode_content_data_dict["modes"]]

sim = bgp.SXS_CCE(sim_id, type=DATA_TYPE, lev="Lev5", radius="R2")
tuned_param_dict_GP = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[sim_id]
unique_modes = {mode for modes in full_modes_list for mode in modes}

fits_full = bgp.BGP_fit_lite(sim.times, 
                                sim.h, 
                                unique_modes, 
                                sim.Mf, 
                                sim.chif_mag, 
                                tuned_param_dict_GP, 
                                bgp.kernel_GP, 
                                t0=18, 
                                T=T, 
                                num_samples=NUM_SAMPLES,
                                spherical_modes = spherical_modes,
                                include_chif=INCLUDE_CHIF,
                                include_Mf=INCLUDE_MF,
                                data_type=DATA_TYPE)