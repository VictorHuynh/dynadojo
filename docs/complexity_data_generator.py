from dynadojo.systems.gilpin_flows import GilpinFlowsSystem
from dynadojo.wrappers import SystemChecker
from dynadojo.utils.lds import plot
from dynadojo.utils.complexity_measures import gp_dim, mse_mv, pca, find_lyapunov_exponents, find_max_lyapunov, kaplan_yorke_dimension, pesin

import json
import os
import dysts
import numpy as np

base_path = os.path.dirname(dysts.__file__)
json_file_path = os.path.join(base_path, 'data', 'chaotic_attractors.json')
with open(json_file_path, 'r') as file:
    systems_data = json.load(file)
all_systems = list(systems_data.keys())

problematic_systems = ["IkedaDelay", "MackeyGlass", "PiecewiseCircuit", "ScrollDelay", "SprottDelay", "VossDelay", "Torus"]
for problematic_system in problematic_systems:
    all_systems.remove(problematic_system)

all_measures = ["gp_dim", "mse_mv", "pca", "max_lyapunov", "kaplan_yorke_dimension", "pesin"]

dictionary = {}
for measure in all_measures:
    dictionary[measure] = {}
for system_name in all_systems:
    for measure in all_measures:
        dictionary[measure][system_name] = []

latent_dim = 3
embed_dim = latent_dim

seeds = []
timesteps_list = []
for i in range(3):
    seeds.append(i + 1)
    timesteps_list.append((i + 1)*1000)

for system_name in all_systems:
    print()
    print("### WORKING ON:", system_name, "###")
    print()
    for seed_number in seeds:
        for timestep_number in timesteps_list:

            print()
            print("     ### CURRENTLY:", seed_number, ",", timestep_number, "###")
            print()

            system = SystemChecker(GilpinFlowsSystem(latent_dim, embed_dim, system_name, seed=seed_number))
            x0 = system.make_init_conds(1)
            x = system.make_data(x0, timesteps=timestep_number)

            dictionary["gp_dim"][system_name].append( (gp_dim(x[0]), timestep_number, seed_number) )
            dictionary["mse_mv"][system_name].append( (mse_mv(x[0]), timestep_number, seed_number) )
            dictionary["pca"][system_name].append( (pca(x[0]), timestep_number, seed_number) )

            model = system._system.system
            lyapunov_spectrum = find_lyapunov_exponents(model, timestep_number)
            dictionary["max_lyapunov"][system_name].append( (find_max_lyapunov(lyapunov_spectrum), timestep_number, seed_number) )
            dictionary["kaplan_yorke_dimension"][system_name].append( (kaplan_yorke_dimension(lyapunov_spectrum), timestep_number, seed_number) )
            dictionary["pesin"][system_name].append( (pesin(lyapunov_spectrum), timestep_number, seed_number) )

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

# Serializing json
json_object = json.dumps(dictionary, cls=NumpyEncoder, indent=4)

# Writing to sample.json
with open("docs/complexity_data.JSON", "w") as outfile:
    outfile.write(json_object)