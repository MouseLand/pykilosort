import os
import yaml

import numpy as np


def get_test_data(test_data_folder, sets):
    values = []
    varnames = None
    for test_set in sets:
        folder = f"{test_data_folder}/{test_set}"
        test_data = {
            filename.split('.csv')[0] : np.loadtxt(f"{folder}/{filename}", delimiter=',') for filename in os.listdir(folder) if filename.endswith(".csv")
        }
        with open(f"{folder}/vars.yml") as f:
            test_data.update(yaml.load(f))
        varnames = ','.join(test_data.keys())
        values.append(tuple(v for v in test_data.values()))
    return varnames, values



