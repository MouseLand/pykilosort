#!/usr/bin/env python
# coding: utf-8

# In[1]:


import enum
import os
import numpy as np
import matplotlib.pyplot as plt


# In[38]:


import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# In[8]:


BASE_PATH = '/home/alexmorley/git_repos/pykilosort/examples/eMouse/data'
PYKILOSORT_SORTING_RESULTS_DIR = f'{BASE_PATH}/python_output/'
MATLAB_SORTING_RESULTS_DIR = f'{BASE_PATH}/matlab_output/'
[os.makedirs(d) for d in (PYKILOSORT_SORTING_RESULTS_DIR,MATLAB_SORTING_RESULTS_DIR)]

### Should checkout latest pykilosort & Kilosort2 master here
KILOSORT2_DIR = '/home/alexmorley/git_repos/Kilosort2/'
PYKILOSORT_DIR = '/home/alexmorley/git_repos/pykilosort/'

class Operations(enum.Enum):
    simulation = 'SIMULATION'
    matlab_sorting = 'MATLAB_SORTING'
    pykilosort_sorting = 'PYKILOSORT_SORTING'
    
FORCE_RUN = {Operations.simulation, Operations.matlab_sorting, Operations.pykilosort_sorting}

simulation_opts = {
    'chanMapName': 'chanMap_3B_64sites.mat',
    'NchanTOT': 64.0
}

opts = {
    'chanMap': f'{BASE_PATH}/{simulation_opts["chanMapName"]}',
    'fs': 30000.,
    'fshigh': 150.,
    'minfr_goodchannels': 0.1000,
    'Th': [6.0, 2.0],
    'lam': 10.,
    'AUCsplit': 0.9000,
    'minFR': 0.0200,
    'momentum': [20., 400],
    'sigmaMask': 30.,
    'ThPre': 8.,
    'reorder': 1,
    'nskip': 25.,
    'spkTh': -6.,
    'GPU': 1,
    'nfilt_factor': 4.,
    'ntbuff': 64.0,
    'NT': 65600.,
    'whiteningRange': 32.,
    'nSkipCov': 25.0,
    'scaleproc': 200.,
    'nPCs': 3.,
    'useRAM': 0,
    'sorting': 2,
    'NchanTOT': float(simulation_opts['NchanTOT']),
    'trange': [0., float('inf')],
    'fproc': '/tmp/temp_wh.dat',
    'rootZ': MATLAB_SORTING_RESULTS_DIR,
    'fbinary': f'{BASE_PATH}/sim_binary.imec.ap.bin',
    'fig': False
}  


# ## Setup MATLAB<sup>TM</sup> engine

# In[3]:


import matlab.engine

# If true we start a new matlab engine, if false we try to connect to an existing open matlab workspace.
# The latter is helpful for debugging.
new_session = False 
if new_session:
    eng = matlab.engine.start_matlab()
else:
    eng = matlab.engine.connect_matlab()
    
eng.addpath(eng.genpath(KILOSORT2_DIR));
eng.addpath(eng.genpath(f'{KILOSORT2_DIR}/../npy-matlab'));


# ## Generate simulated data using MATLAB<sup>TM</sup> 

# In[4]:


if Operations.simulation in FORCE_RUN:
    useGPU = True
    useParPool = False

    opts["chanMap"] = eng.make_eMouseChannelMap_3B_short(BASE_PATH, simulation_opts["NchanTOT"])
    opts["chanMap"] = f'{BASE_PATH}/{opts["chanMap"]}'
    eng.make_eMouseData_drift(BASE_PATH, KILOSORT2_DIR, simulation_opts["chanMapName"], useGPU, useParPool, nargout=0)
else:
    assert os.path.isfile(opts["chanMap"])


# Write out the channel data to numpy files too 

# In[5]:


x = eng.load(opts['chanMap'])
eng.writeNPY(x['chanMap'], f'{BASE_PATH}/chanMap.npy', nargout=0)
eng.writeNPY(x['xcoords'], f'{BASE_PATH}/xc.npy', nargout=0)
eng.writeNPY(x['ycoords'], f'{BASE_PATH}/yc.npy', nargout=0)


# ## Sort simulated data using Kilosort2 via MATLAB engine

# In[9]:


if Operations.matlab_sorting in FORCE_RUN:
    # make sure to convert list to matlab arrays
    ops = eng.struct({k: (matlab.double(v) if isinstance(v, list) else v) for k,v in opts.items()})
    rootZ = eng.char(opts['rootZ'])
    if not new_session: 
        eng.workspace['ops'] = ops
        eng.workspace['rootZ'] = rootZ

    rez = eng.function_kilosort(rootZ, ops)
os.listdir(MATLAB_SORTING_RESULTS_DIR)


# ## Sort simulated data using pykilosort

# In[40]:


import pykilosort
from pathlib import Path
from importlib import reload
from pykilosort import main
reload(main)

pykilosort.add_default_handler()


# In[35]:


np.load(PYKILOSORT_SORTING_RESULTS_DIR + '/.kilosort/sim_binary.imec.ap/igood.npy')


# In[43]:


probe = pykilosort.Bunch()
probe.NchanTOT = int(opts['NchanTOT'])
probe.chanMap = np.load(BASE_PATH+'/chanMap.npy').flatten().astype(int)
probe.kcoords = np.ones(int(opts['NchanTOT']))
probe.xc = np.load(BASE_PATH+'/xc.npy').flatten()
probe.yc = np.load(BASE_PATH+'/yc.npy').flatten()

rez = main.run(
    dat_path = opts['fbinary'],
    dir_path = Path(PYKILOSORT_SORTING_RESULTS_DIR),
    output_path = Path(PYKILOSORT_SORTING_RESULTS_DIR),
    params = None,
    probe=probe,
    dtype = np.int16,
    n_channels = int(opts['NchanTOT']),
    sample_rate = opts['fs'],
    clear_context = True,
)


# In[52]:


type(np.array)


# In[55]:


current_dir = MATLAB_SORTING_RESULTS_DIR

import pydantic

class SortingResults(pydantic.BaseModel):
    templates: np.ndarray
    spike_times: np.ndarray
    channel_positions: np.ndarray
        
    class Config:
        arbitrary_types_allowed = True

def get_results(dirname):
    return SortingResults(
        templates = np.load(f"{dirname}/templates.npy"),
        spike_times = np.load(f"{dirname}/spike_times.npy"),
        channel_positions = np.load(f"{dirname}/channel_positions.npy"),
    )

results = {
    'matlab': get_results(MATLAB_SORTING_RESULTS_DIR),
    'python': get_results(PYKILOSORT_SORTING_RESULTS_DIR)
}


# In[106]:


matlab_templates = [results['matlab'].templates[x,:,:].ravel() for x in range(results['matlab'].templates.shape[0])]
python_templates = [results['python'].templates[x,:,:].ravel() for x in range(results['python'].templates.shape[0])]

similarity_matrix = np.array([[np.dot(m, p) for p in python_templates] for m in matlab_templates])

plt.figure(figsize=(13,7))
plt.imshow(similarity_matrix, vmin=0, vmax=1)
plt.xlabel('Python Units')
plt.ylabel('MATLAB Units')

plt.colorbar()


# In[68]:


f, axs = plt.subplots(2, 2, figsize=(10,15))

for i, (name,res) in enumerate(results.items()):
    templates = res.templates
    unit = 2

    for channel in range(templates.shape[2]):
        axs[0, i].plot(templates[unit,:,channel].T + 0.1 * channel);
    
    axs[0, i].set_title(name)


# In[84]:


for unit in range(82):
    plt.plot(templates[unit,:,0].T);


# In[39]:


templates[:,0,:].shape


# ## 
