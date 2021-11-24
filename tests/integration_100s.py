import shutil
from pathlib import Path
import numpy as np

import pykilosort
from deploy.serverpc.kilosort2.run_pykilosort import run_spike_sorting_ibl, ibl_pykilosort_params

INTEGRATION_DATA_PATH = Path("/datadisk/Data/spike_sorting/pykilosort_tests")
SCRATCH_DIR = Path.home().joinpath("scratch", 'pykilosort')
shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
SCRATCH_DIR.mkdir(exist_ok=True)
DELETE = False  # delete the intermediate run products, if False they'll be copied over
# bin_file = INTEGRATION_DATA_PATH.joinpath("imec_385_100s.ap.bin")

params = ibl_pykilosort_params()
label = ""
# params['preprocessing_function'] = 'kilosort2'
cluster_times_path = INTEGRATION_DATA_PATH.joinpath("cluster_times")

MULTIPARTS = False
if MULTIPARTS:
    bin_file = list(INTEGRATION_DATA_PATH.rglob("imec_385_100s.ap.cbin"))
    bin_file.sort()
    # _make_compressed_parts(bin_file)
    ks_output_dir = INTEGRATION_DATA_PATH.joinpath(
        f"{pykilosort.__version__}" + label, bin_file[0].name.split('.')[0] + 'multi_parts')
else:
    bin_file = INTEGRATION_DATA_PATH.joinpath("imec_385_100s.ap.cbin")
    ks_output_dir = INTEGRATION_DATA_PATH.joinpath(f"{pykilosort.__version__}" + label, bin_file.name.split('.')[0])


ks_output_dir.mkdir(parents=True, exist_ok=True)
alf_path = ks_output_dir.joinpath('alf')

run_spike_sorting_ibl(bin_file, delete=DELETE, scratch_dir=SCRATCH_DIR, neuropixel_version=1,
                      ks_output_dir=ks_output_dir, alf_path=alf_path, log_level='DEBUG', params=params)

if DELETE == False:
    import shutil
    from easyqc.gui import viewseis
    working_directory = SCRATCH_DIR.joinpath('.kilosort', bin_file.name)
    pre_proc_file = working_directory.joinpath('proc.dat')
    intermediate_directory = ks_output_dir.joinpath('intermediate')
    intermediate_directory.mkdir(exist_ok=True)

    shutil.copy(pre_proc_file, intermediate_directory)

    # PosixPath('/home/olivier/scratch/pykilosort/.pykilosort/imec_385_100s.ap.cbin')

##
import matplotlib.pyplot as plt
import one.alf.io as alfio
from brainbox.plot import driftmap
from pathlib import Path
from ibllib.io import spikeglx
import numpy as np
import scipy.signal
from brainbox.metrics.single_units import quick_unit_metrics
from easyqc.gui import viewseis
INTEGRATION_DATA_PATH = Path("/datadisk/Data/spike_sorting/pykilosort_tests")
plt.close('all')
runs = list(INTEGRATION_DATA_PATH.rglob('imec_385_100s'))


import pandas as pd
csv = []
eqcs = []
raw = False
for i, run in enumerate(runs):
    run_label = run.parts[-2]
    if run.joinpath('intermediate').exists():
        bin_file = next(INTEGRATION_DATA_PATH.rglob("imec_385_100s.ap.cbin"))
        sr = spikeglx.Reader(bin_file)
        pre_proc_file = run.joinpath('intermediate', 'proc.dat')
        nc = 384
        ns = int(pre_proc_file.stat().st_size / 2 / nc)
        mmap = np.memmap(pre_proc_file, dtype=np.int16, mode='r', shape=(ns, nc))
        start = 0
        end = start + 80000
        print(run)
        eqcs.append(viewseis(mmap[start:end, :], si=1 / 30, taxis=0, title=run.parts[-2] + 'tap'))
        if raw is False:
            raw = sr[start:end, :-1].T
            from ibllib.plots.figures import ephys_bad_channels
            from ibllib.dsp.voltage import detect_bad_channels
            labels, features = detect_bad_channels(raw, fs=30000)
            ffig, eeqcs = ephys_bad_channels(raw, 30000, labels, features, title='integration_100s', save_dir=INTEGRATION_DATA_PATH)
        break
    fig_file = INTEGRATION_DATA_PATH.joinpath('_'.join(run.parts[-2:]) + '.png')
    # if fig_file.exists():
    #     continue
    fig, ax = plt.subplots(figsize=(12, 8))
    spikes = alfio.load_object(run.joinpath('alf'), 'spikes')
    clusters = alfio.load_object(run.joinpath('alf'), 'clusters')
    clusters.keys()
    driftmap(spikes['times'], spikes['depths'], plot_style='bincount', t_bin=0.1, d_bin=20,  vmax=5, ax=ax)
    ax.set(title=f"{run_label}")

    nspi = spikes.times.size
    nclu = clusters.channels.size
    qc = quick_unit_metrics(spikes['clusters'], spikes['times'], spikes['amps'], spikes['depths'])
    fig.savefig(fig_file)
    csv.append(dict(label=run_label, sorted=True, nspikes=nspi, nclusters=nclu, quality=np.mean(qc.label)))
pd.DataFrame(csv)
