from pathlib import Path
import os
from pykilosort import Bunch, run
import numpy as np

from ibllib.io import spikeglx
from ibllib.ephys.spikes import ks2_to_alf


def _sample2v(ap_file):
    md = spikeglx.read_meta_data(ap_file.with_suffix(".meta"))
    s2v = spikeglx._conversion_sample2v_from_meta(md)
    return s2v["ap"][0]


def run_kilosort(dat_path, delete=False):
    # todo redirect log output
    dat_path = Path(dat_path)
    probe = Bunch()
    probe.NchanTOT = 385
    probe.chanMap = np.arange(384)
    probe.xc = np.tile(np.array([43., 11., 59., 27.]), 96)
    probe.yc = np.repeat(np.arange(20, 3841, 20), 2)
    probe.kcoords = np.zeros(384)
    run(dat_path, probe=probe, dir_path=dat_path.parent, n_channels=385, dtype=np.int16,
        sample_rate=3e4)

    if delete:
        os.system(f'rm -rf {str(dat_path.parent / ".kilosort")}')

# it's very useful to be able to split the temp path from the data path. In IBL we have a dedicated scratch drive for ks2 (SSD)
data_files = [
    "/datadisk/FlatIron/angelakilab/Subjects/NYU-23/2020-10-15/002/raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec1.ap.cbin",  # 0
    "/datadisk/FlatIron/angelakilab/Subjects/NYU-23/2020-10-15/002/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.cbin",  # 1 ok
    "/datadisk/FlatIron/churchlandlab/Subjects/CSHL047/2020-01-20/001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec.ap.cbin",  # 2 ok
    '/media/olivier/Nunux/sdsc/zadorlab/Subjects/CSH_ZAD_025/2020-07-29/001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.cbin',  # 3 ok - full
    '/media/olivier/Nunux/sdsc/zadorlab/Subjects/CSH_ZAD_025/2020-07-29/001/raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec1.ap.cbin',  # 4 ok
    '/media/olivier/Nunux/sdsc/zadorlab/Subjects/CSH_ZAD_029/2020-09-09/001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.cbin',  # 5 ok - full save + fkrun
    '/media/olivier/Nunux/sdsc/zadorlab/Subjects/CSH_ZAD_029/2020-09-09/001/raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec1.ap.cbin',  # 6
    '/media/olivier/Nunux/sdsc/wittenlab/2020-03-15/002/raw_ephys_data/probe00/_iblrig_ephysData.raw_g0_t0.imec.ap.cbin'   # 7 bonus witten - baseline
]

import time
label = 'baseline'
DELETE = True
scratch_dir = Path('/home/olivier/scratch')

for index in [6, 0]:
    tstart = time.time()

    dat_path = Path(data_files[index])
    print(dat_path)

    folder_name = '_'.join(list(dat_path.parts[-6:-3]) + [dat_path.parts[-2]])
    temp_folder = scratch_dir.joinpath(folder_name)
    temp_folder.mkdir(exist_ok=True, parents=True)
    link_file = temp_folder.joinpath(dat_path.name)
    if not link_file.exists():
        link_file.symlink_to(dat_path)
    if not link_file.with_suffix('.ch').exists():
        link_file.with_suffix('.ch').symlink_to(dat_path.with_suffix('.ch'))
    if not link_file.with_suffix('.meta').exists():
        link_file.with_suffix('.meta').symlink_to(dat_path.with_suffix('.meta'))

    run_kilosort(link_file, delete=DELETE)
    print(f"ran in {time.time() - tstart} seconds")

    ks_path = temp_folder.joinpath('output')
    alf_path = temp_folder.joinpath('alf')
    s2v = _sample2v(next(temp_folder.glob('*.ap.cbin')))
    ks2_to_alf(ks_path, temp_folder, alf_path, ampfactor=s2v)

    print(f"mkdir -p /media/olivier/Nunux/spike_sorting/{folder_name}/{label}")

    cmd_cp_kilosort = f"rsync -av --progress {temp_folder.joinpath('.kilosort')} /media/olivier/Nunux/spike_sorting/{folder_name}/{label}"
    cmd_cp_alf = f"rsync -av --progress {temp_folder.joinpath('alf')} /media/olivier/Nunux/spike_sorting/{folder_name}/{label}"
    cmd_cp_output = f"rsync -av --progress {temp_folder.joinpath('output')} /media/olivier/Nunux/spike_sorting/{folder_name}/{label}"

    print(cmd_cp_output)
    os. system(cmd_cp_output)
    print(cmd_cp_alf)
    os. system(cmd_cp_alf)
    if not DELETE:
        print(cmd_cp_kilosort)
        os. system(cmd_cp_kilosort)
