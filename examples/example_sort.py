from pathlib import Path

import numpy as np

import kachery as ka
from pykilosort import Bunch, add_default_handler, run
from spikeextractors.extractors import bindatrecordingextractor as dat
from spikeforest2_utils import AutoRecordingExtractor, AutoSortingExtractor

dat_path = Path("test/test.bin").absolute()
dir_path = dat_path.parent

ka.set_config(fr="default_readonly")
recording_path = "sha1dir://c0879a26f92e4c876cd608ca79192a84d4382868.manual_franklab/tetrode_600s/sorter1_1"
recording = AutoRecordingExtractor(recording_path, download=True)
recording.write_to_binary_dat_format(str(dat_path))
n_channels = len(recording.get_channel_ids())


probe = Bunch()
probe.NchanTOT = n_channels
probe.chanMap = np.array(range(0, n_channels))
probe.kcoords = np.ones(n_channels)
probe.xc = recording.get_channel_locations()[:, 0]
probe.yc = recording.get_channel_locations()[:, 1]


add_default_handler(level="DEBUG")

params = {
    "nfilt_factor": 8,
    "AUCsplit": 0.85,
    "nskip": 5 
}

run(
    dat_path,
    params=params,
    probe=probe,
    dir_path=dir_path,
    n_channels=probe.NchanTOT,
    dtype=np.int16,
    sample_rate=recording.get_sampling_frequency(),
)
