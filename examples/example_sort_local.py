import os
from pathlib import Path

import numpy as np

from examples.utils import create_test_directory
from pykilosort import Bunch, add_default_handler, run
from spikeextractors.extractors import bindatrecordingextractor as dat

_dir_path = os.path.dirname(os.path.realpath(__file__))
test_dir = _dir_path+"/test/"
dat_path = Path(test_dir + "test.bin").absolute()
if not dat_path.is_file():
    create_test_directory(test_dir)

dir_path = dat_path.parent

recording = dat.BinDatRecordingExtractor(
        dat_path,
        sampling_frequency=3e4,
        numchan=4,
        dtype='int16'
)
#recording.write_to_binary_dat_format(str(dat_path))
n_channels = len(recording.get_channel_ids())


probe = Bunch()
probe.NchanTOT = n_channels
probe.chanMap = np.array(range(0, n_channels))
probe.kcoords = np.ones(n_channels)
probe.xc = np.load(dir_path.as_posix()+'/xc.npy')
probe.yc = np.load(dir_path.as_posix()+'/yc.npy')


add_default_handler(level="DEBUG")

params = {
    "nfilt_factor": 8,
    "AUCsplit": 0.85,
    "nskip": 5 
}

def run_example():
    run(
        dat_path,
        params=params,
        probe=probe,
        dir_path=dir_path,
        n_channels=probe.NchanTOT,
        dtype=np.int16,
        sample_rate=3e4 #recording.get_sampling_frequency(),
    )

if __name__ ==  "__main__":
    run_example()
