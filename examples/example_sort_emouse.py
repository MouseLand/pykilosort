import os
from pathlib import Path

import numpy as np

from examples.utils import create_test_directory
from pykilosort import Bunch, add_default_handler, run
from spikeextractors.extractors import bindatrecordingextractor as dat

# Config
_dir_path = os.path.dirname(os.path.realpath(__file__))
test_dir = _dir_path+"/eMouse/data/"
dat_path = Path(test_dir + "sim_binary.imec.ap.bin").absolute()

# Set up (and download if necessary) a test recording
if not dat_path.is_file():
    create_test_directory(test_dir)
dir_path = dat_path.parent

# Use spikeinterface to get a consistent interface to the recording
recording = dat.BinDatRecordingExtractor(
        dat_path,
        sampling_frequency=3e4,
        numchan=64,
        dtype='int16'
)
n_channels = len(recording.get_channel_ids())

# Load the config into pykilosort2.
# This should be automatic in the future.
probe = Bunch()
probe.NchanTOT = n_channels
probe.chanMap = np.load(dir_path.as_posix()+'/chanMap.npy').flatten().astype(int)
probe.kcoords = np.ones(n_channels)
probe.xc = np.load(dir_path.as_posix()+'/xc.npy').flatten()
probe.yc = np.load(dir_path.as_posix()+'/yc.npy').flatten()


add_default_handler(level="DEBUG")

#params = {
#    "nfilt_factor": 8,
#    "AUCsplit": 0.85,
#    "nskip": 5 
#}
params = None #{}

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

# This can be run as a script or as part of a test
if __name__ ==  "__main__":
    run_example()
