from pathlib import Path
import numpy as np
from spikeextractors.extractors import bindatrecordingextractor as dat

from pykilosort import add_default_handler, run, Bunch

add_default_handler(level='DEBUG')

dat_path = Path('test/test.bin')
dir_path = dat_path.parent
probe = Bunch()
probe.NchanTOT = 4

probe.chanMap = np.load(dir_path / 'chanMap.npy').squeeze().astype(np.int64)
probe.xc = np.load(dir_path / 'xc.npy').squeeze()[:]
probe.yc = np.load(dir_path / 'yc.npy').squeeze()
probe.kcoords = np.load(dir_path / 'kcoords.npy').squeeze()

import pdb; pdb.set_trace()

run(dat_path, probe=probe, dir_path=dir_path, n_channels=probe.NchanTOT, dtype=np.int16, sample_rate=3e4)
