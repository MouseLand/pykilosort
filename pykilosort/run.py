from pathlib import Path
import os
from pykilosort import Bunch, run
import numpy as np


def run_kilosort(dat_path, delete=False):
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
        os.system(f'rm -rf {str(dat_path.parent / "output" / "pc_features.npy")}')
        os.system(f'rm -rf {str(dat_path.parent / "output" / "template_features.npy")}')
