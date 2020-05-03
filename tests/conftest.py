from math import ceil
from pathlib import Path
from pytest import fixture

import numpy as np

from pykilosort.utils import Bunch, read_data

from pykilosort import add_default_handler
add_default_handler(level='DEBUG')


from math import ceil
from pathlib import Path
from pytest import fixture

import numpy as np


@fixture
def data_path():
    path = (Path(__file__).parent / '../../data/').resolve()
    assert path.exists()
    return path


@fixture
def dat_path(data_path):
    return data_path / 'imec_385_100s.bin'


@fixture
def raw_data(dat_path):
    # WARNING: Fortran order
    return read_data(dat_path, shape=(385, -1), dtype=np.int16)


@fixture
def params():

    np.random.seed(0)

    params = Bunch()

    params.fs = 30000.
    params.fshigh = 150.
    params.fslow = None
    params.ntbuff = 64
    params.NT = 65600
    params.NTbuff = params.NT + 4 * params.ntbuff  # we need buffers on both sides for filtering
    params.nSkipCov = 25
    params.whiteningRange = 32
    params.scaleproc = 200
    params.spkTh = -6
    params.nt0 = 61
    params.minfr_goodchannels = .1
    params.nfilt_factor = 4
    params.nt0 = 61
    params.nt0min = ceil(20 * params.nt0 / 61)

    return params


@fixture
def probe(data_path):
    probe = Bunch()
    probe.NchanTOT = 385
    # WARNING: indexing mismatch with MATLAB hence the -1
