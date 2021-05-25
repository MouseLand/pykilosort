import numpy as np

from pykilosort.datashift2 import shift_data, align_block2, apply_drift_transform
from pykilosort.utils import Bunch

def test_shift_data():
    data = np.asfortranarray(np.random.randint(-100, 100, size=(65000, 384), dtype=np.int16))
    transformation = np.random.normal(size=(384, 384))
    test_result = shift_data(data, transformation)
    assert type(test_result == np.ndarray)
    assert test_result.dtype == np.int16
    assert np.isfortran(test_result)
    assert np.allclose(test_result, np.asfortranarray((data @ transformation.T).astype("int16")))

def test_align_block2():

    # Rigid Case
    spike_histograms = np.zeros((20, 2, 3))
    spike_histograms[2:4,:,0] = 1
    spike_histograms[3:5,:,1] = 1
    spike_histograms[4:6,:,2] = 1
    imin, yblk, F0 = align_block2(spike_histograms, np.arange(20), 1)
    imin -= np.mean(imin)
    assert np.allclose(imin, np.array([[1],[0],[-1]]))


def test_apply_drift_transform():

    # Rigid Case
    dat = np.zeros((50,10), order='F')
    dat[:,4] = 10
    shifts_in = np.array([1])
    ysamp = np.array([5])
    probe = Bunch()
    probe.yc = np.arange(10)
    probe.xc = np.zeros(10)
    sig = 1

    data_shifted = apply_drift_transform(dat, shifts_in, ysamp, probe, sig)
    assert np.argmax(np.mean(data_shifted, axis=0)) == 5
