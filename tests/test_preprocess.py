from pathlib import Path
import numpy as np
from scipy.signal import lfilter as lfilter_cpu
import cupy as cp

from pykilosort.postprocess import my_conv2, merge_by_order

sig = 250
tmax = 1000

test_path = Path(__file__).parent


def test_convolve_real_data():
    file_s1 = test_path.joinpath('my_conv2_input.npy')
    file_expected = test_path.joinpath('my_conv2_output.npy')
    if not file_s1.exists() or not file_expected.exists():
        return

    s1 = np.load(file_s1)
    s1_expected = np.load(file_expected)

    def diff(out):
        return np.max(np.abs(cp.asnumpy(out[1000:-1000, :1]) - s1_expected[1000:-1000, :1]))

    x = cp.asarray(s1)
    assert diff(my_conv2(x, 250, 0, nwin=0, pad=None)) < 1e-6
    assert diff(my_conv2(x, 250, 0, nwin=0, pad='zeros')) < 1e-6
    assert diff(my_conv2(x, 250, 0, nwin=0, pad='constant')) < 1e-6

    assert diff(my_conv2(x, 250, 0, nwin=10000, pad=None)) < 1e-3
    assert diff(my_conv2(x, 250, 0, nwin=10000, pad='zeros')) < 1e-3
    assert diff(my_conv2(x, 250, 0, nwin=10000, pad='constant')) < 1e-3

    assert diff(my_conv2(x, 250, 0, nwin=0, pad='flip')) < 1e-6
    assert diff(my_conv2(x, 250, 0, nwin=10000, pad='flip')) < 1e-3


def create_test_dataset():
    cp = np  # cpu mode only here
    s1 = np.load(test_path.joinpath('my_conv2_input.npy'))
    s0 = np.copy(s1)
    tmax = np.ceil(4 * sig)
    dt = cp.arange(-tmax, tmax + 1)
    gauss = cp.exp(-dt ** 2 / (2 * sig ** 2))
    gauss = (gauss / cp.sum(gauss)).astype(np.float32)

    cNorm = lfilter_cpu(gauss, 1., np.r_[np.ones(s1.shape[0]), np.zeros(int(tmax))])
    cNorm = cNorm[int(tmax):]

    s1 = lfilter_cpu(gauss, 1, np.r_[s1, np.zeros((int(tmax), s1.shape[1]))], axis=0)
    s1 = s1[int(tmax):] / cNorm[:, np.newaxis]

    np.save(test_path.joinpath('my_conv2_output.npy'), s1)


def test_merge_by_order():

    n_chan = 32
    n_rank = 3
    n1 = 20
    n2 = 30

    total = n1 + n2

    features1 = np.asfortranarray(np.random.randn(n_chan, n_rank, n1))
    features2 = np.asfortranarray(np.random.randn(n_chan, n_rank, n2))

    times1 = np.sort(np.random.choice(total, n1, replace=False))
    mask = np.ones(total, dtype='bool')
    mask[times1] = False
    times2 = np.where(mask)[0]

    output = merge_by_order(features1, features2, times1, times2, axis=2)

    test_output = np.zeros((n_chan, n_rank, total), dtype=features1.dtype, order='F')
    for i in range(total):
        if i in times1:
            location = np.where(times1 == i)[0][0]
            test_output[:,:,i] = features1[:,:,location]
        else:
            location = np.where(times2 == i)[0][0]
            test_output[:,:,i] = features2[:,:,location]

    assert np.allclose(output, test_output)
