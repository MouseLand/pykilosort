import numpy as np

from pykilosort.postprocess import ccg


def slow_ccg(st1, st2, nbins, tbin):
    diffs = np.rint((st1 - st2.reshape(-1,1)).flatten() / tbin).astype('int')
    output = np.histogram(diffs, bins=np.arange(-nbins, nbins+2))[0]
    return output


def test_ccg():

    spike_times = np.cumsum(np.random.randint(100, 30000, 500)) / 30000
    nbins = 500
    tbin = 0.1

    output = ccg(spike_times, spike_times, nbins, tbin)
    output_comparison = slow_ccg(spike_times, spike_times, nbins, tbin)

    assert np.allclose(output[1:-1], output_comparison[1:-1])
