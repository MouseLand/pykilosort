import math

from pykilosort import cupy as cu
import numpy as np

from pykilosort import cluster, utils


def test_get_closest_channels():
    probe = utils.Bunch(
            {
                "NchanTOT": 4,
                "chanMap": np.array([0, 1, 2, 3]),
                "kcoords": np.array([1.0, 1.0, 1.0, 1.0]),
                "xc": np.array([0.0, 0.0, 1.0, 1.0]),
                "yc": np.array([0.0, 1.0, 0.0, 1.0]),
                "Nchan": 4,
                }
            )
    sigma = 30
    NchanClosest = 4

    iC, mask, C2C = cluster.getClosestChannels(probe, sigma, NchanClosest)

    assert cu.all(iC == cu.array([
        [0, 1, 2, 3],
        [1, 0, 0, 1],
        [2, 3, 3, 2],
        [3, 2, 1, 0]
        ]))

    assert cu.allclose([0.5], mask, atol=0.05)

    assert cu.allclose(math.sqrt(2), cu.fliplr(C2C).diagonal())


def test_sort_batches2():
    ccb0 = cu.random.rand(10,10)
    ccb1, isort = cluster.sortBatches2(ccb0)

    assert cu.all(ccb1 == ccb0[isort, :][:, isort])
