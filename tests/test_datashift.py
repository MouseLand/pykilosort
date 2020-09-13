import os
import pytest
import cupy as cp
import numpy as np

from pykilosort.cluster import getClosestChannels2

from .helpers import get_test_data

TEST_DATA_FOLDER = "tests/fixtures/data"

@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    *get_test_data(TEST_DATA_FOLDER, ["set_1", "set_2"])    
)
def test_get_closest_channels(xc, yc, xcup, ycup, NchanClosest, iC, dist):
    actual_iC, actual_dist = getClosestChannels2(ycup, xcup, yc, xc, NchanClosest)

    assert actual_iC.shape == iC.shape
    assert actual_dist.shape == dist.shape

    assert (cp.asnumpy(actual_iC) == iC - 1).all() # iC is an index
    assert ((cp.asnumpy(actual_dist) - dist) < 0.0001).all()


