import numpy as np

from ..utils import Bunch


""" Some standard probe geometries - please check before using! """

def np1_probe():
    """ Returns a Neuropixels 1 probe as a Bunch object for use in pykilosort """
    probe = Bunch()
    probe.NchanTOT = 385
    probe.chanMap = np.arange(384)
    probe.xc = np.tile(np.array([43., 11., 59., 27.]), 96)
    probe.yc = np.repeat(np.arange(20, 3841, 20.), 2)
    probe.kcoords = np.zeros(384)
    return probe


def np2_probe():
    """ Returns a Neuropixels 2 probe as a Bunch object for use in pykilosort """
    probe = Bunch()
    probe.NchanTOT = 385
    probe.chanMap = np.arange(384)
    probe.xc = np.tile(np.array([0., 32.]), 192)
    probe.yc = np.repeat(np.arange(0, 2866, 15.), 2)
    probe.kcoords = np.zeros(384)
    return probe
