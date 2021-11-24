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


def np2_4shank_probe(shank):
    """
    Returns a Neuropixels 2 4-shank probe as a Bunch object
    :param shank: Shank to return, int between 0 and 3 inclusive
    :return: Bunch object for pykilosort
    """
    assert type(shank) == int, 'Shank index must be an integer'
    assert 0 <= shank <= 3, 'Shank index must be between 0 and 3'

    probe = Bunch()
    probe.NchanTOT = 385

    if shank in [0, 1]:
        probe.chanMap = np.concatenate((np.arange(48*shank, 48*(shank+1)),
                                        np.arange(48*(shank+2), 48*(shank+3))))
    if shank in [2, 3]:
        probe.chanMap = np.concatenate((np.arange(48*(shank+2), 48*(shank+3)),
                                        np.arange(48*(shank+4), 48*(shank+5))))

    probe.xc = np.tile([0., 32.], 48) + shank * 200
    probe.yc = np.repeat(np.arange(2880, 3586, 15.), 2)
    probe.kcoords = np.zeros(96)

    return probe
