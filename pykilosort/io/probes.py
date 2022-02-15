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


def get_4shank_channels_np2(shank):
    """
    Returns the channel indices for a given shank on a np2 4 shank probe
    :param shank: Shank to return, int between 0 and 3 inclusive
    :return: Numpy array
    """
    assert type(shank) == int, 'Shank index must be an integer'
    assert 0 <= shank <= 3, 'Shank index must be between 0 and 3'

    if shank in [0, 1]:
        return np.concatenate((np.arange(48*shank, 48*(shank+1)),
                                        np.arange(48*(shank+2), 48*(shank+3))))
    if shank in [2, 3]:
        return np.concatenate((np.arange(48*(shank+2), 48*(shank+3)),
                                        np.arange(48*(shank+4), 48*(shank+5))))

def np2_4shank_probe(shank=None):
    """
    Returns a Neuropixels 2 4-shank probe as a Bunch object
    :param shank: Optional, return only a single shank, int between 0 and 3 inclusive
    :return: Bunch object for pykilosort
    """
    if shank is not None:
        assert type(shank) == int, 'Shank index must be an integer'
        assert 0 <= shank <= 3, 'Shank index must be between 0 and 3'

    probe = Bunch()
    probe.NchanTOT = 385

    if shank is None:
        # Return whole probe
        probe.chanMap = np.arange(384)
        probe.kcoords = np.zeros(384)
        probe.xc = np.zeros(384)
        probe.yc = np.zeros(384)

        for shank_id in range(4):
            shank_channels = get_4shank_channels_np2(shank_id)
            probe.xc[shank_channels] = np.tile([0., 32.], 48) + shank_id * 200
            probe.yc[shank_channels] = np.repeat(np.arange(2880, 3586, 15.), 2)

        return probe

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
