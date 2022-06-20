import typing as t

import numpy as np
from pydantic import BaseModel, Field, validator

from spikeglx import Reader
from ..utils import Bunch


class Probe(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    n_channels_total: int = Field(None, description='Total number of channels including sync channels')

    channel_map: np.ndarray = Field(None, description='Channel Indices')
    xcoords: np.ndarray = Field(None, description='X coordinates of channels')
    ycoords: np.ndarray = Field(None, description='Y coordinates of channels')

    channel_groups: t.Optional[np.ndarray] = Field(None, description='Channel Group Indices')

    sample_shifts: t.Optional[np.ndarray] = Field(None, description='Channel Sample Shifts')

    @property
    def n_channels(self) -> int:
        return len(self.channel_map)

    @validator('xcoords', 'ycoords', 'channel_groups', 'sample_shifts')
    def check_lengths_equal(cls, value, values, field):
        assert len(value) == len(values['channel_map']), \
            f'Length of {field.name} does not match the channel map'
        return value

    def keys(self):
        return self.__fields__.keys()


def neuropixel_probe_from_metafile(file_path):
    """
    Uses IBL's SpikeGLX reader to automatically load the probe from the metafile
    :param file_path: Path to metafile, str or pathlib Path
    :return: Probe object
    """
    reader = Reader(file_path)

    probe = Probe(
        n_channels_total = reader.nc,
        channel_map = reader.geometry['ind'],
        xcoords = reader.geometry['x'],
        ycoords = reader.geometry['y'],
        channel_groups = reader.geometry['shank'].astype('int'),
        sample_shifts = reader.geometry['sample_shift'],
    )

    return probe


""" Some standard probe geometries - please check before using! """

def np1_probe(sync_channel=True):
    """ Returns a Neuropixels 1 probe for use in pykilosort """

    probe_args = {
        'n_channels_total': 385 if sync_channel else 384,
        'channel_map': np.arange(384),
        'xcoords': np.tile(np.array([43., 11., 59., 27.]), 96),
        'ycoords': np.repeat(np.arange(20, 3841, 20.), 2),
        'channel_groups': np.zeros(384),
        'sample_shifts': np.tile(np.repeat(np.arange(12)/12, 2), 16),
    }

    return Probe(**probe_args)


def np2_probe(sync_channel=True):
    """ Returns a Neuropixels 2 probe for use in pykilosort """
    probe_args = {
        'n_channels_total': 385 if sync_channel else 384,
        'channel_map': np.arange(384),
        'xcoords': np.tile(np.array([0., 32.]), 192),
        'ycoords': np.repeat(np.arange(0, 2866, 15.), 2),
        'channel_groups': np.zeros(384),
        'sample_shifts': np.tile(np.repeat(np.arange(16)/16, 2), 12),
    }

    return Probe(**probe_args)


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

    probe_args = {}
    probe_args['n_channels_total'] = 385

    if shank is None:
        # Return whole probe
        probe_args['channel_map'] = np.arange(384)
        probe_args['channel_groups'] = np.zeros(384)
        probe_args['xcoords'] = np.zeros(384)
        probe_args['ycoords'] = np.zeros(384)

        for shank_id in range(4):
            shank_channels = get_4shank_channels_np2(shank_id)
            probe_args['xcoords'][shank_channels] = np.tile([0., 32.], 48) + shank_id * 200
            probe_args['ycoords'][shank_channels] = np.repeat(np.arange(2880, 3586, 15.), 2)

        return Probe(**probe_args)

    if shank in [0, 1]:
        probe_args['channel_map'] = np.concatenate((np.arange(48*shank, 48*(shank+1)),
                                                np.arange(48*(shank+2), 48*(shank+3))))
    if shank in [2, 3]:
        probe_args['channel_map'] = np.concatenate((np.arange(48*(shank+2), 48*(shank+3)),
                                                np.arange(48*(shank+4), 48*(shank+5))))

    probe_args['xcoords'] = np.tile([0., 32.], 48) + shank * 200
    probe_args['ycoords'] = np.repeat(np.arange(2880, 3586, 15.), 2)
    probe_args['channel_groups'] = np.zeros(96)

    return Probe(**probe_args)
