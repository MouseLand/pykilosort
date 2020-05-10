from spikeforest2_utils import AutoRecordingExtractor, AutoSortingExtractor

import kachery as ka
import numpy as np

ka.set_config(fr='default_readonly')

recording_path = 'sha1dir://c0879a26f92e4c876cd608ca79192a84d4382868.manual_franklab/tetrode_600s/sorter1_1'

recording = AutoRecordingExtractor(recording_path, download=True)

recording.write_to_binary_dat_format('test/test.bin')

np.save('test/xc.npy', recording.get_channel_locations()[:,0])
np.save('test/yc.npy', recording.get_channel_locations()[:,1].T)
n_channels = len(recording.get_channel_ids())
np.save('test/chanMap.npy', np.array(range(0, n_channels))) 
np.save('test/kcoords.npy', np.ones(n_channels))
