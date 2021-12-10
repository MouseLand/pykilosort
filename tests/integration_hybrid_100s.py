import os
from pathlib import Path
import numpy as np

import pykilosort
from deploy.serverpc.kilosort2.run_pykilosort import run_spike_sorting_ibl

# Minimum recall to pass checks (percentage)
MIN_RECALL = 90

# Minimum precision to pass checks (percentage)
MIN_PRECISION = 90

# Maximum allowed difference in time for two spike events to be considered the same (in samples)
TOLERANCE = 30


def run_checks(output_path, cluster_times_path):
    """
    Checks that the spike sorting results in output_path match the groud truth times stored
    in the folder given by cluster_times_path
    :param output_path: Path to spike sorting output, pathlib Path
    :param cluster_times_path: Path to ground truth times, pathlib Path
    """
    output_path = Path(output_path)
    cluster_times_path = Path(cluster_times_path)

    assert os.path.isdir(output_path), 'Unable to find output'
    assert os.path.isdir(cluster_times_path), 'Unable to find hybrid clusters'

    # Load times and template IDs detected by spike sorting
    spike_times = np.load(output_path / 'spike_times.npy')
    spike_templates = np.load(output_path / 'spike_templates.npy')

    # Ensure spike times are sorted
    idx = np.argsort(spike_times)
    spike_times = spike_times[idx]
    spike_templates = spike_templates[idx]

    n_templates = np.max(spike_templates) + 1

    failed_clusters = []
    n_clusters = len(os.listdir(cluster_times_path))

    for file in os.listdir(cluster_times_path):
        hybrid_id = file.split('_')[-1][:-4]
        true_times = np.load(cluster_times_path / file)

        precisions = np.zeros(n_templates)
        recalls = np.zeros(n_templates)
        for i in range(n_templates):
            detected_times = spike_times[spike_templates == i]
            precisions[i], recalls[i] = calculate_metrics(detected_times, true_times, TOLERANCE)

        # Choose best cluster by finding the one with the best recall
        best_id = np.argmax(recalls)
        best_precision = precisions[best_id]
        best_recall = recalls[best_id]

        if (best_precision < MIN_PRECISION) or (best_recall < MIN_RECALL):
            failed_clusters.append(hybrid_id)

    if len(failed_clusters) == 0:
        print(f'All {n_clusters} clusters passed')
    else:
        print(f'These clusters failed: {failed_clusters}')
        raise ValueError("Checks failed")


def calculate_metrics(event_times, true_events, tolerance):
    """
    Returns the precision and recall of a detected cluster in comparison with an hybrid cluster
    with known times
    WARNING: Times for both clusters must be sorted
    :param event_times: Sorted Numpy array, event times of detected cluster
    :param true_events: Sorted Numpy array, event times of known hybrid cluster
    :param tolerance: Allowed time tolerance for two events to be considered the same
    :return: Precision and recall as percentages
    """

    if (len(event_times) == 0) or (len(true_events) == 0):
        return 0, 0

    # Percentage of event_times within given tolerance of an event in true_events
    precision = percentage_hits(event_times, true_events, tolerance)

    # Percentage of true_events within given tolerance of an event in event_times
    recall = percentage_hits(true_events, event_times, tolerance)

    return precision, recall


def percentage_hits(events, targets, tolerance):
    """
    Returns percentage of events that are within the tolerance of a target in the array targets
    WARNING: Array targets must be sorted
    :param events: Numpy array
    :param targets: Numpy array
    :param tolerance: Allowed tolerance
    :return: Float
    """

    positions = np.searchsorted(targets, events)

    # Check if closest target before event is within range
    hits_lower = np.logical_and((events - targets[np.maximum(positions - 1, 0)] <= tolerance),
                                (positions != 0))

    # Check if closest target after event is within range
    hits_upper = np.logical_and((targets[np.minimum(positions, len(targets) - 1)] - events <= tolerance),
                                (positions != len(targets)))

    # Combine the two
    hits = np.logical_or(hits_lower, hits_upper)

    # Return percentage of events with a match
    return np.mean(hits) * 100

# import numpy as np
# from easyqc.gui import viewseis
# datfile = "/datadisk/Data/spike_sorting/pykilosort_tests/.kilosort/imec_385_100s.ap/proc.dat"
# w = np.memmap(datfile, shape=(46 * 65600, 384), dtype=np.int16)
#
# viewseis(w[:120000, :], si=1, taxis=0)


def _make_compressed_parts(bin_file):
    import shutil
    from ibllib.io import spikeglx
    BATCH_WRITE_BYTES = int(1024 * 1024 * 128)
    file_part1 = bin_file.parent.joinpath(bin_file.stem + '.part1.bin')
    file_part2 = bin_file.parent.joinpath(bin_file.stem + '.part2.bin')

    def dump_file(file_in, file_out, part):
        nbytes = Path(bin_file).stat().st_size
        print(nbytes)
        if part == 1:
            offset = int(0)
            max_offset = int(nbytes / 2)
        elif part == 2:
            offset = int(nbytes / 2)
            max_offset = int(nbytes)
        with file_in.open('rb') as fidr, file_out.open('wb+') as fidw:
            while True:
                fidr.seek(offset, 0)
                print(fidr.tell(), offset)
                bytes2read = int(np.minimum(BATCH_WRITE_BYTES, max_offset - offset))
                dat = np.fromfile(fidr, dtype=np.int16, offset=0, count=int(bytes2read / 2))
                # print(dat.shape, bytes2read, fidr.tell(), offset, max_offset)
                dat.tofile(fidw)
                offset += bytes2read
                if offset == max_offset:
                    break
        print(f'done file {part}')
        shutil.copy(file_in.with_suffix('.meta'), file_out.with_suffix('.meta'))
        srout = spikeglx.Reader(file_out)
        srout.compress_file(keep_original=False)
    dump_file(bin_file, file_part1, part=1)
    dump_file(bin_file, file_part2, part=2)
    # read after write checks
    sr0 = spikeglx.Reader(bin_file)
    sr1 = spikeglx.Reader(file_part1.with_suffix('.cbin'))
    sr2 = spikeglx.Reader(file_part2.with_suffix('.cbin'))
    assert np.all(sr0[200000:201000, :] == sr1[200000:201000, :])
    assert np.all(sr0[int(65 * 30000):int(65.2 * 30000), :] == sr2[int(15 * 30000):int(15.2 * 30000), :])

import shutil

INTEGRATION_DATA_PATH = Path("/datadisk/Data/spike_sorting/pykilosort_tests")
SCRATCH_DIR = Path.home().joinpath("scratch", 'pykilosort')
shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
# bin_file = INTEGRATION_DATA_PATH.joinpath("imec_385_100s.ap.bin")

cluster_times_path = INTEGRATION_DATA_PATH.joinpath("cluster_times")

MULTIPARTS = False
if MULTIPARTS:
    bin_file = list(INTEGRATION_DATA_PATH.rglob("hybrid_data_100s.part*.cbin"))
    bin_file.sort()
    # _make_compressed_parts(bin_file)
    ks_output_dir = INTEGRATION_DATA_PATH.joinpath(
        f"{pykilosort.__version__}", bin_file[0].name.split('.')[0] + 'multi_parts')
else:
    bin_file = INTEGRATION_DATA_PATH.joinpath("hybrid_data_100s.bin")
    ks_output_dir = INTEGRATION_DATA_PATH.joinpath(f"{pykilosort.__version__}", bin_file.name.split('.')[0])


ks_output_dir.mkdir(parents=True, exist_ok=True)
alf_path = ks_output_dir.joinpath('alf')

run_spike_sorting_ibl(bin_file, delete=True, scratch_dir=SCRATCH_DIR,
                      ks_output_dir=ks_output_dir, alf_path=alf_path, log_level='DEBUG')


run_checks(ks_output_dir, cluster_times_path)
