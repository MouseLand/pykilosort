from pathlib import Path
import datetime
import json
import logging
import shutil

import numpy as np

from ibllib.io import spikeglx
from ibllib.ephys import spikes, neuropixel
from one.alf.files import get_session_path
from pykilosort import add_default_handler, run, Bunch, __version__
from pykilosort.io.params import KilosortParams


_logger = logging.getLogger("pykilosort")


def _get_multi_parts_records(bin_file):
    """ Looks for the multiple parts of the recording using sequence files from ibllib"""
    # if multiple files are already provided, do not look for sequence files
    if isinstance(bin_file, list) or isinstance(bin_file, tuple):
        for bf in bin_file:
            if not Path(bf).exists():
                raise FileNotFoundError(bf)
        return bin_file
    # if there is no sequence file attached to the binary file, return just the bin file
    bin_file = Path(bin_file)
    sequence_file = bin_file.parent.joinpath(f"{bin_file.stem.replace('.ap', '.sequence.json')}")
    if not sequence_file.exists():
        if not Path(bin_file).exists():
            raise FileNotFoundError(bin_file)
        else:
            return bin_file
    # if there is a sequence file, return all files if they're all present and this is the first index
    with sequence_file.open() as fid:
        seq = json.load(fid)
    if seq['index'] > 0:
        _logger.warning(f"Multi-part raw ephys: returns empty as this is not the first "
                        f"index in the sequence. Check: {sequence_file}")
        return
    # the common anchor path to look for other meta files is the subject path
    subject_folder = get_session_path(bin_file)
    subject_folder_seq = get_session_path(Path(seq['files'][0])).parents[1]
    # reconstruct path of each binary file, exit with None if one is not found
    cbin_files = []
    for f in seq['files']:
        meta_file = subject_folder.joinpath(Path(f).relative_to(subject_folder_seq))
        cbin_file = next(meta_file.parent.glob(meta_file.stem + '.*bin'), None)
        if cbin_file is None:
            _logger.error(f"Multi-part raw ephys error: missing bin file in folder {meta_file.parent}")
            return
        cbin_files.append(cbin_file)
    return cbin_files


def _sample2v(ap_file):
    md = spikeglx.read_meta_data(ap_file.with_suffix(".meta"))
    s2v = spikeglx._conversion_sample2v_from_meta(md)
    return s2v["ap"][0]


def run_spike_sorting_ibl(bin_file, scratch_dir=None, delete=True,
                          ks_output_dir=None, alf_path=None, log_level='INFO', params=None):
    """
    This runs the spike sorting and outputs the raw pykilosort without ALF conversion
    :param bin_file: binary file full path to
    :param scratch_dir: working directory (home of the .kilosort folder) SSD drive preferred.
    :param delete: bool, optional, defaults to True: whether or not to delete the .kilosort temp folder
    :param ks_output_dir: string or Path: output directory defaults to None, in which case it will output in the
     scratch directory.
    :param alf_path: strint or Path, optional: if specified, performs ks to ALF conversion in the specified folder
    :param log_level: string, optional, defaults to 'INFO'
    :return:
    """
    START_TIME = datetime.datetime.now()
    # handles all the paths infrastructure
    assert scratch_dir is not None
    bin_file = _get_multi_parts_records(bin_file)
    scratch_dir.mkdir(exist_ok=True, parents=True)
    ks_output_dir = Path(ks_output_dir) if ks_output_dir is not None else scratch_dir.joinpath('output')
    log_file = scratch_dir.joinpath(f"_{START_TIME.isoformat()}_kilosort.log")
    add_default_handler(level=log_level)
    add_default_handler(level=log_level, filename=log_file)
    # construct the probe geometry information
    if params is None:
        params = ibl_pykilosort_params(bin_file[0]) if isinstance(bin_file, list) else ibl_pykilosort_params(bin_file)
    try:
        _logger.info(f"Starting Pykilosort version {__version__}, output in {bin_file.parent}")
        run(bin_file, dir_path=scratch_dir, output_dir=ks_output_dir, **params)
        if delete:
            shutil.rmtree(scratch_dir.joinpath(".kilosort"), ignore_errors=True)
    except Exception as e:
        _logger.exception("Error in the main loop")
        raise e
    [_logger.removeHandler(h) for h in _logger.handlers]
    shutil.move(log_file, ks_output_dir.joinpath('spike_sorting_pykilosort.log'))

    # convert the pykilosort output to ALF IBL format
    if alf_path is not None:
        s2v = _sample2v(bin_file)
        alf_path.mkdir(exist_ok=True, parents=True)
        spikes.ks2_to_alf(ks_output_dir, bin_file, alf_path, ampfactor=s2v)


def ibl_pykilosort_params(bin_file):

    params = KilosortParams()
    params.preprocessing_function = 'destriping'
    params.probe = probe_geometry(bin_file)
    # params = {k: dict(params)[k] for k in sorted(dict(params))}
    return dict(params)


def probe_geometry(bin_file):
    """
    Loads the geometry from the meta-data file of the spikeglx acquisition system
    sr: ibllib.io.spikeglx.Reader or integer with neuropixel version 1 or 2
    """
    if isinstance(bin_file, list):
        sr = spikeglx.Reader(bin_file[0])
        h = sr.geometry
        ver = sr
    else:
        assert(bin_file == 1 or bin_file == 2)
        h = neuropixel.trace_header(version=bin_file)
        ver = bin_file
    nc = h['x'].size
    probe = Bunch()
    probe.NchanTOT = nc + 1
    probe.chanMap = np.arange(nc)
    probe.xc = h['x']
    probe.yc = h['y']
    probe.x = h['x']
    probe.y = h['y']
    probe.kcoords = np.zeros(nc)
    probe.neuropixel_version = ver
    probe.sample_shift = h['sample_shift']
    return probe
