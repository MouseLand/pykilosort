import os
from contextlib import contextmanager
from functools import reduce
import json
import logging
from math import ceil
from pathlib import Path
import operator
import os.path as op
import re
from time import perf_counter, strftime

from tqdm import tqdm
import numpy as np
from numpy.lib.format import (
    _check_version, _write_array_header, header_data_from_array_1_0, dtype_to_descr)
import cupy as cp

from .event import emit, connect, unconnect  # noqa

logger = logging.getLogger(__name__)


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class Bunch(dict):
    """A subclass of dictionary with an additional dot syntax."""
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        """Return a new Bunch instance which is a copy of the current Bunch instance."""
        return Bunch(super(Bunch, self).copy())


def copy_bunch(old_bunch):
    """
    A function to copy a bunch object with a deep copy of numpy arrays
    :param old_bunch: Bunch object to be copied
    :return: New Bunch object
    """
    assert isinstance(old_bunch, Bunch)

    new_bunch = Bunch()
    for key in old_bunch.keys():
        if type(old_bunch[key]) == np.ndarray:
            new_bunch[key] = old_bunch[key].copy()
        else:
            new_bunch[key] = old_bunch[key]

    return new_bunch


def p(x):
    print("shape", x.shape, "mean", "%5e" % x.mean())
    print(x[:2, :2])
    print()
    print(x[-2:, -2:])


def _extend(x, i0, i1, val, axis=0):
    """Extend an array along a dimension and fill it with some values."""
    shape = x.shape
    if x.shape[axis] < i1:
        s = list(x.shape)
        s[axis] = i1 - s[axis]
        x = cp.concatenate((x, cp.zeros(tuple(s), dtype=x.dtype, order='F')), axis=axis)
        assert x.shape[axis] == i1
    s = [slice(None, None, None)] * x.ndim
    s[axis] = slice(i0, i1, 1)
    x[tuple(s)] = val
    for i in range(x.ndim):
        if i != axis:
            assert x.shape[i] == shape[i]
    return x


def is_fortran(x):
    if isinstance(x, np.ndarray):
        return x.flags.f_contiguous
    raise ValueError('`is_fortran` is only implemented for numpy arrays')


def _make_fortran(x):
    if isinstance(x, cp.ndarray):
        x = cp.asnumpy(x)
    return np.asfortranarray(x)


# TODO: design - let's move this to an io module
def read_data(dat_path, offset=0, shape=None, dtype=None, axis=0):
    count = shape[0] * shape[1] if shape and -1 not in shape else -1
    buff = np.fromfile(dat_path, dtype=dtype, count=count, offset=offset)
    if shape and -1 not in shape:
        shape = (-1, shape[1]) if axis == 0 else (shape[0], -1)
    if shape:
        buff = buff.reshape(shape, order='F')
    return buff


# TODO: design - let's move this to an io module
def memmap_binary_file(dat_path, n_channels=None, shape=None, dtype=None, offset=None):
    """Memmap a dat file in FORTRAN order, shape (n_channels, n_samples)."""
    assert dtype is not None
    item_size = np.dtype(dtype).itemsize
    offset = offset if offset else 0
    if shape is None:
        assert n_channels is not None
        n_samples = (op.getsize(str(dat_path)) - offset) // (item_size * n_channels)
        shape = (n_channels, n_samples)
    assert shape
    shape = tuple(shape)
    return np.memmap(str(dat_path), dtype=dtype, shape=shape, offset=offset, order='F')


# TODO: design - move this to cuda/cupy module.
def extract_constants_from_cuda(code):
    r = re.compile(r'const int\s+\S+\s+=\s+\S+.+')
    m = r.search(code)
    if m:
        constants = m.group(0).replace('const int', '').replace(';', '').split(',')
        for const in constants:
            a, b = const.strip().split('=')
            yield a.strip(), int(b.strip())


# TODO: design - move this to cuda/cupy module.
def get_cuda(fn):
    path = Path(__file__).parent / 'cuda' / (fn + '.cu')
    assert path.exists
    code = path.read_text()
    code = code.replace('__global__ void', 'extern "C" __global__ void')
    return code, Bunch(extract_constants_from_cuda(code))


# TODO: design - let's move this to an io module
class LargeArrayWriter(object):
    """Save a large array chunk by chunk, in a binary file with FORTRAN order."""
    def __init__(self, path, dtype=None, shape=None):
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self._shape = shape
        assert shape[-1] == -1  # the last axis must be the extendable axis, in FORTRAN order
        assert -1 not in shape[:-1]  # shape may not contain -1 outside the last dimension
        self.fw = open(self.path, 'wb')
        self.extendable_axis_size = 0
        self.total_size = 0

    def append(self, arr):
        # We convert to the requested data type.
        assert arr.flags.f_contiguous  # only FORTRAN order arrays are currently supported
        assert arr.shape[:-1] == self._shape[:-1]
        arr = arr.astype(self.dtype)
        es = arr.shape[-1]
        if arr.flags.f_contiguous:
            arr = arr.T
        # We download the array from the GPU if required.
        # We ensure the array is in FORTRAN order now.
        assert arr.flags.c_contiguous
        if isinstance(arr, cp.ndarray):
            arr = cp.asnumpy(arr)
        arr.tofile(self.fw)
        self.total_size += arr.size
        self.extendable_axis_size += es  # the last dimension, but
        assert prod(self.shape) == self.total_size

    @property
    def shape(self):
        return self._shape[:-1] + (self.extendable_axis_size,)

    def close(self):
        self.fw.close()
        # Save JSON metadata file.
        with open(self.path.with_suffix('.json'), 'w') as f:
            json.dump({'shape': self.shape, 'dtype': str(self.dtype), 'order': 'F'}, f)


# TODO: design - let's move this to an io module
def memmap_large_array(path):
    """Memmap a large array saved by LargeArrayWriter."""
    path = Path(path)
    with open(path.with_suffix('.json'), 'r') as f:
        metadata = json.load(f)
    assert metadata['order'] == 'F'
    dtype = np.dtype(metadata['dtype'])
    shape = metadata['shape']
    return memmap_binary_file(path, shape=shape, dtype=dtype)


# TODO: design - let's move this to an io module
def _npy_header(shape, dtype, order='C'):
    d = {'shape': shape}
    if order == 'C':
        d['fortran_order'] = False
    elif order == 'F':
        d['fortran_order'] = True
    else:
        # Totally non-contiguous data. We will have to make it C-contiguous
        # before writing. Note that we need to test for C_CONTIGUOUS first
        # because a 1-D array is both C_CONTIGUOUS and F_CONTIGUOUS.
        d['fortran_order'] = False

    d['descr'] = dtype_to_descr(dtype)
    return d


# TODO: design - let's move this to an io module
def save_large_array(fp, array, axis=0, desc=None):
    """Save a large, potentially memmapped array, into a NPY file, chunk by chunk to avoid loading
    it entirely in memory."""
    assert axis == 0  # TODO: support other axes
    version = None
    _check_version(version)
    _write_array_header(fp, header_data_from_array_1_0(array), version)
    N = array.shape[axis]
    if N == 0:
        return

    k = int(ceil(float(N) / 100))  # 100 chunks
    assert k >= 1
    for i in tqdm(range(0, N, k), desc=desc):
        chunk = array[i:i + k, ...]
        fp.write(chunk.tobytes())


# TODO: design - let's move this to an io module
class NpyWriter(object):
    def __init__(self, path, shape, dtype, axis=0):
        assert axis == 0  # only concatenation along the first axis is supported right now
        # Only C order is supported at the moment.
        self.shape = shape
        self.dtype = np.dtype(dtype)
        header = _npy_header(self.shape, self.dtype)
        version = None
        _check_version(version)
        self.fp = open(path, 'wb')
        _write_array_header(self.fp, header, version)

    def append(self, chunk):
        if chunk.ndim == len(self.shape):
            assert chunk.shape[1:] == self.shape[1:]
        else:
            assert chunk.shape == self.shape[1:]
        self.fp.write(cp.asnumpy(chunk).tobytes())

    def close(self):
        self.fp.close()


# TODO: design - this is a nice pythonic-ish mirror of the MATLAB global context,
#              - it might be nicer if it didn't inherit from Bunch (we can be more strict
#              - about what attributes are guaranteed to be there etc.).

# TODO: needs_test - pretty key class and need to make sure we can use it safely
class Context(Bunch):
    def __init__(self, context_path):
        super(Context, self).__init__()
        self.context_path = context_path
        self.intermediate = Bunch()
        self.context_path.mkdir(exist_ok=True, parents=True)
        self.timer = {}

    @property
    def metadata_path(self):
        return self.context_path / 'metadata.json'

    def path(self, name, ext='.npy'):
        """Path to an array in the context directory."""
        return self.context_path / (name + ext)

    def read_metadata(self):
        """Read the metadata dictionary from the metadata.json file in the context dir."""
        if not self.metadata_path.exists():
            return Bunch()
        with open(self.metadata_path, 'r') as f:
            return Bunch(json.load(f))

    def write_metadata(self, metadata):
        """Write metadata dictionary in the metadata.json file."""
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def read(self, name):
        """Read an array from memory (intermediate object) or from disk."""
        if name not in self.intermediate:
            path = self.path(name)
            # Load a NumPy file.
            if path.exists():
                logger.debug("Loading %s.npy", name)
                # Memmap for large files.
                mmap_mode = 'r' if op.getsize(path) > 1e8 else None
                self.intermediate[name] = np.load(path, mmap_mode=mmap_mode)
            else:
                # Load a value from the metadata file.
                self.intermediate[name] = self.read_metadata().get(name, None)
        return self.intermediate[name]

    def write(self, **kwargs):
        """Write several arrays."""
        # Load the metadata.
        if self.metadata_path.exists():
            metadata = self.read_metadata()
        else:
            metadata = Bunch()
        # Write all variables.
        for k, v in kwargs.items():
            # Transfer GPU arrays to the CPU before saving them.
            if isinstance(v, cp.ndarray):
                logger.debug("Loading %s from GPU.", k)
                v = cp.asnumpy(v)
            if isinstance(v, np.ndarray):
                p = self.path(k)
                overwrite = ' (overwrite)' if p.exists() else ''
                logger.debug("Saving %s.npy%s", k, overwrite)
                np.save(p, np.asfortranarray(v))
            elif v is not None:
                logger.debug("Save %s in the metadata.json file.", k)
                metadata[k] = v
        # Write the metadata file.
        self.write_metadata(metadata)

    def load(self):
        """Load intermediate results from disk."""
        # Load metadata values that are not already loaded in the intermediate dictionary.
        self.intermediate.update(
            {k: v for k, v in self.read_metadata().items() if k not in self.intermediate})
        # Load NumPy arrays that are not already loaded in the intermediate dictionary.
        names = [f.stem for f in self.context_path.glob('*.npy')]
        self.intermediate.update(
            {name: self.read(name) for name in names if name not in self.intermediate})

    def reset(self):
        """Reset context to a clean state."""
        for file in os.listdir(self.context_path):
            if file.split(".")[-1] == "log":
                log_file = Path(self.context_path / file)
                log_file.rename(self.context_path / f"{log_file.stem}_backup_{strftime('%H%M_%d%m%Y')}.log")
            else:
                os.remove((self.context_path / file).as_posix())

        self.intermediate = Bunch()
        self.timer = {}
        self.load()

    def save(self, **kwargs):
        """Save intermediate results to the ctx.intermediate dictionary, and to disk also.

        This has two effects:
        1. variables are available via ctx.intermediate in the current session
        2. In a future session with ctx.load(), these variables will be readily available in
           ctx.intermediate

        """
        for k, v in kwargs.items():
            if v is not None:
                self.intermediate[k] = v
        kwargs = kwargs or self.intermediate
        self.write(**kwargs)

    @contextmanager
    def time(self, name):
        """Context manager to measure the time of a section of code."""
        logger.info("Starting step %s.", name)
        t0 = perf_counter()
        yield
        t1 = perf_counter()
        self.timer[name] = t1 - t0
        self.show_timer(name)

    def show_timer(self, name=None):
        """Display the results of the timer."""
        if name:
            logger.info("Step `{:s}` took {:.2f}s.".format(name, self.timer[name]))
            return
        for name in self.timer.keys():
            self.show_timer(name)


# TODO: design - let's move this to an io module
def load_probe(probe_path):
    """Load a .mat probe file from Kilosort2, or a PRB file (experimental)."""

    # A bunch with the following attributes:
    _required_keys = ('NchanTOT', 'chanMap', 'xc', 'yc', 'kcoords')
    probe = Bunch()
    probe.NchanTOT = 0
    probe_path = Path(probe_path).resolve()

    if probe_path.suffix == '.prb':
        # Support for PRB files.
        contents = probe_path.read_text()
        metadata = {}
        exec(contents, {}, metadata)
        probe.chanMap = []
        probe.xc = []
        probe.yc = []
        probe.kcoords = []
        for cg in sorted(metadata['channel_groups']):
            d = metadata['channel_groups'][cg]
            ch = d['channels']
            pos = d.get('geometry', {})
            probe.chanMap.append(ch)
            probe.NchanTOT += len(ch)
            probe.xc.append([pos[c][0] for c in ch])
            probe.yc.append([pos[c][1] for c in ch])
            probe.kcoords.append([cg for c in ch])
        probe.chanMap = np.concatenate(probe.chanMap).ravel().astype(np.int32)
        probe.chanMapBackup = probe.chanMap.copy()
        probe.xc = np.concatenate(probe.xc)
        probe.yc = np.concatenate(probe.yc)
        probe.kcoords = np.concatenate(probe.kcoords)

    elif probe_path.suffix == '.mat':
        from scipy.io import loadmat
        mat = loadmat(probe_path)
        probe.xc = mat['xcoords'].ravel().astype(np.float64)
        nc = len(probe.xc)
        probe.yc = mat['ycoords'].ravel().astype(np.float64)
        probe.kcoords = mat.get('kcoords', np.zeros(nc)).ravel().astype(np.float64)
        probe.chanMap = (mat['chanMap'] - 1).ravel().astype(np.int32)  # NOTE: 0-indexing in Python
        probe.chanMapBackup = probe.chanMap.copy()
        probe.NchanTOT = len(probe.chanMap)  # NOTE: should match the # of columns in the raw data

    for n in _required_keys:
        assert n in probe.keys()

    return probe


def create_prb(probe):
    chan_map = np.array(probe.chanMap)
    xc, yc = np.array(probe.xc), np.array(probe.yc)
    try:
        bad_channels = np.array(probe.bad_channels)
    except AttributeError:
        bad_channels = np.array([])
    probe_prb = {}
    unique_channel_groups = np.unique(np.array(probe.kcoords))

    for channel_group in unique_channel_groups:
        probe_prb[channel_group] = {}

        channel_group_pos = np.where(probe.kcoords == channel_group)
        group_channels = chan_map[channel_group_pos]
        group_xc = xc[channel_group_pos]
        group_yc = yc[channel_group_pos]

        probe_prb[channel_group]['channels'] = np.setdiff1d(group_channels, bad_channels).tolist()
        geometry = {}

        for c, channel in enumerate(group_channels):
            geometry[channel] = (group_xc[c], group_yc[c])

        probe_prb[channel_group]['geometry'] = geometry
        probe_prb[channel_group]['graph'] = []

    return probe_prb


def extend_probe(
        probe_layout: Bunch
) -> Bunch:
    """
    Extend  probe layout to account for extra num_channels.

    The probe layout selected by the user may have a different
    number of channels than the requested number of channels in
    the dataset. The function attempts to smartly extend the
    layout of the probe to match the requested number of
    channels.

    In case the requested number of channels is less than the
    total channels on the layout, the original probe layout
    is returned.

    Parameters
    ----------
    probe_layout : Bunch
        Input probe layout which might have to be extended.

    Returns
    -------
    probe_layout : Bunch
        Possibly extended probe layout.

    """
    if len(probe_layout["xc"]) >= probe_layout["Nchan"]:
        # if the requested number of channels is less than the number of
        # channels on the probe, return original probe
        return probe_layout
    else:
        n_channels = probe_layout["Nchan"]
        xc = probe_layout["xc"]
        yc = probe_layout["yc"]

        # the assumption here is that the probe layout has a repetitive pattern
        unique_x = np.sort(np.unique(probe_layout["xc"]))  # unique values for the x-axis
        unique_y = np.sort(np.unique(probe_layout["yc"]))  # unique values for the y-axis

        kcoords = probe_layout["kcoords"]
        chan_map = probe_layout["chanMap"]

        new_channels = n_channels - len(xc)  # number of new channels to be added

        # create new properties
        # this will probably break if new_channels > len(unique_x/y)
        append_x = unique_x[-new_channels]
        append_y = unique_y[-new_channels]

        append_chan_map = np.array([
            i for i in np.arange(chan_map[-1],
                                 chan_map[-1] + new_channels)
        ]) + 1  # to account for 0-ordering
        append_kcoords = np.zeros(new_channels)

        # append new properties to existing properties
        new_xc = np.append(xc, append_x)
        new_yc = np.append(yc, append_y)
        new_chan_map = np.append(chan_map, append_chan_map)
        new_kcoords = np.append(kcoords, append_kcoords)

        # save new properties into probe layout
        probe_layout["xc"] = new_xc
        probe_layout["yc"] = new_yc
        probe_layout["chanMap"] = new_chan_map
        probe_layout["chanMapBackup"] = new_chan_map.copy()
        probe_layout["kcoords"] = new_kcoords
        probe_layout["NchanTOT"] = len(new_chan_map)

        return probe_layout


def plot_dissimilarity_matrices(ccb, ccbsort, plot_widget):
    ccb = cp.asnumpy(ccb)
    ccbsort = cp.asnumpy(ccbsort)

    plot_widget.add_image(
        array=ccb.T,
        plot_pos=0,
        labels={"left": "batches",
                "bottom": "batches",
                "title": "batch to batch distance"
                },
        cmap_style="dissimilarity",
        levels=(0.5, 0.9),
    )

    plot_widget.add_image(
        array=ccbsort.T,
        plot_pos=1,
        labels={"left": "sorted batches",
                "bottom": "sorted batches",
                "title": "AFTER sorting"
                },
        cmap_style="dissimilarity",
        levels=(0.5, 0.9),
    )
    plot_widget.show()


def plot_diagnostics(temporal_comp, spatial_comp, mu, nsp, plot_widget):
    temporal_comp = cp.asnumpy(temporal_comp)
    spatial_comp = cp.asnumpy(spatial_comp)
    mu = cp.asnumpy(mu)
    nsp = cp.asnumpy(nsp)

    plot_widget.add_image(
        array=temporal_comp[:, :, 0].T,
        plot_pos=0,
        labels={"left": "Time (samples)",
                "bottom": "Unit Number",
                "title": "Temporal Components"},
        cmap_style="diagnostic",
        levels=(-0.4, 0.4),
        normalize=False,
    )

    plot_widget.add_image(
        array=spatial_comp[:, :, 0].T,
        plot_pos=1,
        labels={"left": "Channel Number",
                "bottom": "Unit Number",
                "title": "Spatial Components"},
        cmap_style="diagnostic",
        levels=(-0.2, 0.2),
        normalize=False,
    )

    plot_widget.add_curve(
        x_data=np.arange(len(mu)),
        y_data=mu,
        plot_pos=2,
        labels={"left": "Amplitude (arb. units)",
                "bottom": "Unit Number",
                "title": "Unit Amplitudes"},
        y_lim=(0, 100),
    )

    plot_widget.add_scatter(
        x_data=np.log(1+nsp),
        y_data=mu,
        plot_pos=3,
        labels={"left": "Amplitude (arb. units)",
                "bottom": "Spike Count",
                "title": "Amplitude vs. Spike Count"},
        y_lim=(0, 1e2),
        x_lim=(0, np.log(1e5)),
        semi_log_x=True,
        pxMode=True,
        symbol="o",
        size=2,
        pen="w",
    )
    plot_widget.show()
