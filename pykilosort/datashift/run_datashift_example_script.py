# %load_ext autoreload
# %autoreload 2

# +
import enum
import logging
import math
import os
import shutil
from pathlib import Path
from pprint import pprint


import matplotlib.pyplot as plt
import numpy as np
import pykilosort as ks
from phylib.io.traces import get_ephys_reader
from pydantic import BaseModel
from pykilosort import datashift, params
from pykilosort.cluster import clusterSingleBatches
from pykilosort.datashift.align_block import align_block2
from pykilosort.learn import extractTemplatesfromSnippets, learnAndSolve8b
from pykilosort.params import KilosortParams, Probe
from pykilosort.preprocess import (
    get_good_channels,
    get_Nbatch,
    get_whitening_matrix,
    preprocess,
)
from pykilosort.postprocess import find_merges, rezToPhy, set_cutoff, splitAllClusters
from pykilosort.utils import Bunch, Context, load_probe, memmap_large_array

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# +
# Config

rootZ = "/home/alexmorley/git_repos/Kilosort2/datashift"
rootH = "/tmp/"
pathToYourConfigFile = "/home/alexmorley/git_repos/Kilosort2/configFiles"
chanMapFile = "NP2_kilosortChanMap.mat"

opts = {
    "chanMap": f"{rootZ}/{chanMapFile}",
    "trange": [0.0, 20000.0 * 20],
    "NchanTOT": float(384),
    "minfr_goodchannels": 0.0,
    "sig": 20,
    "fshigh": 300,
    "trackfinal": 0,
    "datashift": {
        "nblocks": 5,
        "sig": 20,
        "output_filename": "/tmp/datashifted.bin",
        "overwrite": False,
    },
    "fbinary": f"{rootZ}/data_cropped.bin",
    "fs": 30000.0,
    "nPCs": 6.0,
}

probe = ks.params.Probe.load_from_npy(
    rootZ=rootZ, NchanTOT=int(opts["NchanTOT"]), kcoords=np.ones(int(opts["NchanTOT"]))
)

# +
# Run Function Args

dat_path = opts["fbinary"]
dir_path = Path(rootZ)
output_dir = Path(rootZ + "/py")
dtype = np.int16
n_channels = int(opts["NchanTOT"])
sample_rate = opts["fs"]
clear_context = False  # Operations.pykilosort_sorting in FORCE_RUN
# -

### Setup
params = ks.params.KilosortParams(**opts, probe=probe, genericSpkTh=10.0)
raw_data = get_ephys_reader(
    get_ephys_reader(
        dat_path, dtype=dtype, sample_rate=sample_rate, n_channels=params.probe.NchanTOT
    )[: 20000 * 20, :],
    sample_rate=sample_rate,
)

dir_path = dir_path or Path(dat_path).parent
n_samples, n_channels = raw_data.shape
logger.info("Loaded raw data with %d channels, %d samples.", n_channels, n_samples)

### Create the context.
ctx_path = dir_path / ".kilosort" / raw_data.name
if clear_context:
    logger.info(f"Clearing context at {ctx_path} ...")
    shutil.rmtree(ctx_path, ignore_errors=True)
ctx = Context(ctx_path)
ctx.params = params
ctx.probe = params.probe
ctx.raw_data = raw_data
ctx.load()
ir = ctx.intermediate
ir.Nbatch = Nbatch = get_Nbatch(raw_data, params)
params.probe.Nchan = params.probe.NchanTOT
params.Nfilt = params.nfilt_factor * params.probe.Nchan
NrankPC = params.nPCs

# +
### Preprocess
# -

if not "proc" in ir:
    ir.Wrot = get_whitening_matrix(raw_data=raw_data, probe=params.probe, params=params)
    ctx.write(Wrot=ir.Wrot)

    ir.proc_path = ctx.path("proc", ".dat")
    preprocess(ctx)

    ir.proc_path
    ir.proc = np.memmap(ir.proc_path, dtype=raw_data.dtype, mode="r", order="F")


# +
ir.xc, ir.yc = params.probe.xc, params.probe.yc

# The min and max of the y and x ranges of the channels
ymin = min(ir.yc)
ymax = max(ir.yc)
xmin = min(ir.xc)
xmax = max(ir.xc)

# Determine the average vertical spacing between channels.
# Usually all the vertical spacings are the same, i.e. on Neuropixels probes.
dmin = np.median(np.diff(np.unique(ir.yc)))
print(f"pitch is {dmin} um\n")
yup = np.arange(
    start=ymin, step=dmin / 2, stop=ymax + (dmin / 2)
)  # centers of the upsampled y positions

# Determine the template spacings along the x dimension
x_range = xmax - xmin
npt = math.floor(
    x_range / 16
)  # this would come out as 16um for Neuropixels probes, which aligns with the geometry.
xup = np.linspace(xmin, xmax, npt + 1)  # centers of the upsampled x positions


# determine prototypical timecourses by clustering of simple threshold crossings.
wTEMP, wPCA = extractTemplatesfromSnippets(
    proc=ir.proc, probe=params.probe, params=params, Nbatch=Nbatch
)
# -

# Extract all the spikes across the recording that are captured by the
# generic templates. Very few real spikes are missed in this way.
st3 = datashift.standalone_detector(
    wTEMP, wPCA, NrankPC, yup, xup, Nbatch, ir.proc, params.probe, params
)


# +
# binning width across Y (um)
dd = 5

# detected depths
dep = st3[:, 1]

# min and max for the range of depths
dmin = ymin
dep = dep - dmin

dmax = int(1 + np.ceil(max(dep) / dd))
Nbatches = Nbatch

# which batch each spike is coming from
batch_id = st3[:, 4]  # ceil[st3[:,1]/dt]

# preallocate matrix of counts with 20 bins, spaced logarithmically
F = np.zeros((dmax, 20, Nbatches))
for t in range(Nbatches):
    # find spikes in this batch
    ix = np.where(batch_id == t)[0]

    # subtract offset
    dep = st3[ix, 1] - dmin

    # amplitude bin relative to the minimum possible value
    amp = np.log10(np.clip(st3[ix, 2], None, 99)) - np.log10(params.genericSpkTh)
    # normalization by maximum possible value
    amp = amp / (np.log10(100) - np.log10(params.genericSpkTh))

    # multiply by 20 to distribute a [0,1] variable into 20 bins
    # sparse is very useful here to do this binning quickly
    i, j, v, m, n = (
        np.ceil(dep / dd).astype("int"),
        np.ceil(1e-5 + amp * 20).astype("int"),
        np.ones((len(ix), 1)),
        dmax,
        20,
    )
    M = np.zeros((m, n))
    M[i - 1, j - 1] += 1

    # the counts themselves are taken on a logarithmic scale (some neurons
    # fire too much!)
    F[:, :, t] = np.log2(1 + M)

ysamp = dmin + dd * np.arange(1, dmax) - dd / 2
imin, yblk, F0 = align_block2(F, ysamp, params.nblocks)

##
st3_orig = st3

st3 = st3[: np.where(st3[:, 1])[0][-1] + 1, :]
if opts.get("fig", True):
    ax = plt.subplot()
    # plot the shift trace in um
    ax.plot(imin * dd)
    ax = plt.subplot()
    # raster plot of all spikes at their original depths
    st_shift = st3[:, 2]  # + imin(batch_id)' * dd
    for j in range(int(params.genericSpkTh), 100):
        # for each amplitude bin, plot all the spikes of that size in the
        # same shade of gray
        ix = st3[:, 3] == j  # the amplitudes are rounded to integers
        ax.plot(
            st3[ix, 1],
            st_shift[ix],
            marker=".",
            color=[max(0, 1 - j / 40) for _ in range(3)],
        )  # the marker color here has been carefully tuned
    plt.tight_layout()
    plt.show()

# convert to um
dshift = imin * dd

# sort in case we still want to do "tracking"
_, ir.iorig = np.sort(np.mean(dshift, axis=1))

for ibatch in range(Nbatches):
    # register the data batch by batch
    shift_batch_on_disk2(
        ibatch,
        dshift[ibatch, :],
        yblk,
        sig,
        Nbatches,
        params,
        ir.proc,
        shifted_fname=params.datashift.output_filename,
        overwrite=params.datashift.overwrite,
    )
logger.info(f"Shifted up/down {Nbatches} batches")

# keep track of dshift
ir.dshift = dshift
# keep track of original spikes
ir.st0 = st3
