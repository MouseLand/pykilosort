# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import enum
import os
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

## Configuration

PYKILOSORT_DIR = os.environ.get('PYKILOSORT_DIR', None)
if PYKILOSORT_DIR is None:
    import pykilosort
    PYKILOSORT_DIR = os.path.dirname(os.path.dirname(pykilosort.__file__))
KILOSORT2_DIR = os.environ.get('KILOSORT2_DIR', f'{PYKILOSORT_DIR}/../Kilosort2')

BASE_PATH = os.environ.get('BASE_PATH', f'{PYKILOSORT_DIR}/examples/eMouse/data')

rootZ = '/home/alexmorley/git_repos/Kilosort2/datashift'
rootH = '/tmp/'
pathToYourConfigFile = '/home/alexmorley/git_repos/Kilosort2/configFiles'
chanMapFile = 'NP2_kilosortChanMap.mat';

opts = {
    'chanMap': f'{rootZ}/{chanMapFile}',
    'fs': 30000.,
    'fshigh': 150.,
    'minfr_goodchannels': 0.1000,
    'Th': [6.0, 2.0],
    'lam': 10.,
    'AUCsplit': 0.9000,
    'minFR': 0.0200,
    'momentum': [20., 400],
    'sigmaMask': 30.,
    'ThPre': 8.,
    'reorder': 1,
    'nskip': 25.,
    'spkTh': -6.,
    'GPU': 1,
    'nfilt_factor': 4.,
    'ntbuff': 64.0,
    'NT': 65600.,
    'whiteningRange': 32.,
    'nSkipCov': 25.0,
    'scaleproc': 200.,
    'nPCs': 3.,
    'useRAM': 0,
    'sorting': 2,
    #'NchanTOT': float(simulation_opts['NchanTOT']),
    'trange': [0., float('inf')],
    'fproc': '/tmp/temp_wh.dat',
    'rootZ': rootZ,
    'fbinary': f'{rootZ}/data_cropped.bin',
    'fig': False
}

opts= {
    'chanMap': f'{rootZ}/{chanMapFile}',
    'trange': [0., float('inf')],
    'NchanTOT': float(384),
    'minfr_goodchannels': 0.,
    'sig': 20,
    'fshigh': 300,
    'trackfinal': 0,
    'nblocks': 5,
    'datashift': 1,
    'fbinary': f'{rootZ}/data_cropped.bin',
    'fs': 30000.,
}

# +
import logging
import shutil
from pathlib import Path
from phylib.io.traces import get_ephys_reader

import numpy as np
from pprint import pprint
from pydantic import BaseModel

from pykilosort.preprocess import preprocess, get_good_channels, get_whitening_matrix, get_Nbatch
from pykilosort.cluster import clusterSingleBatches
from pykilosort.learn import learnAndSolve8b
from pykilosort.postprocess import find_merges, splitAllClusters, set_cutoff, rezToPhy
from pykilosort.utils import Bunch, Context, memmap_large_array, load_probe
from pykilosort.params import KilosortParams
# -

probe = pykilosort.Bunch()
probe.NchanTOT = int(opts['NchanTOT'])
probe.chanMap = np.load(f'{rootZ}/channel_map.npy').flatten().astype(int)
probe.kcoords = np.ones(int(opts['NchanTOT']))
probe.xc = np.load(rootZ+'/channel_positions.npy')[:,0]
probe.yc = np.load(rootZ+'/channel_positions.npy')[:,1]

dat_path = opts['fbinary']
dir_path = Path(rootZ)
output_dir = Path(rootZ+'/py')
probe = probe
dtype = np.int16
n_channels = int(opts['NchanTOT'])
sample_rate = opts['fs']
clear_context = False #Operations.pykilosort_sorting in FORCE_RUN

# +
### Setup

# +
params = pykilosort.params.KilosortParams(**opts)
raw_data = get_ephys_reader(dat_path, dtype=dtype, sample_rate=sample_rate, n_channels=n_channels)
dir_path = dir_path or Path(dat_path).parent
n_samples, n_channels = raw_data.shape
logger.info("Loaded raw data with %d channels, %d samples.", n_channels, n_samples)

# Create the context.
ctx_path = dir_path / ".kilosort" / raw_data.name
if clear_context:
    logger.info(f"Clearing context at {ctx_path} ...")
    shutil.rmtree(ctx_path, ignore_errors=True)
ctx = Context(ctx_path)
ctx.params = params
ctx.probe = probe
ctx.raw_data = raw_data    
ctx.load()
ir = ctx.intermediate
ir.Nbatch = get_Nbatch(raw_data, params)
probe.Nchan = probe.NchanTOT
params.Nfilt = params.nfilt_factor * probe.Nchan

# +
### Preprocess
# -

from pykilosort.preprocess import get_whitening_matrix

from importlib import reload
reload(pykilosort.preprocess)
from pykilosort.preprocess import get_whitening_matrix

ir.Wrot = pykilosort.preprocess.get_whitening_matrix(
    raw_data=raw_data, probe=probe, params=params
)
ctx.write(Wrot=ir.Wrot)

ir.proc_path = ctx.path("proc", ".dat")
preprocess(ctx)

ir.proc_path
ir.proc = np.memmap(ir.proc_path, dtype=raw_data.dtype, mode="r", order="F")

# +
### Run datashift
# -

import math

# +
ir.xc, ir.yc = probe.xc, probe.yc
ir.ops = Bunch()

# The min and max of the y and x ranges of the channels
ymin = min(ir.yc)
ymax = max(ir.yc)
xmin = min(ir.xc)
xmax = max(ir.xc)

# Determine the average vertical spacing between channels. 
# Usually all the vertical spacings are the same, i.e. on Neuropixels probes. 
dmin = np.median(np.diff(np.unique(ir.yc)))
print(f"pitch is {dmin} um\n")
ir.ops.yup = np.arange(start=ymin, step=dmin/2, stop=ymax) # centers of the upsampled y positions

# Determine the template spacings along the x dimension
x_range = xmax - xmin
npt = math.floor(x_range/16) # this would come out as 16um for Neuropixels probes, which aligns with the geometry. 
ir.ops.xup = np.linspace(xmin, xmax, npt+1) # centers of the upsampled x positions

spkTh = 10 # same as the usual "template amplitude", but for the generic templates

# +
ir.xc, ir.yc = probe.xc, probe.yc
ir.ops = Bunch()

# The min and max of the y and x ranges of the channels
ymin = min(ir.yc)
ymax = max(ir.yc)
xmin = min(ir.xc)
xmax = max(ir.xc)

# Determine the average vertical spacing between channels. 
# Usually all the vertical spacings are the same, i.e. on Neuropixels probes. 
dmin = np.median(np.diff(np.unique(ir.yc)))
print(f"pitch is {dmin} um\n")
ir.ops.yup = np.arange(start=ymin, step=dmin/2, stop=ymax) # centers of the upsampled y positions

# Determine the template spacings along the x dimension
x_range = xmax - xmin
npt = math.floor(x_range/16) # this would come out as 16um for Neuropixels probes, which aligns with the geometry. 
ir.ops.xup = np.linspace(xmin, xmax, npt+1) # centers of the upsampled x positions

spkTh = 10 # same as the usual "template amplitude", but for the generic templates

# Extract all the spikes across the recording that are captured by the
# generic templates. Very few real spikes are missed in this way. 
st3 = standalone_detector(ir, spkTh)

# binning width across Y (um)
dd = 5

# detected depths
dep = st3[:,2]

# min and max for the range of depths
dmin = ymin - 1
dep = dep - dmin

dmax  = 1 + ceil(max(dep)/dd)
Nbatches      = ir.temp.Nbatch

# which batch each spike is coming from
batch_id = st3[:,5] #ceil[st3[:,1]/dt]

# preallocate matrix of counts with 20 bins, spaced logarithmically
F = zeros(dmax, 20, Nbatches)
for t = 1:Nbatches
    # find spikes in this batch
    ix = np.where(batch_id==t)
    
    # subtract offset
    dep = st3(ix,2) - dmin
    
    # amplitude bin relative to the minimum possible value
    amp = log10(min(99, st3(ix,3))) - log10(spkTh)
    
    # normalization by maximum possible value
    amp = amp / (log10(100) - log10(spkTh))
    
    # multiply by 20 to distribute a [0,1] variable into 20 bins
    # sparse is very useful here to do this binning quickly
    M = sparse(ceil(dep/dd), ceil(1e-5 + amp * 20), ones(numel(ix), 1), dmax, 20)    
    
    # the counts themselves are taken on a logarithmic scale (some neurons
    # fire too much!)
    F[:, :, t] = log2(1+M)
end

##
# the 'midpoint' branch is for chronic recordings that have been
# concatenated in the binary file
#if isfield(ops, 'midpoint')
#    # register the first block as usual
#    [imin1, F1] = align_block(F(:, :, 1:ops.midpoint))
#    # register the second block as usual
#    [imin2, F2] = align_block(F(:, :, ops.midpoint+1:end))
#    # now register the average first block to the average second block
#    d0 = align_pairs(F1, F2)
#    # concatenate the shifts
#    imin = [imin1 imin2 + d0]
#    imin = imin - mean(imin)
#    ops.datashift = 1
#else
#    # determine registration offsets 
#    ysamp = dmin + dd * [1:dmax] - dd/2
#    [imin,yblk, F0] = align_block2(F, ysamp, ops.nblocks)
#end

##
if opts.get('fig', True):  
    ax = plt.subplot()
    # plot the shift trace in um
    ax.plot(imin * dd)
    
    ax = plt.subplot()
    # raster plot of all spikes at their original depths
    st_shift = st3(:,2) #+ imin(batch_id)' * dd
    for j = spkTh:100
        # for each amplitude bin, plot all the spikes of that size in the
        # same shade of gray
        ix = st3(:, 3)==j # the amplitudes are rounded to integers
        ax.plot(st3(ix, 1), st_shift(ix), '.', 'color', [1 1 1] * max(0, 1-j/40)) # the marker color here has been carefully tuned
    plt.tight_layout()

# if we're creating a registered binary file for visualization in Phy
if opts.get('fbinaryproc', False):
    with open(opts['fbinaryproc'], 'w') as f:
        pass

# convert to um 
dshift = imin * dd
# sort in case we still want to do "tracking"

_, ir.iorig = np.sort(np.mean(dshift, 2))

# sigma for the Gaussian process smoothing
sig = ir.ops.sig
# register the data batch by batch
for ibatch in range(Nbatches):
    shift_batch_on_disk2(ir, ibatch, dshift[ibatch, :], yblk, sig)
end
fprintf('time #2.2f, Shifted up/down #d batches. \n', toc, Nbatches)

# keep track of dshift 
ir.dshift = dshift
# keep track of original spikes
ir.st0 = st3


# next, we can just run a normal spike sorter, like Kilosort1, and forget about the transformation that has happened in here 

##

# +
Nbatch = ctx.intermediate.Nbatch
proc = ir.proc

NT = params.NT
nPCs = params.nPCs
Nchan = probe.Nchan

batchstart = np.arange(0, NT * Nbatch + 1, NT).astype(np.int64)

# extract the PCA projections
# initialize the covariance of single-channel spike waveforms
CC = cp.zeros(params.nt0, dtype=np.float32)

# from every 100th batch
for ibatch in range(0, Nbatch, 100):
    offset = Nchan * batchstart[ibatch]
    dat = proc.flat[offset:offset + NT * Nchan].reshape((-1, Nchan),
                                                        order='F')
    if dat.shape[0] == 0:
        continue
dat

# +
if "st3" not in ir:
    with ctx.time("learn"):
        out = learnAndSolve8b(ctx)
    logger.info("%d spikes.", ir.st3.shape[0])
    ctx.save(**out)
if stop_after == "learn":
    return ctx
# Special care for cProj and cProjPC which are memmapped .dat files.
ir.cProj = memmap_large_array(ctx.path("fW", ext=".dat")).T
ir.cProjPC = memmap_large_array(ctx.path("fWpc", ext=".dat")).T  # transpose

# -------------------------------------------------------------------------
# Final merges.
#
# This function uses:
#
#       st3, simScore
#
# This function saves:
#
#         st3_m,
#         R_CCG, Q_CCG, K_CCG [optional]
#
if "st3_m" not in ir:
    with ctx.time("merge"):
        out = find_merges(ctx, True)
    ctx.save(**out)
if stop_after == "merge":
    return ctx

# -------------------------------------------------------------------------
# Final splits.
#
# This function uses:
#
#       st3_m
#       W, dWU, cProjPC,
#       iNeigh, simScore
#       wPCA
#
# This function saves:
#
#       st3_s
#       W_s, U_s, mu_s, simScore_s
#       iNeigh_s, iNeighPC_s,
#       Wphy, iList, isplit
#
if "st3_s1" not in ir:
    # final splits by SVD
    with ctx.time("split_1"):
        out = splitAllClusters(ctx, True)
    # Use a different name for both splitting steps.
    out["st3_s1"] = out.pop("st3_s")
    ctx.save(**out)
if stop_after == "split_1":
    return ctx

if "st3_s0" not in ir:
    # final splits by amplitudes
    with ctx.time("split_2"):
        out = splitAllClusters(ctx, False)
    out["st3_s0"] = out.pop("st3_s")
    ctx.save(**out)
if stop_after == "split_2":
    return ctx

# -------------------------------------------------------------------------
# Decide on cutoff.
#
# This function uses:
#
#       st3_s
#       dWU, cProj, cProjPC
#       wPCA
#
# This function saves:
#
#       st3_c, spikes_to_remove,
#       est_contam_rate, Ths, good
#
if "st3_c" not in ir:
    with ctx.time("cutoff"):
        out = set_cutoff(ctx)
    ctx.save(**out)
if stop_after == "cutoff":
    return ctx

logger.info("%d spikes after cutoff.", ir.st3_c.shape[0])
logger.info("Found %d good units.", np.sum(ir.good > 0))

# write to Phy
logger.info("Saving results to phy.")
output_dir = output_dir or f"{dir_path}/output"
with ctx.time("output"):
    rezToPhy(ctx, dat_path=dat_path, output_dir=output_dir)

# Show timing information.
ctx.show_timer()
ctx.write(timer=ctx.timer)
