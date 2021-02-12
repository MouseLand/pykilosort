from math import floor
import logging
import os
from os.path import join
from pathlib import Path
import shutil

from tqdm import tqdm
import numba
import numpy as np
import cupy as cp
import cupyx as cpx
from scipy.interpolate import Akima1DInterpolator

from .postprocess import my_conv2_cpu
from .cptools import ones, svdecon, var, mean, free_gpu_memory
from .learn import extractTemplatesfromSnippets
from .preprocess import convolve_gpu, _is_vect, _make_vect
from .utils import get_cuda, Bunch

logger = logging.getLogger(__name__)


def getClosestChannels2(ycup, xcup, yc, xc, NchanClosest):
    # this function outputs the closest channels to each channel,
    # as well as a Gaussian-decaying mask as a function of pairwise distances
    # sigma is the standard deviation of this Gaussian-mask
    # compute distances between all pairs of channels
    xc = cp.asarray(xc, dtype=np.float32, order='F')
    yc = cp.asarray(yc, dtype=np.float32, order='F')
    xcup = cp.asarray(xcup, dtype=np.float32, order='F')
    ycup = cp.asarray(ycup, dtype=np.float32, order='F')
    C2C = ((xc[:, np.newaxis] - xcup[:].T.flatten()[:, np.newaxis].T) ** 2 + (
                yc[:, np.newaxis] - ycup[:].T.flatten()[:, np.newaxis].T) ** 2)
    C2C = cp.sqrt(C2C)
    Nchan, NchanUp = C2C.shape

    # sort distances
    isort = cp.argsort(C2C, axis=0)

    # take NchanCLosest neighbors for each primary channel
    iC = isort[:NchanClosest, :]

    # in some cases we want a mask that decays as a function of distance between pairs of channels
    # this is an awkward indexing to get the corresponding distances
    ix = iC + cp.arange(0, Nchan * NchanUp, Nchan)
    dist = C2C.T.ravel()[ix]

    return iC, dist


def get_batch(params, probe, ibatch, Nbatch, proc) -> cp.ndarray:
    batchstart = np.arange(0, params.NT * Nbatch + 1, params.NT).astype(np.int64)

    offset = probe.Nchan * batchstart[ibatch]
    dat = proc.flat[offset : offset + params.NT * probe.Nchan].reshape(
        (-1, probe.Nchan), order="F"
    )

    # move data to GPU and scale it back to unit variance
    dataRAW = cp.asarray(dat, dtype=np.float32) / params.scaleproc
    return dataRAW


def spikedetector3(Params, drez, wTEMP, iC, dist, v2, iC2, dist2):
    code, constants = get_cuda("spikedetector3")

    NT = int(Params[0])
    Nchan = int(Params[1])
    nt0 = int(Params[2])
    Nnearest = int(Params[3])
    Nrank = int(Params[4])
    NchanUp = int(Params[7])

    Nthreads = constants.Nthreads
    NrankMax = constants.NrankMax
    maxFR = constants.maxFR
    nt0max = constants.nt0max
    NchanMax = constants.NchanMax
    nsizes = constants.nsizes

    # tpB = (8, 2 * nt0 - 1)
    # tpF = (16, Nnearest)
    tpS = (nt0, 16)

    d_Params = cp.asarray(Params, dtype=np.float64, order="F")
    d_data = cp.asarray(drez, dtype=np.float32, order="F")
    d_W = cp.asarray(wTEMP, dtype=np.float32, order="F")
    d_iC = cp.asarray(iC, dtype=np.int32, order="F")
    d_dist = cp.asarray(dist, dtype=np.float32, order="F")
    d_v2 = cp.asarray(v2, dtype=np.float32, order="F")
    d_iC2 = cp.asarray(iC2, dtype=np.int32, order="F")
    d_dist2 = cp.asarray(dist2, dtype=np.float32, order="F")

    dimst = (NT, NchanUp)
    d_dout = cp.zeros(dimst, dtype=np.float32, order="F")
    d_kkmax = cp.zeros(dimst, dtype=np.int32, order="F")

    d_dfilt = cp.zeros((Nrank, NT, Nchan), dtype=np.float32, order="F")
    d_dmax = cp.zeros((NT, NchanUp), dtype=np.float32, order="F")
    d_st = cp.zeros((maxFR * 5), dtype=np.int32, order="F")
    d_cF = cp.zeros((maxFR * Nnearest,), dtype=np.float32, order="F")
    d_counter = cp.zeros(2, dtype=np.int32, order="F")

    # filter the data with the temporal templates
    Conv1D = cp.RawKernel(code, "Conv1D")
    Conv1D((Nchan,), (Nthreads,), (d_Params, d_data, d_W, d_dfilt))

    # sum each template across channels, square, take max
    sumChannels = cp.RawKernel(code, "sumChannels")
    tpP = (int(NT / Nthreads), NchanUp)
    sumChannels(
        tpP, (Nthreads,), (d_Params, d_dfilt, d_dout, d_kkmax, d_iC, d_dist, d_v2)
    )

    # get the max of the data
    max1D = cp.RawKernel(code, "max1D")
    max1D((NchanUp,), (Nthreads,), (d_Params, d_dout, d_dmax))

    # take max across nearby channels
    tpP = (int(NT / Nthreads), NchanUp)
    maxChannels = cp.RawKernel(code, "maxChannels")
    maxChannels(
        tpP,
        (Nthreads,),
        (
            d_Params,
            d_dout,
            d_dmax,
            d_iC,
            d_iC2,
            d_dist2,
            d_kkmax,
            d_dfilt,
            d_st,
            d_counter,
            d_cF,
        ),
    )
    counter = cp.asnumpy(d_counter)[0]

    minSize = min(maxFR, counter)
    d_sto = d_st[: 4 * minSize].reshape((4, minSize), order="F")
    d_cF2 = d_cF[: Nnearest * minSize].reshape((Nnearest, minSize), order="F")

    return d_dout.get(), d_kkmax.get(), d_sto.get(), d_cF2.get()


def kernelD(xp0, yp0, length):
    D = xp0.shape[0]
    N = xp0.shape[1] if len(xp0.shape) > 1 else 1
    M = yp0.shape[1] if len(yp0.shape) > 1 else 1

    K = np.zeros((N, M))
    cs = M

    for i in range(int(M * 1.0 / cs)):
        ii = np.arange(i * cs, min(M, (i + 1) * cs))
        mM = len(ii)

        xp = np.tile(xp0, (mM, 1)).T[np.newaxis, :, :]
        yp = np.tile(yp0[:, ii], (N, 1)).reshape((D, N, mM))
        a = (xp - yp) ** 2
        b = 1.0 / (length ** 2)
        Kn = np.exp(-np.sum((a * b) / 2, axis=0))

        K[:, ii] = Kn

    return K


def align_block2(F, ysamp, nblocks):

    # F is y bins by amp bins by batches
    # ysamp are the coordinates of the y bins in um

    Nbatches = F.shape[2]

    # look up and down this many y bins to find best alignment
    n = 15
    dc = np.zeros((2 * n + 1, Nbatches))
    dt = range(-n, n + 1)

    # we do everything on the GPU for speed, but it's probably fast enough on
    # the CPU
    Fg = F

    # mean subtraction to compute covariance
    Fg = Fg - np.mean(Fg, axis=0)

    # initialize the target "frame" for alignment with a single sample
    F0 = Fg[:, :, min(300, np.floor(Fg.shape[2] / 2).astype("int")) - 1]
    F0 = F0[:, :, np.newaxis]

    # first we do rigid registration by integer shifts
    # everything is iteratively aligned until most of the shifts become 0.
    niter = 10
    dall = np.zeros((niter, Nbatches))
    for iter in range(niter):
        for t in range(len(dt)):
            # for each NEW potential shift, estimate covariance
            Fs = np.roll(Fg, dt[t], axis=0)
            dc[t, :] = np.mean(np.mean(Fs * F0, axis=0), axis=0)
        if iter + 1 < niter:
            # up until the very last iteration, estimate the best shifts
            imax = np.argmax(dc, axis=0)
            # align the data by these integer shifts
            for t in range(len(dt)):
                ib = imax == t
                Fg[:, :, ib] = np.roll(Fg[:, :, ib], dt[t], axis=0)
                dall[iter, ib] = dt[t]
            # new target frame based on our current best alignment
            F0 = np.mean(Fg, axis=2)[:, :, np.newaxis]

    # now we figure out how to split the probe into nblocks pieces
    # if nblocks = 1, then we're doing rigid registration
    nybins = F.shape[0]
    yl = np.floor(nybins / nblocks).astype("int") - 1
    # MATLAB rounds 0.5 to 1. Python uses "Bankers Rounding".
    # Numpy uses round to nearest even. Force the result to be like MATLAB
    # by adding a tiny constant.
    ifirst = np.round(np.linspace(0, nybins - yl - 1, 2 * nblocks - 1) + 1e-10).astype(
        "int"
    )
    ilast = ifirst + yl  # 287

    ##

    nblocks = len(ifirst)
    yblk = np.zeros((len(ifirst), 1))

    # for each small block, we only look up and down this many samples to find
    # nonrigid shift
    n = 5
    dt = np.arange(-n, n + 1)

    # this part determines the up/down covariance for each block without
    # shifting anything
    dcs = np.zeros((2 * n + 1, Nbatches, nblocks))
    for j in range(nblocks):
        isub = np.arange(ifirst[j], ilast[j])
        yblk[j] = np.mean(ysamp[isub])
        Fsub = Fg[isub, :, :]
        for t in range(len(dt)):
            Fs = np.roll(Fsub, dt[t], axis=0)
            dcs[t, :, j] = np.mean(np.mean(Fs * F0[isub, :, :], axis=0), axis=0)

    # to find sub-integer shifts for each block ,
    # we now use upsampling, based on kriging interpolation
    dtup = np.linspace(-n, n, (2 * n * 10) + 1)
    K = kernelD(
        dt[np.newaxis, :], dtup[np.newaxis], 1
    )  # this kernel is fixed as a variance of 1
    dcs = cp.array(dcs)
    # dcs = my_conv2_cpu(dcs, .5, [0,1,2])
    for i in range(dcs.shape[0]):
        dcs[i, :, :] = my_conv2_cpu(
            dcs[i, :, :], 0.5, [0, 1]
        )  # some additional smoothing for robustness, across all dimensions
    for i in range(dcs.shape[2]):
        dcs[:, :, i] = my_conv2_cpu(
            dcs[:, :, i], 0.5, [0]
        )  # some additional smoothing for robustness, across all dimensions
        # dcs = my_conv2(cp.array(dcs), .5, [1, 2]) # some additional smoothing for robustness, across all dimensions
    dcs = dcs.get()

    # return K, dcs, dt, dtup
    imin = np.zeros((Nbatches, nblocks))
    for j in range(nblocks):
        # using the upsampling kernel K, get the upsampled cross-correlation
        # curves
        dcup = np.matmul(K.T, dcs[:, :, j])

        # find the max index of these curves
        imax = np.argmax(dcup, axis=0)

        # add the value of the shift to the last row of the matrix of shifts
        # (as if it was the last iteration of the main rigid loop )
        dall[niter - 1, :] = dtup[imax]

        # the sum of all the shifts equals the final shifts for this block
        imin[:, j] = np.sum(dall, axis=0)

    return imin, yblk, F0


def extended(ysamp, n, diff=None):
    if diff is None:
        diff = ysamp[1] - ysamp[0]
    pre = [ysamp[0] - i * diff for i in range(n, 0, -1)]
    post = [ysamp[-1] + i * diff for i in range(1, n)]
    return np.concatenate([pre, ysamp, post])


def zero_pad(shifts_in, n):
    pre = [0 for i in range(n, 0, -1)]
    post = [0 for i in range(1, n)]
    return np.concatenate([pre, shifts_in, post])


def kernel2D(xp, yp, sig):
    distx = np.abs(xp[:, 0] - yp[:, 0][np.newaxis, :].T)
    disty = np.abs(xp[:, 1] - yp[:, 1][np.newaxis, :].T)

    sigx = sig
    sigy = 1.5 * sig

    p = 1
    K = np.exp(-((distx / sigx) ** p) - (disty / sigy) ** p)

    return K


def shift_batch_on_disk2(
    ibatch,
    shifts_in,
    ysamp,
    sig,
    Nbatch,
    params,
    probe,
    proc,
    shifted_fname=None,
    overwrite=False,
    plot=False,
):

    # register one batch of a whitened binary file
    NT = params.NT
    Nchan = probe.Nchan

    batchstart = range(
        0, params.NT * Nbatch, params.NT
    )  # batches start at these timepoints
    offset = Nchan * batchstart[ibatch]
    offset_bytes = 2 * offset  # binary file offset in bytes

    # upsample the shift for each channel using interpolation
    if len(ysamp) > 1:
        # zero pad input so "extrapolation" tends to zero
        # MATLAB uses a "modified Akima" which is proprietry :(
        _ysamp = extended(ysamp, 2, 10000)
        _shifts_in = zero_pad(shifts_in, 2)
        interpolation_function = Akima1DInterpolator(_ysamp, _shifts_in)

        # interpolation_function = interp1d(ysamp, shifts_in, kind='cubic', fill_value=([0],[0])) #'extrapolate')
        shifts = interpolation_function(probe.yc, nu=0, extrapolate="True")


    # load the batch
    dat = proc.flat[offset : offset + params.NT * probe.Nchan].reshape(
        (-1, probe.Nchan), order="F"
    )  # / params.scaleproc

    # 2D coordinates for interpolation
    xp = np.vstack([probe.xc, probe.yc]).T

    # 2D kernel of the original channel positions
    Kxx = kernel2D(xp, xp, sig)

    # 2D kernel of the new channel positions
    yp = xp
    yp[:, 1] = yp[:, 1] - shifts  # * sig
    Kyx = kernel2D(yp, xp, sig)

    # kernel prediction matrix
    M = np.linalg.solve((Kxx + 0.01 * np.eye(Kxx.shape[0])), Kyx)

    # the multiplication has to be done on the GPU (but its not here)
    # dati = gpuArray(single(dat)) * gpuArray(M).T
    dati = dat @ M.T

    dat_cpu = np.asfortranarray(dati.astype("int16"))

    if shifted_fname is not None:
        # if the user wants to have a registered version of the binary file
        # this one is not split into batches
        mode = "ab" if ibatch == 0 else "wb"
        with open(shifted_fname, mode) as fid2:
            ifirst = params.ntbuff
            ilast = params.NT + 1
            if ibatch == 0:
                ifirst = 0
                ilast = params.NT - params.ntbuff + 1
            dat_cpu[ifirst:ilast, :].tofile(fid2)

    # if overwrite:
    #     with open(ops.fproc, "wb") as fid:
    #         fid.seek(offset_bytes)
    #         # normally we want to write the aligned data back to the same file
    #         dat_cpu.tofile(fid)  # write this batch to binary file

    return dat_cpu, dat, shifts


def standalone_detector(wTEMP, wPCA, NrankPC, yup, xup, Nbatch, proc, probe, params):
    """
    Detects spikes across the entire recording using generic templates.
    Each generic template has rank one (in space-time).
    In time, we use the 1D template prototypes found in wTEMP.
    In space, we use Gaussian weights of several sizes, centered
    on (x,y) positions that are part of a super-resolution grid covering the
    entire probe (pre-specified in the calling function).
    In total, there ~100x more generic templates than channels.
    """

    # minimum/base sigma for the Gaussian.
    sig = 10

    # grid of centers for the generic tempates
    ycup, xcup = np.meshgrid(yup, xup)

    # Get nearest channels for every template center.
    # Template products will only be computed on these channels.
    NchanNear = 10
    iC, dist = getClosestChannels2(ycup, xcup, probe.yc, probe.xc, NchanNear)

    # Templates with centers that are far from an active site are discarded
    dNearActiveSite = 30
    igood = dist[0, :] < dNearActiveSite
    iC = iC[:, igood]
    dist = dist[:, igood]
    ycup = cp.array(ycup).T.ravel()[igood]
    xcup = cp.array(xcup).T.ravel()[igood]

    # number of nearby templates to compare for local template maximum
    NchanNearUp = 10 * NchanNear
    iC2, dist2 = getClosestChannels2(ycup, xcup, ycup, xcup, NchanNearUp)

    # pregenerate the Gaussian weights used for spatial components
    nsizes = 5
    v2 = cp.zeros((5, dist.shape[1]), dtype=np.float32)
    for k in range(0, nsizes):
        v2[k, :] = np.sum(np.exp(-2 * (dist ** 2) / (sig * (k + 1)) ** 2), 0)

    # build up Params
    NchanUp = iC.shape[1]
    Params = (
        params.NT,
        probe.Nchan,
        params.nt0,
        NchanNear,
        NrankPC,
        params.nt0min,
        params.genericSpkTh,
        NchanUp,
        NchanNearUp,
        sig,
    )

    # preallocate the results
    st3 = np.zeros((1000000, 5))
    st3[:, 4] = -1  # batch_id can be zero
    t0 = 0
    nsp = 0  # counter for total number of spikes

    for k in tqdm(range(0, Nbatch), desc="Detecting Spikes"):
        # get a batch of whitened and filtered data
        dataRAW = get_batch(params, probe, k, Nbatch, proc)

        # run the CUDA function on this batch
        dat, kkmax, st, cF = spikedetector3(
            Params, dataRAW, wTEMP, iC, dist, v2, iC2, dist2
        )
        # upsample the y position using the center of mass of template products
        # coming out of the CUDA function.
        ys = probe.yc[cp.asnumpy(iC)]
        cF0 = np.maximum(cF, 0)
        cF0 = cF0 / np.sum(cF0, 0)
        iChan = st[1, :]
        yct = np.sum(cF0 * ys[:, iChan], 0)

        # build st for the current batch
        st[1, :] = yct

        # the first batch is special (no pre-buffer)
        ioffset = params.ntbuff if k > 0 else 0

        toff = params.nt0min + t0 - ioffset + (params.NT - params.ntbuff) * k
        st[0, :] = st[0, :] + toff
        # these offsets ensure the times are computed correctly

        # st[4, :] = k  # add batch number
        st = np.concatenate([st, np.full((1, st.shape[1]), k)])

        nsp0 = st.shape[1]
        if nsp0 + nsp > st3.shape[0]:
            # Pre-allocate more space if needed
            st3 = np.concatenate((st3, np.zeros((1000000, 5))), axis=0)

        st3[nsp : nsp0 + nsp, :] = st.T
        nsp = nsp + nsp0

        if k % 100 == 0 | k == (Nbatch - 1):
            logger.info(f"{k+1} batches, {nsp} spikes")
    return st3[:nsp]


def get_drift(spike_times, spike_depths, spike_amps, spike_batches, probe,
              nblocks=5, genericSpkTh = 10):

    ymin = min(probe.yc)

    # binning width across Y (um)
    dd = 5

    # min and max for the range of depths
    dmin = ymin

    dmax = int(1 + np.ceil(max(spike_depths) / dd))
    Nbatches = int(np.max(spike_batches + 1))


    # preallocate matrix of counts with 20 bins, spaced logarithmically
    F = np.zeros((dmax, 20, Nbatches))
    for t in range(Nbatches):
        # find spikes in this batch
        ix = np.where(spike_batches == t)[0]

        # subtract offset
        spike_depths_batch = spike_depths[ix] - dmin

        # amplitude bin relative to the minimum possible value
        amp = np.log10(np.clip(spike_amps[ix], None, 99)) - np.log10(genericSpkTh)
        # normalization by maximum possible value
        amp = amp / (np.log10(100) - np.log10(genericSpkTh))

        # multiply by 20 to distribute a [0,1] variable into 20 bins
        # sparse is very useful here to do this binning quickly
        i, j, v, m, n = (
            np.ceil(spike_depths_batch / dd).astype("int"),
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
    imin, yblk, F0 = align_block2(F, ysamp, nblocks)

    return imin, yblk


def datashift2(ctx):
    """
    Main function to re-register the preprocessed data
    """
    params = ctx.params
    probe = ctx.probe
    raw_data = ctx.raw_data
    ir = ctx.intermediate
    Nbatch = ir.Nbatch

    ir.xc, ir.yc = probe.xc, probe.yc

    # The min and max of the y and x ranges of the channels
    ymin = min(ir.yc)
    ymax = max(ir.yc)
    xmin = min(ir.xc)
    xmax = max(ir.xc)

    # Determine the average vertical spacing between channels.
    # Usually all the vertical spacings are the same, i.e. on Neuropixels probes.
    dmin = np.median(np.diff(np.unique(ir.yc)))
    logger.info(f"pitch is {dmin} um\n")
    yup = np.arange(
        start=ymin, step=dmin / 2, stop=ymax + (dmin / 2)
    )  # centers of the upsampled y positions

    # Determine the template spacings along the x dimension
    x_range = xmax - xmin
    npt = floor(
        x_range / 16
    )  # this would come out as 16um for Neuropixels probes, which aligns with the geometry.
    xup = np.linspace(xmin, xmax, npt + 1)  # centers of the upsampled x positions

    # determine prototypical timecourses by clustering of simple threshold crossings.
    wTEMP, wPCA = extractTemplatesfromSnippets(
        proc=ir.proc, probe=probe, params=params, Nbatch=Nbatch
    )

    # Extract all the spikes across the recording that are captured by the
    # generic templates. Very few real spikes are missed in this way.
    st3 = standalone_detector(
        wTEMP, wPCA, params.nPCs, yup, xup, Nbatch, ir.proc, probe, params
    )

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

    # convert to um
    dshift = imin * dd

    # sort in case we still want to do "tracking"
    ir.iorig = np.argsort(np.mean(dshift, axis=1))

    # for ibatch in range(Nbatches):
    #     # register the data batch by batch
    #     shift_batch_on_disk2(
    #         ibatch,
    #         dshift[ibatch, :],
    #         yblk,
    #         params.sig,
    #         Nbatches,
    #         params,
    #         probe,
    #         ir.proc,
    #         shifted_fname=params.output_filename,
    #         overwrite=params.overwrite,
    #     )
    # logger.info(f"Shifted up/down {Nbatches} batches")

    # keep track of dshift
    ir.dshift = dshift
    # keep track of original spikes
    ir.st0 = st3

    ir.F = F
    ir.F0 = F0

    return Bunch(iorig=ir.iorig, dshift=dshift, st0=st3, F=F, F0=F0)
