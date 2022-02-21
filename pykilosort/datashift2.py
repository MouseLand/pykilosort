from math import floor
import logging
import os

from tqdm.auto import tqdm, trange
import numpy as np
import cupy as cp
from scipy.interpolate import Akima1DInterpolator
from scipy.sparse import coo_matrix

from .postprocess import my_conv2_cpu
from .learn import extractTemplatesfromSnippets
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
    dt = np.arange(-n, n + 1)

    # we do everything on the GPU for speed, but it's probably fast enough on
    # the CPU
    Fg = F

    # mean subtraction to compute covariance
    Fg = Fg - np.mean(Fg, axis=0)

    # initialize the target "frame" for alignment with a single sample
    F0 = Fg[:, :, min(299, np.floor(Fg.shape[2] / 2).astype("int")) - 1]
    F0 = F0[:, :, np.newaxis]

    # first we do rigid registration by integer shifts
    # everything is iteratively aligned until most of the shifts become 0.
    niter = 10
    dall = np.zeros((niter, Nbatches))
    for iteration in range(niter):
        for t in range(len(dt)):
            # for each NEW potential shift, estimate covariance
            Fs = np.roll(Fg, dt[t], axis=0)
            dc[t, :] = np.mean(Fs * F0, axis=(0,1))
        if iteration + 1 < niter:
            # up until the very last iteration, estimate the best shifts
            imax = np.argmax(dc, axis=0)
            # align the data by these integer shifts
            for t in range(len(dt)):
                ib = imax == t
                Fg[:, :, ib] = np.roll(Fg[:, :, ib], dt[t], axis=0)
                dall[iteration, ib] = dt[t]
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
    yblk = np.zeros(len(ifirst))

    # for each small block, we only look up and down this many samples to find
    # nonrigid shift
    n = 5
    dt = np.arange(-n, n + 1)

    # this part determines the up/down covariance for each block without
    # shifting anything
    dcs = np.zeros((2 * n + 1, Nbatches, nblocks))
    for j in range(nblocks):
        isub = np.arange(ifirst[j], ilast[j]+1)
        yblk[j] = np.mean(ysamp[isub])
        Fsub = Fg[isub, :, :]
        for t in range(len(dt)):
            Fs = np.roll(Fsub, dt[t], axis=0)
            dcs[t, :, j] = np.mean(Fs * F0[isub, :, :], axis=(0,1))

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
    distx = np.abs(xp[:, 0] - yp[:, 0][np.newaxis, :].T).T
    disty = np.abs(xp[:, 1] - yp[:, 1][np.newaxis, :].T).T

    sigx = sig
    sigy = 1.5 * sig

    p = 1
    K = np.exp(-((distx / sigx) ** p) - (disty / sigy) ** p)

    return K


def shift_data(data, shift_matrix):
    """
    Applies the shift transformation to the data via matrix multiplication
    :param data: Data matrix to be shifted, numpy memmap array, (n_time, n_channels)
                dtype int16, f-contiguous
    :param shift_matrix: Tranformation matrix, numpy array, (n_channels, n_channels)
                dtype float64, c-contiguous
    :return: Shifted data, numpy array, (n_time, n_channels)
                dtype int16, f-contiguous
    """

    data_shifted = np.asfortranarray((data @ shift_matrix.T).astype("int16"))

    return data_shifted


def interpolate_1D(sample_shifts, sample_coords, probe_coords):
    """
    Interpolates the shifts found in one dimension to estimate the shifts for each channel
    :param sample_shifts: Detected shifts, numpy array
    :param sample_coords: Coordinates at which the detected shifts were found, numpy array
    :param probe_coords: Coordinates of the probe channels, numpy array
    :return: Upsampled shifts for each channel, numpy array
    """

    assert len(sample_coords) == len(sample_shifts)

    if len(sample_coords) == 1:
        return np.full(len(probe_coords), sample_shifts[0])

    else:
        # zero pad input so "extrapolation" tends to zero
        # MATLAB uses a "modified Akima" which is proprietry :(
        # _ysamp = extended(ysamp, 2, 10000)
        # _shifts_in = zero_pad(shifts_in, 2)
        # interpolation_function = Akima1DInterpolator(_ysamp, _shifts_in)
        interpolation_function = Akima1DInterpolator(sample_coords, sample_shifts)

        # interpolation_function = interp1d(ysamp, shifts_in, kind='cubic', fill_value=([0],[0])) #'extrapolate')
        shifts = interpolation_function(probe_coords, nu=0, extrapolate="True")

        return shifts


def get_kernel_matrix(probe, shifts, sig):
    """
    Calculate kernel prediction matrix for Gaussian Kriging interpolation
    :param probe: Bunch object with individual numpy arrays for the channel coordinates
    :param shifts: Numpy array of the estimated shift for each channel
    :param sig: Standard deviation used in Gaussian interpolation, float value
    :return: Prediction matrix, numpy array
    """

    # 2D coordinates of the original channel positions
    coords_old = np.vstack([probe.xc, probe.yc]).T

    # 2D coordinates of the new channel positions
    coords_new = np.copy(coords_old)
    coords_new[:, 1] = coords_new[:, 1] - shifts

    # 2D kernel of the original channel positions
    Kxx = kernel2D(coords_old, coords_old, sig)

    # 2D kernel of the new channel positions
    Kyx = kernel2D(coords_new, coords_old, sig)

    # kernel prediction matrix
    prediction_matrix = Kyx @ np.linalg.pinv(Kxx + 0.01 * np.eye(Kxx.shape[0]))

    return prediction_matrix


def apply_drift_transform(dat, shifts_in, ysamp, probe, sig):
    """
    Apply kriging interpolation on data batch
    :param dat: Data batch, (n_time, n_channels)
    :param shifts_in: Shifts per block (n_blocks)
    :param ysamp: Y-coords for block centres (n_blocks)
    :param probe: Bunch object with xc (xcoords) and yc (ycoords) attributes
    :param sig: Standard deviation for Gaussian in kriging interpolation
    :return: Shifted data batch via a kriging transformation
    """

    # upsample to get shifts for each channel
    shifts = interpolate_1D(shifts_in, ysamp, probe.yc)

    # kernel prediction matrix
    kernel_matrix = get_kernel_matrix(probe, shifts, sig)

    # apply shift transformation to the data
    data_shifted = shift_data(dat, kernel_matrix)

    return data_shifted


def shift_batch_on_disk2(
    ibatch,
    shifts_in,
    ysamp,
    sig,
    probe,
    data_loader,
):

    # load the batch
    dat = data_loader.load_batch(ibatch, rescale=False)

    # upsample the shift for each channel using interpolation
    shifts = interpolate_1D(shifts_in, ysamp, probe.yc)

    # kernel prediction matrix
    kernel_matrix = get_kernel_matrix(probe, shifts, sig)

    # apply shift transformation to the data
    data_shifted = shift_data(dat, kernel_matrix)

    # write the aligned data back to the same file
    data_loader.write_batch(ibatch, data_shifted)


def standalone_detector(wTEMP, wPCA, NrankPC, yup, xup, Nbatch, data_loader, probe, params):
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
    dNearActiveSite = np.median(np.diff(np.unique(probe.yc)))
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
    for ibatch in range(0, nsizes):
        v2[ibatch, :] = np.sum(np.exp(-2 * (dist ** 2) / (sig * (ibatch + 1)) ** 2), 0)

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

    # preallocate the results we assume 50 spikes per channel per second max
    rl = data_loader.data.shape[0] / params.fs / probe.Nchan   # record length
    st3 = np.zeros((int(np.ceil(rl * 50 * probe.Nchan)), 5))
    st3[:, 4] = -1  # batch_id can be zero
    t0 = 0
    nsp = 0  # counter for total number of spikes

    pbar = trange(Nbatch, desc="Detecting Spikes, 0 batches, 0 spikes", leave=True)
    for ibatch in pbar:
        # get a batch of whitened and filtered data
        dataRAW = data_loader.load_batch(ibatch)

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

        # add time offset to get correct spike times
        toff = t0 + params.nt0min + params.NT * ibatch
        st[0, :] = st[0, :] + toff

        # st[4, :] = k  # add batch number
        st = np.concatenate([st, np.full((1, st.shape[1]), ibatch)])

        nsp0 = st.shape[1]
        if nsp0 + nsp > st3.shape[0]:
            # Pre-allocate more space if needed
            st3 = np.concatenate((st3, np.zeros((1000000, 5))), axis=0)

        st3[nsp : nsp0 + nsp, :] = st.T
        nsp = nsp + nsp0

        if ibatch % 10 == 0 or ibatch == (Nbatch - 1):
            pbar.set_description(f"Detecting Spikes, {ibatch+1} batches, {nsp} spikes", refresh=True)

    spikes = Bunch()
    spikes.times = st3[:nsp, 0]
    spikes.depths = st3[:nsp, 1]
    spikes.amps = st3[:nsp, 2]
    spikes.batches = st3[:nsp, 4]

    return spikes


def get_drift(spikes, probe, Nbatches, nblocks=5, genericSpkTh=10):
    """
    Estimates the drift using the spiking activity found in the first pass through the data
    :param spikes: Bunch object, contains the depths, amplitudes, times and batches of the spikes.
                    Each attribute is stored as a 1d numpy array
    :param probe: Bunch object, contains the x and y coordinates stored as 1D numpy arrays
    :param Nbatches: No batches in the dataset
    :param nblocks: No of blocks to divide the probe into when estimating drift
    :param genericSpkTh: Min amplitude of spiking activity found
    :return: dshift: 2D numpy array of drift estimates per batch and per sub-block in um
                    size (Nbatches, 2*nblocks-1)
            yblk: 1D numpy array containing average y position of each sub-block
    """

    # binning width across Y (um)
    dd = 5

    # min and max for the range of depths
    dmin = min(probe.yc) - 1

    dmax = int(1 + np.ceil((max(probe.yc) - dmin) / dd))

    # preallocate matrix of counts with 20 bins, spaced logarithmically
    F = np.zeros((dmax, 20, Nbatches))
    for t in range(Nbatches):
        # find spikes in this batch
        ix = np.where(spikes.batches == t)[0]

        # subtract offset
        spike_depths_batch = spikes.depths[ix] - dmin

        # amplitude bin relative to the minimum possible value
        spike_amps_batch = np.log10(np.clip(spikes.amps[ix], None, 99)) - np.log10(genericSpkTh)
        # normalization by maximum possible value
        spike_amps_batch = spike_amps_batch / (np.log10(100) - np.log10(genericSpkTh))

        # multiply by 20 to distribute a [0,1] variable into 20 bins
        # sparse is very useful here to do this binning quickly
        i, j, v, m, n = (
            np.ceil(1e-5 + spike_depths_batch / dd).astype("int"),
            np.minimum(np.ceil(1e-5 + spike_amps_batch * 20), 20).astype("int"),
            np.ones(len(ix)),
            dmax,
            20,
        )
        M = coo_matrix((v, (i-1, j-1)), shape=(m,n)).toarray()

        # the counts themselves are taken on a logarithmic scale (some neurons
        # fire too much!)
        F[:, :, t] = np.log2(1 + M)

    ysamp = dmin + dd * np.arange(1, dmax+1) - dd / 2
    imin, yblk, F0 = align_block2(F, ysamp, nblocks)

    # convert to um
    dshift = imin * dd

    return dshift, yblk


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
    logger.info(f"pitch is {dmin} um")
    yup = np.arange(
        start=ymin, step=dmin / 2, stop=ymax + (dmin / 2)
    )  # centers of the upsampled y positions

    # Determine the template spacings along the x dimension
    x_range = xmax - xmin
    npt = floor(
        x_range / 16
    )  # this would come out as 16um for Neuropixels probes, which aligns with the geometry.
    xup = np.linspace(xmin, xmax, npt + 1)  # centers of the upsampled x positions

    # Set seed
    if params.seed:
        np.random.seed(params.seed)

    # determine prototypical timecourses by clustering of simple threshold crossings.
    wTEMP, wPCA = extractTemplatesfromSnippets(
        data_loader=ir.data_loader, probe=probe, params=params, Nbatch=Nbatch
    )

    # Extract all the spikes across the recording that are captured by the
    # generic templates. Very few real spikes are missed in this way.
    spikes = standalone_detector(
        wTEMP, wPCA, params.nPCs, yup, xup, Nbatch, ir.data_loader, probe, params
    )

    if params.save_drift_spike_detections:
        drift_path = ctx.context_path / 'drift'
        if not os.path.isdir(drift_path):
            os.mkdir(drift_path)
        np.save(drift_path / 'spike_times.npy', spikes.times)
        np.save(drift_path / 'spike_depths.npy', spikes.depths)
        np.save(drift_path / 'spike_amps.npy', spikes.amps)

    dshift, yblk = get_drift(spikes, probe, Nbatch, params.nblocks, params.genericSpkTh)

    # sort in case we still want to do "tracking"
    iorig = np.argsort(np.mean(dshift, axis=1))

    # register the data batch by batch
    for ibatch in tqdm(range(Nbatch), desc='Shifting Data'):

        # load the batch from binary file
        dat = ir.data_loader.load_batch(ibatch, rescale=False)

        # align via kriging interpolation
        data_shifted = apply_drift_transform(dat, dshift[ibatch, :], yblk, probe, params.sig_datashift)

        # write the aligned data back to the same file
        ir.data_loader.write_batch(ibatch, data_shifted)

    logger.info(f"Shifted up/down {Nbatch} batches")

    return Bunch(iorig=iorig, dshift=dshift, yblk=yblk)
