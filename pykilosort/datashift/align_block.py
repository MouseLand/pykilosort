import cupy as cp
import numpy as np

import numpy.matlib
from pykilosort.postprocess import my_conv2_cpu


def kernelD(xp0, yp0, length):
    D = xp0.shape[0]
    N = xp0.shape[1] if len(xp0.shape) > 1 else 1
    M = yp0.shape[1] if len(yp0.shape) > 1 else 1

    K = np.zeros((N, M))
    cs = M

    for i in range(int(M * 1.0 / cs)):
        ii = np.arange(i * cs, min(M, (i + 1) * cs))
        mM = len(ii)

        xp = np.matlib.repmat(xp0, mM, 1).T[np.newaxis, :, :]
        yp = np.matlib.repmat(yp0[:, ii], N, 1).reshape((D, N, mM))
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


if False:
    imin, yblk, F0 = align_block2(F, ysamp, nblocks)
