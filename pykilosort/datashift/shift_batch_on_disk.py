from scipy.interpolate import Akima1DInterpolator, interp1d
import numpy as np
import matplotlib.pyplot as plt


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
    proc,
    shifted_fname=None,
    overwrite=False,
    plot=False,
):

    # register one batch of a whitened binary file
    NT = params.NT
    Nchan = params.probe.Nchan

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
        shifts = interpolation_function(params.probe.yc, nu=0, extrapolate="True")

    if plot:
        plt.plot(shifts, color="b")
        plt.plot(expected_shifts, color="r", alpha=0.5)

    # load the batch
    dat = proc.flat[offset : offset + params.NT * params.probe.Nchan].reshape(
        (-1, params.probe.Nchan), order="F"
    )  # / params.scaleproc
    if False:  # plot:
        plt.figure(figsize=(10, 20))
        plt.plot(dat[0:1000, :100] + 1000 * np.arange(100), color="b")
        plt.plot(
            expected_dat[0:1000, :100] + 1000 * np.arange(100), color="r", alpha=0.5
        )

    # 2D coordinates for interpolation
    xp = np.vstack([params.probe.xc, params.probe.yc]).T

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

    if overwrite:
        with open(ops.fproc, "wb") as fid:
            fid.seek(offset_bytes)
            # normally we want to write the aligned data back to the same file
            dat_cpu.tofile(fid)  # write this batch to binary file

    return dat_cpu, dat, shifts
