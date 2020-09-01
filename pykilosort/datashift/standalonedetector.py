import cupy as cu
import numpy as np

from pykilosort.cluster import getClosestChannels2
from pykilosort.learn import extractTemplatesfromSnippets

# TODO: lets just make the data object "batch iterable" everywhere in the codebase
def get_batch(params, ibatch, Nbatch) -> cu.ndarray:
    batchstart = np.arange(0, params.NT * Nbatch + 1, params.NT).astype(np.int64)

    offset = Nchan * batchstart[ibatch]
    dat = proc.flat[offset : offset + params.NT * params.Nchan].reshape(
        (-1, params.Nchan), order="F"
    )

    # move data to GPU and scale it back to unit variance
    dataRAW = cp.asarray(dat, dtype=np.float32) / params.scaleproc
    return dataRAW


def standalone_detector(yup, xup, Nbatch, proc, probe, params):
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

    # determine prototypical timecourses by clustering of simple threshold crossings.
    NrankPC = 6
    wTEMP, wPCA = extractTemplatesfromSnippets(
        proc=proc, probe=probe, params=params, Nbatch=Nbatch
    )

    # Get nearest channels for every template center.
    # Template products will only be computed on these channels.
    NchanNear = 10
    iC, dist = getClosestChannels2(ycup, xcup, rez.yc, rez.xc, NchanNear)

    # Templates with centers that are far from an active site are discarded
    dNearActiveSite = 30
    igood = dist[1, :] < dNearActiveSite
    iC = iC[:, igood]
    dist = dist[:, igood]
    ycup = ycup[igood]
    xcup = xcup[igood]

    # number of nearby templates to compare for local template maximum
    NchanNearUp = 10 * NchanNear
    iC2, dist2 = getClosestChannels2(ycup, xcup, ycup, xcup, NchanNearUp)

    # pregenerate the Gaussian weights used for spatial components
    nsizes = 5
    v2 = cu.zeros(5, size(dist, 2), np.dtype("f"))
    for k in range(nsizes):
        v2[k, :] = np.sum(exp(-2 * dist ^ 2 / (sig * k) ^ 2), 1)
    end

    # build up Params
    NchanUp = iC.shape[1]
    Params = (
        params.NT,
        params.Nchan,
        params.nt0,
        NchanNear,
        NrankPC,
        params.nt0min,
        spkTh,
        NchanUp,
        NchanNearUp,
        sig,
    )

    # preallocate the results
    st3 = np.zeros(1000000, 5)
    t0 = 0
    nsp = 0  # counter for total number of spikes

    for k in range(Nbatch):
        # get a batch of whitened and filtered data
        dataRAW = get_batch(params, k, Nbatch)

        # run the CUDA function on this batch
        dat, kkmax, st, cF = spikedetector3(
            Params, dataRAW, wTEMP, iC - 1, dist, v2, iC2 - 1, dist2
        )

        # upsample the y position using the center of mass of template products
        # coming out of the CUDA function.
        ys = rez.yc[iC]
        cF0 = max(0, cF)
        cF0 = cF0 / np.sum(cF0, 1)
        iChan = st[2, :] + 1
        yct = np.sum(cF0 * ys[:, iChan], 1)

        # build st for the current batch
        st[2, :] = yct

        # the first batch is special (no pre-buffer)
        ioffset = 0 if k == 1 else ops.ntbuff

        toff = ops.nt0min + t0 - ioffset + (ops.NT - ops.ntbuff) * (k - 1)
        st[1, :] = (
            st[1, :] + toff
        )  # these offsets ensure the times are computed correctly

        st[5, :] = k  # batch number

        nsp0 = st.shape[1]
        if nsp0 + nsp > st3.shape[0]:
            # extend array
            # st3[nsp + 1e6, 1] = 0 # if we need to preallocate more space
            raise NotImplementedError("Extra pre-allocation not implemented")

        st3[nsp : nsp0 + nsp, :] = st.T
        nsp = nsp + nsp0

        if rem(k, 100) == 1 | k == ops.Nbatch:
            print(f"{toc} sec, {k} batches, {nsp} spikes")
