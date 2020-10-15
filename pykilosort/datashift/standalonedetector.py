import cupy as cu
import numpy as np
from time import time
from tqdm import tqdm

from pykilosort import cluster
from pykilosort.learn import extractTemplatesfromSnippets
from pykilosort.utils import get_cuda

from contextlib import contextmanager


@contextmanager
def timetracker(process_name: str):
    start = time()
    yield
    secs = time() - start
    # print(f"{process_name} took {secs} secs")


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

    with timetracker("allocate gpu"):
        d_Params = cu.asarray(Params, dtype=np.float64, order="F")
        d_data = cu.asarray(drez, dtype=np.float32, order="F")
        d_W = cu.asarray(wTEMP, dtype=np.float32, order="F")
        d_iC = cu.asarray(iC, dtype=np.int32, order="F")
        d_dist = cu.asarray(dist, dtype=np.float32, order="F")
        d_v2 = cu.asarray(v2, dtype=np.float32, order="F")
        d_iC2 = cu.asarray(iC2, dtype=np.int32, order="F")
        d_dist2 = cu.asarray(dist2, dtype=np.float32, order="F")

        dimst = (NT, NchanUp)
        d_dout = cu.zeros(dimst, dtype=np.float32, order="F")
        d_kkmax = cu.zeros(dimst, dtype=np.int32, order="F")

        d_dfilt = cu.zeros((Nrank, NT, Nchan), dtype=np.float32, order="F")
        d_dmax = cu.zeros((NT, NchanUp), dtype=np.float32, order="F")
        d_st = cu.zeros((maxFR * 5), dtype=np.int32, order="F")
        d_cF = cu.zeros((maxFR * Nnearest,), dtype=np.float32, order="F")
        d_counter = cu.zeros(2, dtype=np.int32, order="F")

    with timetracker("filter gpu"):
        # filter the data with the temporal templates
        Conv1D = cu.RawKernel(code, "Conv1D", ("-G", "-lineinfo"))
        Conv1D((Nchan,), (Nthreads,), (d_Params, d_data, d_W, d_dfilt))

    with timetracker("sum gpu"):
        # sum each template across channels, square, take max
        sumChannels = cu.RawKernel(code, "sumChannels", ("-G", "-lineinfo"))
        tpP = (int(NT / Nthreads), NchanUp)
        sumChannels(
            tpP, (Nthreads,), (d_Params, d_dfilt, d_dout, d_kkmax, d_iC, d_dist, d_v2)
        )

    with timetracker("max gpu"):
        # get the max of the data
        max1D = cu.RawKernel(code, "max1D", ("-G", "-lineinfo"))
        max1D((NchanUp,), (Nthreads,), (d_Params, d_dout, d_dmax))

    with timetracker("maxchannels gpu"):
        # take max across nearby channels
        tpP = (int(NT / Nthreads), NchanUp)
        maxChannels = cu.RawKernel(code, "maxChannels", backend="nvcc")
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
        counter = cu.asnumpy(d_counter)[0]

    with timetracker("resize gpu"):
        minSize = min(maxFR, counter)
        d_sto = d_st[: 4 * minSize].reshape((4, minSize), order="F")
        d_cF2 = d_cF[: Nnearest * minSize].reshape((Nnearest, minSize), order="F")

    return d_dout.get(), d_kkmax.get(), d_sto.get(), d_cF2.get()


# TODO: lets just make the data object "batch iterable" everywhere in the codebase
def get_batch(params, ibatch, Nbatch, proc) -> cu.ndarray:
    batchstart = np.arange(0, params.NT * Nbatch + 1, params.NT).astype(np.int64)

    offset = params.probe.Nchan * batchstart[ibatch]
    dat = proc.flat[offset : offset + params.NT * params.probe.Nchan].reshape(
        (-1, params.probe.Nchan), order="F"
    )

    # move data to GPU and scale it back to unit variance
    dataRAW = cu.asarray(dat, dtype=np.float32) / params.scaleproc
    return dataRAW


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
    iC, dist = cluster.getClosestChannels2(ycup, xcup, probe.yc, probe.xc, NchanNear)

    # Templates with centers that are far from an active site are discarded
    dNearActiveSite = 30
    igood = dist[0, :] < dNearActiveSite
    iC = iC[:, igood]
    dist = dist[:, igood]
    ycup = cu.array(ycup).T.ravel()[igood]
    xcup = cu.array(xcup).T.ravel()[igood]

    # number of nearby templates to compare for local template maximum
    NchanNearUp = 10 * NchanNear
    iC2, dist2 = cluster.getClosestChannels2(ycup, xcup, ycup, xcup, NchanNearUp)

    # pregenerate the Gaussian weights used for spatial components
    nsizes = 5
    v2 = cu.zeros((5, dist.shape[1]), dtype=np.float32)
    for k in range(0, nsizes):
        v2[k, :] = np.sum(np.exp(-2 * (dist ** 2) / (sig * (k + 1)) ** 2), 0)

    # build up Params
    NchanUp = iC.shape[1]
    Params = (
        params.NT,
        params.probe.Nchan,
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

    toc = time()
    for k in tqdm(range(0, Nbatch), desc="Detecting Spikes"):
        # get a batch of whitened and filtered data
        dataRAW = get_batch(params, k, Nbatch, proc)

        # run the CUDA function on this batch
        with timetracker("CUDA"):
            dat, kkmax, st, cF = spikedetector3(
                Params, dataRAW, wTEMP, iC, dist, v2, iC2, dist2
            )

        # upsample the y position using the center of mass of template products
        # coming out of the CUDA function.
        ys = probe.yc[cu.asnumpy(iC)]
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
            # extend array
            # st3[nsp + 1e6, 1] = 0 # if we need to preallocate more space
            raise NotImplementedError("Extra pre-allocation not implemented")

        st3[nsp : nsp0 + nsp, :] = st.T
        nsp = nsp + nsp0

        if k % 100 == 0 | k == (Nbatch - 1):
            print(f"{time() - toc} sec, {k+1} batches, {nsp} spikes")
            toc = time()
    return st3
