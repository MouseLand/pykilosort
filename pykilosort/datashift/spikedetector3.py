def spikedetector3(Params, drez, wTEMP, iC):
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
    d_data = cp.asarray(data, dtype=np.float32, order="F")
    d_W = cp.asarray(wTEMP, dtype=np.float32, order="F")
    d_iC = cp.asarray(iC, dtype=np.int32, order="F")
    d_dist = cp.asarray(dist, dtype=np.float32, order="F")
    d_v2 = cp.asarray(v2, dtype=np.float32, order="F")
    d_iC2 = cp.asarray(iC2, dtype=np.int32, order="F")
    d_dist2 = cp.asarray(dist2, dtype=np.float32, order="F")

    d_dout = cp.zeros((NT, Nchan), dtype=np.float32, order="F")
    d_kkmax = cp.zeros((NT, Nchan), dtype=np.int32, order="F")

    d_dfilt = cp.zeros((Nrank, NT, Nchan), dtype=np.float32, order="F")
    d_dmax = cp.zeros((NT, NchanUp), dtype=np.float32, order="F")
    d_st = cp.zeros(maxFR, dtype=np.int32, order="F")
    d_cF = cp.zeros((Nnearest, maxFR), dtype=np.float32, order="F")
    d_counter = cp.zeros(2, dtype=np.int32, order="F")

    counter = np.zeros(2, dtype=np.int32, order="F")

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
    counter = d_counter[0]
