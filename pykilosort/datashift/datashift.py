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
ir.ops.yup = np.arange(
    start=ymin, step=dmin / 2, stop=ymax
)  # centers of the upsampled y positions

# Determine the template spacings along the x dimension
x_range = xmax - xmin
npt = math.floor(
    x_range / 16
)  # this would come out as 16um for Neuropixels probes, which aligns with the geometry.
ir.ops.xup = np.linspace(xmin, xmax, npt + 1)  # centers of the upsampled x positions

spkTh = 10  # same as the usual "template amplitude", but for the generic templates

# Extract all the spikes across the recording that are captured by the
# generic templates. Very few real spikes are missed in this way.
st3 = standalone_detector(ir, spkTh)

# binning width across Y (um)
dd = 5

# detected depths
dep = st3[:, 2]

# min and max for the range of depths
dmin = ymin - 1
dep = dep - dmin

dmax = 1 + ceil(max(dep) / dd)
Nbatches = ir.temp.Nbatch

# which batch each spike is coming from
batch_id = st3[:, 5]  # ceil[st3[:,1]/dt]

# preallocate matrix of counts with 20 bins, spaced logarithmically
F = np.zeros(dmax, 20, Nbatches)
for t in range(Nbatches):
    # find spikes in this batch
    ix = np.where(batch_id == t)

    # subtract offset
    dep = st3[ix, 2] - dmin

    # amplitude bin relative to the minimum possible value
    amp = log10(min(99, st3[ix, 3])) - log10(spkTh)

    # normalization by maximum possible value
    amp = amp / (log10(100) - log10(spkTh))

    # multiply by 20 to distribute a [0,1] variable into 20 bins
    # sparse is very useful here to do this binning quickly
    M = sparse(ceil(dep / dd), ceil(1e-5 + amp * 20), ones(numel(ix), 1), dmax, 20)

    # the counts themselves are taken on a logarithmic scale (some neurons
    # fire too much!)
    F[:, :, t] = log2(1 + M)
end

##
# the 'midpoint' branch is for chronic recordings that have been
# concatenated in the binary file
# if isfield(ops, 'midpoint')
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
# else
#    # determine registration offsets
#    ysamp = dmin + dd * [1:dmax] - dd/2
#    [imin,yblk, F0] = align_block2(F, ysamp, ops.nblocks)
# end

##
if opts.get("fig", True):
    ax = plt.subplot()
    # plot the shift trace in um
    ax.plot(imin * dd)

    ax = plt.subplot()
    # raster plot of all spikes at their original depths
    st_shift = st3[:, 2]  # + imin(batch_id)' * dd
    for j in range(spkTh, 100):
        # for each amplitude bin, plot all the spikes of that size in the
        # same shade of gray
        ix = st3[:, 3] == j  # the amplitudes are rounded to integers
        ax.plot(
            st3[ix, 1],
            st_shift[ix],
            ".",
            "color",
            [max(0, 1 - j / 40) for i in range(3)],
        )  # the marker color here has been carefully tuned
    plt.tight_layout()

# if we're creating a registered binary file for visualization in Phy
if opts.get("fbinaryproc", False):
    with open(opts["fbinaryproc"], "w") as f:
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
fprintf("time #2.2f, Shifted up/down #d batches. \n", toc, Nbatches)

# keep track of dshift
ir.dshift = dshift
# keep track of original spikes
ir.st0 = st3

# next, we can just run a normal spike sorter, like Kilosort1, and forget about the transformation that has happened in here
