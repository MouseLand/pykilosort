import logging
import shutil
from pathlib import Path
from phylib.io.traces import get_ephys_reader

import numpy as np
from pprint import pprint
from pydantic import BaseModel

from .preprocess import preprocess, get_good_channels, get_whitening_matrix, get_Nbatch
from .cluster import clusterSingleBatches
from .datashift2 import datashift2
from .learn import learnAndSolve8b, compress_templates
from .postprocess import find_merges, splitAllClusters, set_cutoff, rezToPhy
from .utils import Bunch, Context, memmap_large_array, load_probe, copy_bunch
from .params import KilosortParams

logger = logging.getLogger(__name__)


def default_probe(raw_data):
    nc = raw_data.shape[1]
    return Bunch(Nchan=nc, xc=np.zeros(nc), yc=np.arange(nc))


def run(
    dat_path: str = None,
    dir_path: Path = None,
    output_dir: Path = None,
    probe=None,
    params=None,
    stop_after=None,
    clear_context=False,
    **kwargs,
):
    """Launch KiloSort 2.

    probe has the following attributes:
    - xc
    - yc
    - kcoords
    - Nchan

    """

    # Get or create the probe object.
    if isinstance(probe, (str, Path)):
        probe = load_probe(probe)

    raw_data = get_ephys_reader(dat_path, **kwargs)
    assert raw_data.ndim == 2

    # Now, the initial raw data must be in C order, it will be converted to Fortran order
    # in the proc file step, so as to use the existing CUDA kernels from MATLAB.
    assert raw_data.shape[0] > raw_data.shape[1]  # nsamples > nchannels
    n_samples, n_channels = raw_data.shape
    logger.info("Loaded raw data with %d channels, %d samples.", n_channels, n_samples)

    # TODO: design - let's pass in all of the config already parsed and ready into this function
    #              - run should do 1 thing only - run the steps of the algorithm.
    # Get probe.
    probe = probe or default_probe(raw_data)
    assert probe

    # Get params.
    if not isinstance(params, BaseModel):
        params = KilosortParams(**params or {})
    assert params

    # dir path
    dir_path = dir_path or Path(dat_path).parent
    assert dir_path, "Please provide a dir_path"
    dir_path.mkdir(exist_ok=True, parents=True)
    assert dir_path.exists()

    # Create the context.
    ctx_path = dir_path / ".kilosort" / Path(raw_data.name).name
    if clear_context:
        logger.info(f"Clearing context at {ctx_path} ...")
        shutil.rmtree(ctx_path, ignore_errors=True)

    ctx = Context(ctx_path)
    ctx.params = params
    ctx.probe = probe
    ctx.raw_probe = copy_bunch(probe)
    ctx.raw_data = raw_data

    # Load the intermediate results to avoid recomputing things.
    ctx.load()
    # TODO: unclear - what if we have changed something e.g. a parameter? Shouldn't
    #               - we make the path depdendent on at least the hash of the params?
    #               - We should also be able to turn this off for easy testing / experimentation.
    ir = ctx.intermediate

    ir.Nbatch = get_Nbatch(raw_data, params)

    # -------------------------------------------------------------------------
    # Find good channels.
    # NOTE: now we use C order from loading up to the creation of the proc file, which is
    # in Fortran order.
    params.minfr_goodchannels = 0
    if params.minfr_goodchannels > 0:  # discard channels that have very few spikes
        if "igood" not in ir:
            # determine bad channels
            with ctx.time("good_channels"):
                ir.igood = get_good_channels(
                    raw_data=raw_data, probe=probe, params=params
                )
            # Cache the result.
            ctx.write(igood=ir.igood)
        if stop_after == "good_channels":
            return ctx

        # it's enough to remove bad channels from the channel map, which treats them
        # as if they are dead
        ir.igood = ir.igood.ravel().astype("bool")
        probe.chanMap = probe.chanMap[ir.igood]
        probe.xc = probe.xc[ir.igood]  # removes coordinates of bad channels
        probe.yc = probe.yc[ir.igood]
        probe.kcoords = probe.kcoords[ir.igood]
    probe.Nchan = len(
        probe.chanMap
    )  # total number of good channels that we will spike sort
    assert probe.Nchan > 0

    # upper bound on the number of templates we can have
    params.Nfilt = params.nfilt_factor * probe.Nchan

    # -------------------------------------------------------------------------
    # Find the whitening matrix.
    if "Wrot" not in ir:
        # outputs a rotation matrix (Nchan by Nchan) which whitens the zero-timelag covariance
        # of the data
        with ctx.time("whitening_matrix"):
            ir.Wrot = get_whitening_matrix(
                raw_data=raw_data, probe=probe, params=params
            )
        # Cache the result.
        ctx.write(Wrot=ir.Wrot)
    if stop_after == "whitening_matrix":
        return ctx

    # -------------------------------------------------------------------------
    # Preprocess data to create proc.dat
    ir.proc_path = ctx.path("proc", ".dat")
    if not ir.proc_path.exists():
        # Do not preprocess again if the proc.dat file already exists.
        with ctx.time("preprocess"):
            preprocess(ctx)
    if stop_after == "preprocess":
        return ctx

    # Open the proc file.
    # NOTE: now we are always in Fortran order.
    assert ir.proc_path.exists()
    ir.proc = np.memmap(ir.proc_path, dtype=raw_data.dtype, mode="r+", order="F")

    # -------------------------------------------------------------------------
    # # Time-reordering as a function of drift.
    # #
    # # This function saves:
    # #
    # #       iorig, ccb0, ccbsort
    # #
    # if "iorig" not in ir:
    #     with ctx.time("reorder"):
    #         out = clusterSingleBatches(ctx)
    #     ctx.save(**out)
    # if stop_after == "reorder":
    #     return ctx

    if "iorig" not in ir:
        with ctx.time("drift_correction"):
            out = datashift2(ctx)
        ctx.save(**out)
    if stop_after == "drift_correction":
        return ctx

    # -------------------------------------------------------------------------
    #  Main tracking and template matching algorithm.
    #
    # This function uses:
    #
    #         procfile
    #         iorig
    #
    # This function saves:
    #
    #         wPCA, wTEMP
    #         st3, simScore,
    #         cProj, cProjPC,
    #         iNeigh, iNeighPC,
    #         WA, UA, W, U, dWU, mu,
    #         W_a, W_b, U_a, U_b
    #
    if "st3" not in ir:
        with ctx.time("learn"):
            out = learnAndSolve8b(ctx)
        logger.info("%d spikes.", ir.st3.shape[0])
        ctx.save(**out)
    if stop_after == "learn":
        return ctx

    if "U_a" not in ir:
        with ctx.time("compress"):
            out = compress_templates(ctx)
        ctx.save(**out)
    if stop_after == "compress":
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

    #TODO: memmap/save large arrays between stages

    #if "st3_s0" not in ir:
    #    # final splits by amplitudes
    #    with ctx.time("split_2"):
    #        out = splitAllClusters(ctx, False)
    #    out["st3_s0"] = out.pop("st3_s")
    #    ctx.save(**out)
    #if stop_after == "split_2":
    #    return ctx

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

    #TODO:
    # Add optional deletion of temp files

    return ctx


# TODO: use these in the actual main function
def run_preprocess(ctx):
    params = ctx.params
    raw_data = ctx.raw_data
    probe = ctx.probe
    ir = ctx.intermediate

    ir.Nbatch = get_Nbatch(raw_data, params)

    if params.minfr_goodchannels > 0:  # discard channels that have very few spikes
        # determine bad channels
        with ctx.time("good_channels"):
            ir.igood = get_good_channels(raw_data=raw_data, probe=probe, params=params)
        # Cache the result.
        ctx.write(igood=ir.igood)

        # it's enough to remove bad channels from the channel map, which treats them
        # as if they are dead
        ir.igood = ir.igood.ravel()
        probe.chanMap = probe.chanMap[ir.igood]
        probe.xc = probe.xc[ir.igood]  # removes coordinates of bad channels
        probe.yc = probe.yc[ir.igood]
        probe.kcoords = probe.kcoords[ir.igood]

    probe.Nchan = len(
        probe.chanMap
    )  # total number of good channels that we will spike sort
    assert probe.Nchan > 0

    # upper bound on the number of templates we can have
    params.Nfilt = params.nfilt_factor * probe.Nchan

    # -------------------------------------------------------------------------
    # Find the whitening matrix.
    with ctx.time("whitening_matrix"):
        ir.Wrot = get_whitening_matrix(raw_data=raw_data, probe=probe, params=params)
    # Cache the result.
    ctx.write(Wrot=ir.Wrot)

    # -------------------------------------------------------------------------
    # Preprocess data to create proc.dat
    ir.proc_path = ctx.path("proc", ".dat")
    if not ir.proc_path.exists():
        # Do not preprocess again if the proc.dat file already exists.
        with ctx.time("preprocess"):
            preprocess(ctx)

    # Show timing information.
    ctx.show_timer()
    ctx.write(timer=ctx.timer)

    return ctx


def run_spikesort(ctx, sanity_plots=True, plot_widgets=None):
    raw_data = ctx.raw_data
    ir = ctx.intermediate

    assert ir.proc_path.exists()
    ir.proc = np.memmap(ir.proc_path, dtype=raw_data.dtype, mode="r", order="F")

    # -------------------------------------------------------------------------
    # Time-reordering as a function of drift.
    #
    # This function saves:
    #
    #       iorig, ccb0, ccbsort
    #
    if "iorig" not in ir:
        with ctx.time("reorder"):
            out = clusterSingleBatches(ctx, sanity_plots=sanity_plots, plot_widgets=plot_widgets, plot_pos=0)
        ctx.save(**out)

    # -------------------------------------------------------------------------
    #  Main tracking and template matching algorithm.
    #
    # This function uses:
    #
    #         procfile
    #         iorig
    #
    # This function saves:
    #
    #         wPCA, wTEMP
    #         st3, simScore,
    #         cProj, cProjPC,
    #         iNeigh, iNeighPC,
    #         WA, UA, W, U, dWU, mu,
    #         W_a, W_b, U_a, U_b
    #
    if "st3" not in ir:
        with ctx.time("learn"):
            out = learnAndSolve8b(ctx, sanity_plots=sanity_plots, plot_widgets=plot_widgets, plot_pos=1)
        logger.info("%d spikes.", ir.st3.shape[0])
        ctx.save(**out)
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

    if "st3_s0" not in ir:
        # final splits by amplitudes
        with ctx.time("split_2"):
            out = splitAllClusters(ctx, False)
        out["st3_s0"] = out.pop("st3_s")
        ctx.save(**out)

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

    logger.info("%d spikes after cutoff.", ir.st3_c.shape[0])
    logger.info("Found %d good units.", np.sum(ir.good > 0))

    # Show timing information.
    ctx.show_timer()
    ctx.write(timer=ctx.timer)

    return ctx


def run_export(ctx, dat_path, output_dir):
    # write to Phy
    logger.info("Saving results to phy.")
    output_dir = output_dir
    with ctx.time("output"):
        rezToPhy(ctx, dat_path=dat_path, output_dir=output_dir)

    # Show timing information.
    ctx.show_timer()
    ctx.write(timer=ctx.timer)
