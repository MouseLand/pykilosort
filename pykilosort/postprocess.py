from math import ceil, erf, log as log_, sqrt
import logging
import os
from os.path import join
from pathlib import Path
import shutil

from tqdm.auto import tqdm
import numba
import numpy as np
import cupy as cp
import cupyx as cpx
from scipy.signal import lfilter
from scipy.sparse import coo_matrix

from .cptools import svdecon, var, mean, free_gpu_memory, convolve_gpu
from .cluster import getClosestChannels
from .learn import getKernels, getMeWtW, mexSVDsmall2
from .preprocess import _is_vect, _make_vect
from .utils import Bunch, NpyWriter, memmap_large_array, LargeArrayWriter

logger = logging.getLogger(__name__)


def log(x):
    if x == 0:
        return -np.inf
    return log_(x)


def my_conv2(x, sig, varargin=None, **kwargs):
    # TODO: Fix so output matches my_conv2_cpu
    # x is the matrix to be filtered along a choice of axes
    # sig is either a scalar or a sequence of scalars, one for each axis to be filtered
    # varargin can be the dimensions to do filtering, if len(sig) != x.shape
    # if sig is scalar and no axes are provided, the default axis is 2
    if sig <= .25:
        return x
    idims = 1
    if varargin is not None:
        idims = varargin
    idims = _make_vect(idims)
    if _is_vect(idims) and _is_vect(sig):
        sigall = sig
    else:
        sigall = np.tile(sig, len(idims))

    for sig, idim in zip(sigall, idims):
        Nd = x.ndim
        x = cp.transpose(x, [idim] + list(range(0, idim)) + list(range(idim + 1, Nd)))
        dsnew = x.shape
        x = cp.reshape(x, (x.shape[0], -1), order='F')

        tmax = ceil(4 * sig)
        dt = cp.arange(-tmax, tmax + 1)
        gaus = cp.exp(-dt ** 2 / (2 * sig ** 2))
        gaus = gaus / cp.sum(gaus)

        y = convolve_gpu(x, gaus, **kwargs)
        y = y.reshape(dsnew, order='F')
        y = cp.transpose(y, list(range(1, idim + 1)) + [0] + list(range(idim + 1, Nd)))
    return y


def my_conv2_cpu(x, sig, varargin=None, **kwargs):
    # (Alternative conv2 function for testing)
    # x is the matrix to be filtered along a choice of axes
    # sig is either a scalar or a sequence of scalars, one for each axis to be filtered
    # varargin can be the dimensions to do filtering, if len(sig) != x.shape
    # if sig is scalar and no axes are provided, the default axis is 2
    if sig <= .25:
        return x
    idims = 1
    if varargin is not None:
        idims = varargin
    idims = _make_vect(idims)
    if _is_vect(idims) and _is_vect(sig):
        sigall = sig
    else:
        sigall = np.tile(sig, len(idims))

    for sig, idim in zip(sigall, idims):
        Nd = x.ndim
        x = cp.transpose(x, [idim] + list(range(0, idim)) + list(range(idim + 1, Nd)))
        dsnew = x.shape
        x = cp.reshape(x, (x.shape[0], -1), order='F')

        tmax = ceil(4 * sig)
        dt = np.arange(-tmax, tmax + 1)
        gaus = np.exp(-dt ** 2 / (2 * sig ** 2))
        gaus = gaus / np.sum(gaus)

        cNorm = lfilter(gaus, np.array([1.]), np.concatenate((np.ones(dsnew[0]), np.zeros(tmax))))
        cNorm = cNorm[tmax:]

        x_n = cp.asnumpy(x)
        x_n = lfilter(gaus, np.array([1.]), np.concatenate((x_n, np.zeros((tmax, dsnew[1])))),
                      axis=0)
        x_n = x_n[tmax:]
        x_n = np.reshape(x_n, dsnew)

        x_n = x_n / cNorm.reshape(-1, 1)
        x = cp.array(x_n)
        x = cp.transpose(x, list(range(1, idim + 1)) + [0] + list(range(idim + 1, Nd)))

    return x


def ccg_slow(st1, st2, nbins, tbin):
    # this function efficiently computes the crosscorrelogram between two sets
    # of spikes (st1, st2), with tbin length each, timelags =  plus/minus nbins
    # and then estimates how refractory the cross-correlogram is, which can be used
    # during merge decisions.

    st1 = cp.sort(st1)  # makes sure spike trains are sorted in increasing order
    st2 = cp.sort(st2)

    dt = nbins * tbin

    N1 = max(1, len(st1))
    N2 = max(1, len(st2))
    T = cp.concatenate((st1, st2)).max() - cp.concatenate((st1, st2)).min()

    # we traverse both spike trains together, keeping track of the spikes in the first
    # spike train that are within dt of spikes in the second spike train

    ilow = 0  # lower bound index
    ihigh = 0  # higher bound index
    j = 0  # index of the considered spike

    K = cp.zeros(2 * nbins + 1)

    # (DEV_NOTES) the while loop below is far too slow as is

    while j <= N2 - 1:  # traverse all spikes in the second spike train

        while (ihigh <= N1 - 1) and (st1[ihigh] < st2[j] + dt):
            ihigh += 1  # keep increasing higher bound until it's OUTSIDE of dt range

        while (ilow <= N1 - 1) and (st1[ilow] <= st2[j] - dt):
            ilow += 1  # keep increasing lower bound until it's INSIDE of dt range

        if ilow > N1 - 1:
            break  # break if we exhausted the spikes from the first spike train

        if st1[ilow] > st2[j] + dt:
            # if the lower bound is actually outside of dt range, means we overshot (there were no
            # spikes in range)
            # simply move on to next spike from second spike train
            j += 1
            continue

        for k in range(ilow, ihigh):
            # for all spikes within plus/minus dt range
            ibin = cp.rint((st2[j] - st1[k]) / tbin).astype(int)  # convert ISI to integer

            K[ibin + nbins] += 1

        j += 1

    irange1 = cp.concatenate((cp.arange(1, nbins // 2), cp.arange(3 * nbins // 2, 2 * nbins)))
    irange2 = cp.arange(nbins - 50, nbins - 10)
    irange3 = cp.arange(nbins + 11, nbins + 50)

    # normalize the shoulders by what's expected from the mean firing rates
    # a non-refractive poisson process should yield 1

    Q00 = cp.sum(K[irange1]) / (len(irange1) * tbin * N1 * N2 / T)
    # do the same for irange 2
    Q01 = cp.sum(K[irange2]) / (len(irange2) * tbin * N1 * N2 / T)
    # compare to the other shoulder
    Q01 = max(Q01, cp.sum(K[irange3]) / (len(irange3) * tbin * N1 * N2 / T))

    R00 = max(mean(K[irange2]), mean(K[irange3]))  # take the biggest shoulder
    R00 = max(R00, mean(K[irange1]))  # compare this to the asymptotic shoulder

    # test the probability that a central area in the autocorrelogram might be refractory
    # test increasingly larger areas of the central CCG

    a = K[nbins]
    K[nbins] = 0

    Qi = cp.zeros(10)
    Ri = cp.zeros(10)

    for i in range(1, 11):
        irange = cp.arange(nbins - i, nbins + i + 1)  # for this central range of the CCG
        # compute the normalised ratio as above. this should be 1 if there is no refractoriness
        Qi0 = cp.sum(K[irange]) / (2 * i * tbin * N1 * N2 / T)
        Qi[i - 1] = Qi0  # save the normalised probability

        n = cp.sum(K[irange]) / 2
        lam = R00 * i

        # log(p) = log(lam) * n - lam - gammaln(n+1)

        # this is tricky: we approximate the Poisson likelihood with a gaussian of equal mean and
        # variance that allows us to integrate the probability that we would see <N spikes in the
        # center of the cross-correlogram from a distribution with mean R00*i spikes

        p = 1 / 2 * (1 + erf((n - lam) / cp.sqrt(2 * lam)))

        Ri[i - 1] = p  # keep track of p for each bin size i

    K[nbins] = a  # restore the center value of the cross-correlogram

    return K, Qi, Q00, Q01, Ri


# NOTE: we get a 50x time improvement with Numba jit of the existing function,
# but we could probably achieve even more by improving the implementation
@numba.jit(nopython=True, cache=False)
def _ccg_old(st1, st2, nbins, tbin):
    # this function efficiently computes the crosscorrelogram between two sets
    # of spikes (st1, st2), with tbin length each, timelags =  plus/minus nbins
    # and then estimates how refractory the cross-correlogram is, which can be used
    # during merge decisions.

    st1 = np.sort(st1)  # makes sure spike trains are sorted in increasing order
    st2 = np.sort(st2)

    dt = nbins * tbin

    # Avoid divide by zero error.
    T = max(1e-10, np.max(np.concatenate((st1, st2))) - np.min(np.concatenate((st1, st2))))
    N1 = max(1, len(st1))
    N2 = max(1, len(st2))

    # we traverse both spike trains together, keeping track of the spikes in the first
    # spike train that are within dt of spikes in the second spike train

    ilow = 0  # lower bound index
    ihigh = 0  # higher bound index
    j = 0  # index of the considered spike

    K = np.zeros(2 * nbins + 1)

    # (DEV_NOTES) the while loop below is far too slow as is

    while j <= N2 - 1:  # traverse all spikes in the second spike train

        while (ihigh <= N1 - 1) and (st1[ihigh] < st2[j] + dt):
            ihigh += 1  # keep increasing higher bound until it's OUTSIDE of dt range

        while (ilow <= N1 - 1) and (st1[ilow] <= st2[j] - dt):
            ilow += 1  # keep increasing lower bound until it's INSIDE of dt range

        if ilow > N1 - 1:
            break  # break if we exhausted the spikes from the first spike train

        if st1[ilow] > st2[j] + dt:
            # if the lower bound is actually outside of dt range, means we overshot (there were no
            # spikes in range)
            # simply move on to next spike from second spike train
            j += 1
            continue

        for k in range(ilow, ihigh):
            # for all spikes within plus/minus dt range
            ibin = np.rint((st2[j] - st1[k]) / tbin)  # convert ISI to integer
            ibin2 = np.asarray(ibin, dtype=np.int64)

            K[ibin2 + nbins] += 1

        j += 1

    irange1 = np.concatenate((np.arange(1, nbins // 2), np.arange(3 * nbins // 2, 2 * nbins)))
    irange2 = np.arange(nbins - 50, nbins - 10)
    irange3 = np.arange(nbins + 11, nbins + 50)

    # normalize the shoulders by what's expected from the mean firing rates
    # a non-refractive poisson process should yield 1

    Q00 = np.sum(K[irange1]) / (len(irange1) * tbin * N1 * N2 / T)
    # do the same for irange 2
    Q01 = np.sum(K[irange2]) / (len(irange2) * tbin * N1 * N2 / T)
    # compare to the other shoulder
    Q01 = max(Q01, np.sum(K[irange3]) / (len(irange3) * tbin * N1 * N2 / T))

    R00 = max(np.mean(K[irange2]), np.mean(K[irange3]))  # take the biggest shoulder
    R00 = max(R00, np.mean(K[irange1]))  # compare this to the asymptotic shoulder

    # test the probability that a central area in the autocorrelogram might be refractory
    # test increasingly larger areas of the central CCG

    a = K[nbins]
    K[nbins] = 0

    Qi = np.zeros(10)
    Ri = np.zeros(10)

    for i in range(1, 11):
        irange = np.arange(nbins - i, nbins + i + 1)  # for this central range of the CCG
        # compute the normalised ratio as above. this should be 1 if there is no refractoriness
        Qi0 = np.sum(K[irange]) / (2 * i * tbin * N1 * N2 / T)
        Qi[i - 1] = Qi0  # save the normalised probability

        n = np.sum(K[irange]) / 2
        lam = R00 * i
        if lam == 0:
            p = np.nan
        else:
            # NOTE: make sure lam is not zero to avoid divide by zero error
            # lam = max(1e-10, R00 * i)

            # log(p) = log(lam) * n - lam - gammaln(n+1)

            # this is tricky: we approximate the Poisson likelihood with a gaussian of equal mean
            # and variance that allows us to integrate the probability that we would see <N spikes
            # in the center of the cross-correlogram from a distribution with mean R00*i spikes

            p = 1 / 2 * (1 + erf((n - lam) / sqrt(2 * lam)))

        Ri[i - 1] = p  # keep track of p for each bin size i

    K[nbins] = a  # restore the center value of the cross-correlogram

    return K, Qi, Q00, Q01, Ri


def ccg_old(st1, st2, nbins, tbin):
    # TODO: move_to_config - There are a lot of parameters in the ccg computation that may need to
    #                      - vary for different setups.
    st1 = cp.asnumpy(st1)
    st2 = cp.asnumpy(st2)
    try:
        return _ccg_old(st1, st2, nbins, tbin)
    except ValueError:
        return 0


def ccg_metrics(st1, st2, nbins, tbin):
    """
    For two arrays of spike times, use the cross-correlogram to estimate the contamination rate
    and the statistical significance that there are fewer spikes in the refractory period
    :param st1: Array of spike times (seconds), numpy or cupy array
    :param st2: Array of spike times (seconds), numpy or cupy array
    :param nbins: Number of time bins either side, int
    :param tbin: Length of each time bin, float
    :return: contam_ratio: Proportion of refractory period violations, float
             p_value: Statistical significance of fewer spikes in the refractory period, float
    """

    K = ccg(st1, st2, nbins, tbin)

    # Indices for the tails of the ccg
    irange1 = np.concatenate((np.arange(1, nbins // 2), np.arange(3 * nbins // 2, 2 * nbins)))

    # Indices for left shoulder of the ccg
    irange2 = np.arange(nbins - 50, nbins - 10)

    # Indices for right shoulder of the ccg
    irange3 = np.arange(nbins + 11, nbins + 50)

    # Estimate the average non-refractory ccg rate by the maximum rate across these ranges
    ccg_rate = max(
        np.mean(K[irange1]),
        np.mean(K[irange2]),
        np.mean(K[irange3]),
    )

    # Set centre of CCG to 0 to avoid double-counted spikes
    K[nbins] = 0

    # test the probability that a central area in the autocorrelogram might be refractory
    # test increasingly larger areas of the central CCG

    contam_rates = np.zeros(10)
    p_values = np.zeros(10)

    for i in range(1, 11):

        irange = np.arange(nbins - i, nbins + i + 1)

        # for this central range of the CCG compute the mean CCG rate
        # as central value is set to 0, divide by 2*i
        contam_rates[i - 1] = np.sum(K[irange]) /  2*i

        n = np.sum(K[irange]) / 2
        lam = ccg_rate * i
        if lam == 0:
            p = 1
        else:
            # NOTE: make sure lam is not zero to avoid divide by zero error
            # lam = max(1e-10, R00 * i)

            # log(p) = log(lam) * n - lam - gammaln(n+1)

            # this is tricky: we approximate the Poisson likelihood with a gaussian of equal mean
            # and variance that allows us to integrate the probability that we would see <N spikes
            # in the center of the cross-correlogram from a distribution with mean R00*i spikes

            p = 1 / 2 * (1 + erf((n - lam) / sqrt(2 * lam)))

        p_values[i - 1] = p  # keep track of p for each bin size i

    # Use the central region that has lowest refractory violation rate
    p_value = np.min(p_values)
    if ccg_rate == 0:
        if np.min(contam_rates) == 0:
            contam_ratio = 0 # CCG is empty so contamination rate set to 0
        else:
            contam_ratio = 1 # Contamination rate is infinite so set to full contamination
    else:
        contam_ratio = np.min(contam_rates) / ccg_rate

    return contam_ratio, p_value


def ccg(st1, st2, nbins, tbin):
    """
    Computes the cross-correlogram for two arrays of spike times
    :param st1: Array of spike times (seconds), numpy or cupy array
    :param st2: Array of spike times (seconds), numpy or cupy array
    :param nbins: Number of time bins either side, int
    :param tbin: Length of each time bin, float
    :return: Cross-correlogram, numpy array
    """
    if (len(st1) == 0) or (len(st2) == 0):
        return np.zeros(2*nbins + 1)

    st1 = cp.asnumpy(st1)
    st2 = cp.asnumpy(st2)

    return _ccg(st1, st2, nbins, tbin)


@numba.jit(nopython=True, cache=False)
def _ccg(st1, st2, nbins, tbin):
    """ JIT compiled ccg function for speed """

    st1 = np.sort(st1)  # makes sure spike trains are sorted in increasing order
    st2 = np.sort(st2)

    dt = nbins * tbin

    # Avoid divide by zero error.
    T = max(1e-10, np.max(np.concatenate((st1, st2))) - np.min(np.concatenate((st1, st2))))
    N1 = max(1, len(st1))
    N2 = max(1, len(st2))

    # we traverse both spike trains together, keeping track of the spikes in the first
    # spike train that are within dt of spikes in the second spike train

    ilow = 0  # lower bound index
    ihigh = 0  # higher bound index
    j = 0  # index of the considered spike

    K = np.zeros(2 * nbins + 1)

    while j <= N2 - 1:  # traverse all spikes in the second spike train

        while (ihigh <= N1 - 1) and (st1[ihigh] < st2[j] + dt):
            ihigh += 1  # keep increasing higher bound until it's OUTSIDE of dt range

        while (ilow <= N1 - 1) and (st1[ilow] <= st2[j] - dt):
            ilow += 1  # keep increasing lower bound until it's INSIDE of dt range

        if ilow > N1 - 1:
            break  # break if we exhausted the spikes from the first spike train

        if st1[ilow] > st2[j] + dt:
            # if the lower bound is actually outside of dt range, means we overshot (there were no
            # spikes in range)
            # simply move on to next spike from second spike train
            j += 1
            continue

        for k in range(ilow, ihigh):
            # for all spikes within plus/minus dt range
            ibin = np.rint((st2[j] - st1[k]) / tbin)  # convert ISI to integer
            ibin2 = np.asarray(ibin, dtype=np.int64)

            K[ibin2 + nbins] += 1

        j += 1

    return K


def clusterAverage(clu, spikeQuantity):
    # get the average of some quantity across spikes in each cluster, given the
    # quantity for each spike
    #
    # e.g.
    # > clusterDepths = clusterAverage(clu, spikeDepths)
    #
    # clu and spikeQuantity must be vector, same size
    #
    # using a super-tricky algorithm for this - when you make a sparse
    # array, the values of any duplicate indices are added. So this is the
    # fastest way I know to make the sum of the entries of spikeQuantity for each of
    # the unique entries of clu
    _, cluInds, spikeCounts = np.unique(clu, return_inverse=True, return_counts=True)

    # summation
    q = coo_matrix((spikeQuantity, (cluInds, np.zeros(len(clu), dtype='int32')))) \
        .toarray().flatten()

    # had sums so dividing by spike counts gives the mean depth of each cluster
    clusterQuantity = q / spikeCounts

    return clusterQuantity


def sort_and_inverse(array):
    """
    Get sorted indices and their inverse for a Numpy array
    :param array: Numpy array
    :return: Sorted indices and their inverse
    """
    perm = np.argsort(array)
    perm_inv = np.zeros_like(perm)
    perm_inv[perm] = np.arange(len(perm))

    return perm, perm_inv


def merge_by_order(array1, array2, ordering1, ordering2, axis=0):
    """
    Concatenate two arrays along given axis according to the given ordering arrays
    This is used to concatenate spike features according to their spike times
    :param array1: Array to be concatenated, fortran order numpy array
    :param array2: Array to be concatenated, fortran order numpy array
    :param ordering1: Ordering for first array, numpy array
    :param ordering2: Ordering for second array, numpy array
    :param axis: Axis along which to concatenate
    :return: Concatenated array, fortran order numpy array
    """
    assert array1.dtype == array2.dtype
    dtype = array1.dtype

    perm1 = np.argsort(ordering1)
    perm2 = np.argsort(ordering2)

    ordering1 = ordering1[perm1]
    ordering2 = ordering2[perm2]

    n1 = array1.shape[axis]
    assert n1 == len(ordering1)
    n2 = array2.shape[axis]
    assert n2 == len(ordering2)

    # Get shape of new array
    new_shape = list(array1.shape)
    new_shape[axis] += array2.shape[axis]

    # Initialise new array
    new_array = np.zeros(new_shape, dtype=dtype, order='F')

    # Get indices for inserting first array
    indices1 = np.arange(n1) + np.searchsorted(ordering2, ordering1)

    # Indices for inserting second array
    mask = np.ones(n1 + n2, dtype='bool')
    mask[indices1] = False
    indices2 = np.where(mask)[0]

    slicer_new = [slice(None)] * new_array.ndim
    slicer_new[axis] = indices1
    slicer_old = [slice(None)] * array1.ndim
    slicer_old[axis] = perm1
    new_array[tuple(slicer_new)] = array1[tuple(slicer_old)]

    slicer_new = [slice(None)] * new_array.ndim
    slicer_new[axis] = indices2
    slicer_old = [slice(None)] * array2.ndim
    slicer_old[axis] = perm2
    new_array[tuple(slicer_new)] = array2[tuple(slicer_old)]

    return new_array


def get_spike_features(feature_path, cluster_id):
    """
    Load spike features for the cluster given by cluster_id
    :param feature_path: Path to spike features folder, Pathlib path
    :param cluster_id: ID of cluster to load
    :return: Spike features, fortran order numpy array
    """

    # Load array as memmap
    memmaped_array = memmap_large_array(feature_path / f'spike_features_{cluster_id}')

    # Convert to numpy array and return
    return np.array(memmaped_array)


def delete_spike_features(feature_path, cluster_id):
    """
    Delete spike features for the cluster given by cluster_id
    :param feature_path: Path to spike features folder, Pathlib path
    :param cluster_id: ID of cluster to delete
    """

    os.remove(feature_path / f'spike_features_{cluster_id}')
    os.remove(feature_path / f'spike_features_{cluster_id}.json')


def write_spike_features(feature_path, cluster_id, dtype, shape, array=None):
    """
    Write array to the spike features on disk for the cluster given by cluster_id
    :param feature_path: Path to spike features
    :param cluster_id: Cluster ID to write to
    :param dtype: Array dtype
    :param shape: Shape of array for array writer
    :param array: Array to write, fortran order numpy array
    """
    writer = LargeArrayWriter(
        feature_path / f'spike_features_{cluster_id}',
        dtype = dtype,
        shape = shape,
        )
    if array is not None:
        writer.append(array)
    writer.close()


def merge_spike_features(feature_path, cluster_1, times_1, cluster_2, times_2, array_shape):
    """
    Merge the spike features for two clusters according to their times, save the new features under
     cluster_1 and delete the old features for cluster_2
    :param feature_path: Path to folder of spike features
    :param cluster_1: Cluster ID of first cluster that will have all the spikes after merging
    :param times_1: Spike times of first cluster
    :param cluster_2: Cluster ID of second cluster that will be empty after merging
    :param times_2: Spike times of second cluster
    :param array_shape: Shape parameter to pass to the array writer
    :return:
    """
    feature_path = Path(feature_path)

    # Load spike features for the clusters
    features_1 = get_spike_features(feature_path, cluster_1)
    features_2 = get_spike_features(feature_path, cluster_2)

    # Combine to get new features with spikes ordered according to their times
    new_features = merge_by_order(features_1,features_2,times_1, times_2, axis=2)

    # Delete the old spike feature files
    delete_spike_features(feature_path, cluster_1)
    delete_spike_features(feature_path, cluster_2)

    # Save new features under cluster_1
    write_spike_features(feature_path, cluster_1, new_features.dtype, array_shape, new_features)

    # Save empty array under cluster_2
    write_spike_features(feature_path, cluster_2, new_features.dtype, array_shape)


def split_features(feature_path, original_cluster, new_cluster, indices, array_shape):
    """
    Split the spike features on disk, spikes in indices are re-assigned from old_cluster to
    new_cluster
    :param feature_path: Path to spike features, Pathlib path
    :param original_cluster: Cluster ID of original cluster, int
    :param new_cluster: Cluster ID of new cluster, int
    :param indices: Spikes to re-assign to new cluster, boolean numpy array
    :param array_shape: Array shape argument to pass to spike feature writer
    """

    feature_path = Path(feature_path)

    # Load spike features of original cluster
    old_features = get_spike_features(feature_path, original_cluster)
    assert old_features.shape[2] == len(indices)

    # New features based on split
    features1 = old_features[:,:,~indices]
    features2 = old_features[:,:,indices]

    delete_spike_features(feature_path, original_cluster)
    assert f'spike_features_{new_cluster}' not in os.listdir(feature_path)

    # Save new features
    write_spike_features(feature_path, original_cluster, old_features.dtype, array_shape,
                         features1)
    write_spike_features(feature_path, new_cluster, old_features.dtype, array_shape,
                         features2)


def find_merges(ctx):
    # this function merges clusters based on template correlation
    # however, a merge is veto-ed if refractory period violations are introduced

    params = ctx.params
    ir = ctx.intermediate

    # TODO: move_to_config
    dt = 1. / 1000  # step size for CCG binning
    nbins = 500  # number of bins used for cross-correlograms
    NchanNear = min(ctx.probe.Nchan, 32)
    Nrank = 3

    # (DEV_NOTES) nbins is not a variable in Marius' code, I include it here to avoid
    # unexplainable, hard-coded constants later

    st3 = ir.st3
    Xsim = ir.simScore  # this is the pairwise similarity score
    Nk = Xsim.shape[0]
    Xsim = Xsim - np.diag(np.diag(Xsim))

    # sort by firing rate first
    nspk = np.zeros(Nk)
    for j in range(Nk):
        # determine total number of spikes in each neuron
        nspk[j] = np.sum(st3[:, 1] == j)

    # we traverse the set of neurons in ascending order of firing rates
    isort = np.argsort(nspk)

    logger.debug('Initialized spike counts.')

    if params.low_memory:
        feature_path = ctx.context_path / 'spike_features'

    for j in tqdm(range(Nk), desc='Finding merges'):
        # find all spikes from this cluster
        s1 = st3[:, 0][st3[:, 1] == isort[j]] / params.fs

        if s1.size != nspk[isort[j]]:
            # this is a check to make sure new clusters are combined correctly into bigger clusters
            # TODO: unclear - don't we want to bail in that case?
            logger.warn('Lost track of spike counts.')

        # sort all the pairs of this neuron, discarding any that have fewer spikes

        uu = Xsim[isort[j], :] * (nspk > s1.size)
        ix = np.argsort(uu)[::-1]
        ccsort = uu[ix]
        ienu = int(np.nonzero(ccsort < .5)[0][0])

        # ccsort = -cp.sort(-Xsim[isort[j]] * (nspk > len(s1)))  # sort in descending order
        # ix = cp.argsort(-Xsim[isort[j]] * (nspk > len(s1)))

        # if ccsort[len(ccsort) - 1] > 0.5:
        #     ienu = len(ccsort)
        # else:
        #     ienu = cp.argmax(ccsort < 0.5)

        # for all pairs above 0.5 correlation

        for k in range(ienu):
            # find the spikes of the pair
            s2 = st3[:, 0][st3[:, 1] == ix[k]] / params.fs

            # Compute contamination ratio and probability the center is refractory from the
            # cross-correlogram. The p_value kicks in when there are very few spikes
            contam_ratio, p_value = ccg_metrics(s1, s2, nbins, dt)

            # TODO: move_to_config
            if (contam_ratio < 0.2) and (p_value < 0.05):  # if both refractory criteria are met
                i = ix[k]
                # now merge j into i and move on
                # simply overwrite all the spikes of neuron j with i (i>j by construction)
                st3[:, 1][st3[:, 1] == isort[j]] = i
                nspk[i] = nspk[i] + nspk[isort[j]]  # update number of spikes for cluster i
                # Update spike features
                if params.low_memory:
                    merge_spike_features(feature_path, int(i), s2, int(isort[j]), s1,
                                         array_shape=(NchanNear, Nrank, -1))
                logger.debug(f'Merged {isort[j]} into {i}')
                # TODO: unclear - the comment below looks important :)
                # YOU REALLY SHOULD MAKE SURE THE PC CHANNELS MATCH HERE
                # break % if a pair is found, we don't need to keep going
                # (we'll revisit this cluster when we get to the merged cluster)
                break

    ctx.save(st3=st3)

    if params.save_temp_files:
        np.save(ctx.context_path / 'temp_splits' / 'st3_merge.npy', st3)


def splitAllClusters(ctx, flag):
    # I call this algorithm "bimodal pursuit"
    # split clusters if they have bimodal projections
    # the strategy is to maximize a bimodality score and find a single vector projection
    # that maximizes it. If the distribution along that maximal projection crosses a
    # bimodality threshold, then the cluster is split along that direction
    # it only uses the PC features for each spike, stored in ir.cProjPC

    params = ctx.params
    probe = ctx.probe
    ir = ctx.intermediate
    Nchan = ctx.probe.Nchan

    wPCA = cp.asarray(ir.wPCA)  # use PCA projections to reconstruct templates when we do splits
    assert wPCA.shape[1] == 3

    # Take intermediate arrays from context.
    st3 = cp.asnumpy(ir.st3)
    cProjPC = ir.cProjPC
    dWU = ir.dWU

    # For the following arrays that will be overwritten by this function, try to get
    # it from a previous call to this function (as it is called twice), otherwise
    # get it from before (without the _s suffix).
    W = ir.get('W_s', ir.W)
    simScore = ir.get('simScore_s', ir.simScore)
    iNeigh = ir.get('iNeigh_s', ir.iNeigh)
    iNeighPC = ir.get('iNeighPC_s', ir.iNeighPC)

    # this is the threshold for splits, and is one of the main parameters users can change
    ccsplit = params.AUCsplit

    Nrank = 3
    NchanNear = min(Nchan, 32)
    Nnearest = min(Nchan, 32)
    sigmaMask = params.sigmaMask

    ik = -1
    Nfilt = W.shape[1]
    nsplits = 0

    # determine what channels each template lives on
    iC, mask, C2C = getClosestChannels(probe, sigmaMask, NchanNear)

    # the waveforms must be aligned to this sample
    nt0min = params.nt0min
    # find the peak abs channel for each template
    iW = np.argmax(np.abs((dWU[nt0min - 1, :, :])), axis=0)

    # keep track of original cluster for each cluster. starts with all clusters being their
    # own origin.
    isplit = np.arange(Nfilt)
    dt = 1. / 1000
    nccg = 0

    if params.low_memory:
        feature_path = ctx.context_path / 'spike_features'

    while ik < Nfilt:
        if ik % 100 == 0:
            # periodically write updates
            logger.info(f'Found {nsplits} splits, checked {ik}/{Nfilt} clusters, nccg {nccg}')
        ik += 1

        isp = (st3[:, 1] == ik)  # get all spikes from this cluster
        nSpikes = isp.sum()
        logger.debug(f"Splitting template {ik}/{Nfilt} with {nSpikes} spikes.")
        free_gpu_memory()

        if nSpikes < 300:
            # TODO: move_to_config
            # do not split if fewer than 300 spikes (we cannot estimate
            # cross-correlograms accurately)
            continue

        ss = st3[isp, 0] / params.fs  # convert to seconds

        # get the PC projections for these spikes
        if params.low_memory:
            clp0 = memmap_large_array(feature_path / f'spike_features_{ik}').T
        else:
            clp0 = cProjPC[isp, :, :]

        clp0 = cp.asarray(clp0, dtype=cp.float32)  # upload to the GPU
        clp0 = clp0.reshape((clp0.shape[0], -1), order='F')
        clp = clp0 - mean(clp0, axis=0) # mean center them

        isp = np.nonzero(isp)[0]

        # subtract a running average, because the projections are NOT drift corrected
        clp = clp - my_conv2(clp, 250, 0)

        # now use two different ways to initialize the bimodal direction
        # the main script calls this function twice, and does both initializations

        if flag:
            u, s, v = svdecon(clp.T)
            #    u, v = -u, -v  # change sign for consistency with MATLAB
            w = u[:, 0]  # initialize with the top PC
        else:
            w = mean(clp0, axis=0)  # initialize with the mean of NOT drift-corrected trace
            w = w / cp.sum(w ** 2) ** 0.5  # unit-normalize

        # initial projections of waveform PCs onto 1D vector
        x = cp.dot(clp, w)
        s1 = var(x[x > mean(x)])  # initialize estimates of variance for the first
        s2 = var(x[x < mean(x)])  # and second gaussian in the mixture of 1D gaussians

        mu1 = mean(x[x > mean(x)])  # initialize the means as well
        mu2 = mean(x[x < mean(x)])
        # and the probability that a spike is assigned to the first Gaussian
        p = mean(x > mean(x))

        # initialize matrix of log probabilities that each spike is assigned to the first
        # or second cluster
        logp = cp.zeros((nSpikes, 2), order='F')

        # do 50 pursuit iteration

        logP = cp.zeros(50)  # used to monitor the cost function

        # TODO: move_to_config - maybe...
        for k in range(50):
            # for each spike, estimate its probability to come from either Gaussian cluster
            logp[:, 0] = -1. / 2 * log(s1) - ((x - mu1) ** 2) / (2 * s1) + log(p)
            logp[:, 1] = -1. / 2 * log(s2) - ((x - mu2) ** 2) / (2 * s2) + log(1 - p)

            lMax = logp.max(axis=1)
            logp = logp - lMax[:, cp.newaxis]  # subtract the max for floating point accuracy
            rs = cp.exp(logp)  # exponentiate the probabilities

            pval = cp.log(cp.sum(rs, axis=1)) + lMax  # get the normalizer and add back the max
            logP[k] = mean(pval)  # this is the cost function: we can monitor its increase

            rs = rs / cp.sum(rs, axis=1)[:, cp.newaxis]  # normalize so that probabilities sum to 1

            p = mean(rs[:, 0])  # mean probability to be assigned to Gaussian 1
            # new estimate of mean of cluster 1 (weighted by "responsibilities")
            mu1 = cp.dot(rs[:, 0], x) / cp.sum(rs[:, 0])
            # new estimate of mean of cluster 2 (weighted by "responsibilities")
            mu2 = cp.dot(rs[:, 1], x) / cp.sum(rs[:, 1])

            s1 = cp.dot(rs[:, 0], (x - mu1) ** 2) / cp.sum(rs[:, 0])  # new estimates of variances
            s2 = cp.dot(rs[:, 1], (x - mu2) ** 2) / cp.sum(rs[:, 1])

            if (k >= 10) and (k % 2 == 0):
                # starting at iteration 10, we start re-estimating the pursuit direction
                # that is, given the Gaussian cluster assignments, and the mean and variances,
                # we re-estimate w
                # these equations follow from the model
                StS = cp.matmul(
                    clp.T, clp * (rs[:, 0] / s1 + rs[:, 1] / s2)[:, cp.newaxis]) / nSpikes
                StMu = cp.dot(clp.T, rs[:, 0] * mu1 / s1 + rs[:, 1] * mu2 / s2) / nSpikes

                # this is the new estimate of the best pursuit direction
                w = cp.linalg.solve(StS.T, StMu)
                w = w / cp.sum(w ** 2) ** 0.5  # which we unit normalize
                x = cp.dot(clp, w)

        # these spikes are assigned to cluster 1
        ilow = rs[:, 0] > rs[:, 1]
        # the mean probability of spikes assigned to cluster 1
        plow = mean(rs[:, 0][ilow])
        phigh = mean(rs[:, 1][~ilow])  # same for cluster 2
        # the smallest cluster has this proportion of all spikes
        nremove = min(mean(ilow), mean(~ilow))

        # did this split fix the autocorrelograms?
        # compute the cross-correlogram between spikes in the putative new clusters
        ilow_cpu = cp.asnumpy(ilow)
        contam_ratio, p_value = ccg_metrics(ss[ilow_cpu], ss[~ilow_cpu], 500, dt)

        # if the CCG has a dip, don't do the split.
        # These thresholds are consistent with the ones from merges.
        # TODO: move_to_config (or at least a single constant so the are the same as the merges)
        if (contam_ratio < 0.25) and (p_value < 0.05):  # if both metrics are below threshold.
            nccg += 1  # keep track of how many splits were voided by the CCG criterion
            continue

        # now decide if the split would result in waveforms that are too similar
        # the reconstructed mean waveforms for putative cluster 1
        # c1 = cp.matmul(wPCA, cp.reshape((mean(clp0[ilow, :], 0), 3, -1), order='F'))
        c1 = cp.matmul(wPCA, mean(clp0[ilow, :], 0).reshape((3, -1), order='F'))
        # the reconstructed mean waveforms for putative cluster 2
        # c2 = cp.matmul(wPCA, cp.reshape((mean(clp0[~ilow, :], 0), 3, -1), order='F'))
        c2 = cp.matmul(wPCA, mean(clp0[~ilow, :], 0).reshape((3, -1), order='F'))

        cc = cp.corrcoef(c1.ravel(), c2.ravel())  # correlation of mean waveforms
        n1 = sqrt(cp.sum(c1 ** 2))  # the amplitude estimate 1
        n2 = sqrt(cp.sum(c2 ** 2))  # the amplitude estimate 2

        r0 = 2 * abs((n1 - n2) / (n1 + n2))

        # if the templates are correlated, and their amplitudes are similar, stop the split!!!

        # TODO: move_to_config
        if (cc[0, 1] > 0.9) and (r0 < 0.2):
            continue

        # finaly criteria to continue with the split: if the split piece is more than 5% of all
        # spikes, if the split piece is more than 300 spikes, and if the confidences for
        # assigning spikes to # both clusters exceeds a preset criterion ccsplit
        # TODO: move_to_config
        if (nremove > 0.05) and (min(plow, phigh) > ccsplit) and (
                min(cp.sum(ilow), cp.sum(~ilow)) > 300):
            # one cluster stays, one goes
            Nfilt += 1

            # the templates for the splits have been estimated from PC coefficients

            # (DEV_NOTES) code below involves multiple CuPy arrays changing shape to accomodate
            # the extra cluster, this could potentially be done more efficiently?

            dWU = cp.concatenate((
                cp.asarray(dWU), cp.zeros((*dWU.shape[:-1], 1), order='F')), axis=2)
            dWU[:, iC[:, iW[ik]], Nfilt - 1] = c2
            dWU[:, iC[:, iW[ik]], ik] = c1

            # the temporal components are therefore just the PC waveforms
            W = cp.asarray(W)
            W = cp.concatenate((W, cp.transpose(cp.atleast_3d(wPCA), (0, 2, 1))), axis=1)
            assert W.shape[1] == Nfilt

            # copy the best channel from the original template
            iW = cp.asarray(iW)
            iW = cp.pad(iW, (0, (Nfilt - len(iW))), mode='constant')
            iW[Nfilt - 1] = iW[ik]
            assert iW.shape[0] == Nfilt

            # copy the provenance index to keep track of splits
            isplit = cp.asarray(isplit)
            isplit = cp.pad(isplit, (0, (Nfilt - len(isplit))), mode='constant')
            isplit[Nfilt - 1] = isplit[ik]
            assert isplit.shape[0] == Nfilt

            st3[isp[ilow_cpu], 1] = Nfilt - 1  # overwrite spike indices with the new index

            # copy similarity scores from the original
            simScore = cp.asarray(simScore)
            simScore = cp.pad(
                simScore, (0, (Nfilt - simScore.shape[0])), mode='constant')
            simScore[:, Nfilt - 1] = simScore[:, ik]
            simScore[Nfilt - 1, :] = simScore[ik, :]
            # copy similarity scores from the original
            simScore[ik, Nfilt - 1] = 1  # set the similarity with original to 1
            simScore[Nfilt - 1, ik] = 1  # set the similarity with original to 1
            assert simScore.shape == (Nfilt, Nfilt)

            # copy neighbor template list from the original
            iNeigh = cp.asarray(iNeigh)
            iNeigh = cp.pad(
                iNeigh, ((0, 0), (0, (Nfilt - iNeigh.shape[1]))), mode='constant')
            iNeigh[:, Nfilt - 1] = iNeigh[:, ik]
            assert iNeigh.shape[1] == Nfilt

            # copy neighbor channel list from the original
            iNeighPC = cp.asarray(iNeighPC)
            iNeighPC = cp.pad(
                iNeighPC, ((0, 0), (0, (Nfilt - iNeighPC.shape[1]))), mode='constant')
            iNeighPC[:, Nfilt - 1] = iNeighPC[:, ik]
            assert iNeighPC.shape[1] == Nfilt

            if params.low_memory:
                # change spike features on disk to reflect the split
                split_features(feature_path, int(ik), int(Nfilt - 1),
                               ilow_cpu, (NchanNear, Nrank, -1))

            # try this cluster again
            # the cluster piece that stays at this index needs to be tested for splits again
            # before proceeding
            ik -= 1
            # the piece that became a new cluster will be tested again when we get to the end
            # of the list
            nsplits += 1  # keep track of how many splits we did
    #         pbar.update(ik)
    # pbar.close()

    logger.info(
        f'Finished splitting. Found {nsplits} splits, checked '
        f'{ik}/{Nfilt} clusters, nccg {nccg}')

    Nfilt = W.shape[1]  # new number of templates
    Nrank = 3
    Nchan = probe.Nchan
    Params = cp.array(
        [0, Nfilt, 0, 0, W.shape[0], Nnearest, Nrank, 0, 0, Nchan, NchanNear, nt0min, 0],
        dtype=cp.float64)  # make a new Params to pass on parameters to CUDA

    # we need to re-estimate the spatial profiles

    # we get the time upsampling kernels again
    Ka, Kb = getKernels(params)
    # we run SVD
    W, U, mu = mexSVDsmall2(Params, dWU, W, iC, iW, Ka, Kb)

    # we re-compute similarity scores between templates
    WtW, iList = getMeWtW(W.astype(cp.float32), U.astype(cp.float32), Nnearest)
    # ir.iList = iList  # over-write the list of nearest templates

    isplit = simScore == 1  # overwrite the similarity scores of clusters with same parent
    simScore = WtW.max(axis=2)
    simScore[isplit] = 1  # 1 means they come from the same parent

    iNeigh = iList[:, :Nfilt]  # get the new neighbor templates
    iNeighPC = iC[:, iW[:Nfilt]]  # get the new neighbor channels

    # for Phy, we need to pad the spikes with zeros so the spikes are aligned to the center of
    # the window
    Wphy = cp.concatenate(
        (cp.zeros((1 + nt0min, Nfilt, Nrank), order='F'), W), axis=0)

    # ir.isplit = isplit  # keep track of origins for each cluster

    if params.save_temp_files:
        np.save(ctx.context_path / 'temp_splits' / 'st3_split.npy', cp.asnumpy(st3))

    #TODO:
    # st3, W, U, mu, simScore, iNeigh, iNeighPC overwrites
    # isplit overwrites or saves
    # Wphy saves
    # iList is repeated and should be deleted (need to check this)

    ctx.save(
        st3=st3,

        W_s=cp.asnumpy(W),
        U_s=cp.asnumpy(U),
        mu_s=cp.asnumpy(mu),
        simScore_s=cp.asnumpy(simScore),
        iNeigh_s=cp.asnumpy(iNeigh),
        iNeighPC_s=cp.asnumpy(iNeighPC),

        Wphy=cp.asnumpy(Wphy),
        iList=cp.asnumpy(iList),
        isplit=cp.asnumpy(isplit),
    )

    if params.save_temp_files:
        ctx.write(st3_s=st3)


def set_cutoff(ctx):
    # after everything else is done, this function takes spike trains and cuts off
    # any noise they might have picked up at low amplitude values
    # We look for bimodality in the amplitude plot, thus setting an individual threshold
    # for each neuron.
    # Also, this function calls "good" and "bad" clusters based on the auto-correlogram

    ir = ctx.intermediate
    params = ctx.params

    st3 = ir.st3  # st3_s1 is saved by the first splitting step
    # cProj = ir.cProj
    # cProjPC = ir.cProjPC

    nbins = 500     # no of bins for CCG
    dt = 1. / 1000  # step size for CCG binning

    Nk = ir.Wphy.shape[1]  # number of templates

    # sort by firing rate first
    good = np.zeros(Nk)
    Ths = np.zeros(Nk)
    est_contam_rate = np.zeros(Nk)

    for j in tqdm(range(Nk), desc='Setting cutoff'):
        ix = np.where(st3[:, 1] == j)[0]  # find all spikes from this neuron
        ss = st3[ix, 0] / params.fs  # convert to seconds
        if ss.size == 0:
            continue  # break if there are no spikes

        vexp = st3[ix, 3]  # vexp is the relative residual variance of the spikes

        Th = params.Th[0]  # start with a high threshold

        fcontamination = 0.1  # acceptable contamination rate
        est_contam_rate[j] = 1

        while Th >= params.Th[1]:
            # continually lower the threshold, while the estimated unit contamination is low
            st = ss[vexp > Th]  # take spikes above the current threshold
            if len(st) == 0:
                Th -= 0.5  # if there are no spikes, we need to keep lowering the threshold
                continue

            # compute the refractory violation metrics using the cross-corellogram
            contam_ratio, p_value = ccg_metrics(st, st, nbins, dt)

            # if the unit is already contaminated, we break, and use the next higher threshold
            if (contam_ratio > fcontamination) or (p_value > 0.05):
                break
            else:
                if (Th == params.Th[0]) and (contam_ratio < 0.05):
                    # only on the first iteration, we consider if the unit starts well isolated
                    # if it does, then we put much stricter criteria for isolation
                    # to make sure we don't settle for a relatively high contamination unit
                    fcontamination = min(0.05, max(0.01, contam_ratio * 2))

                    # if the unit starts out contaminated, we will settle with the higher
                    # contamination rate

                # this unit is good, because we will stop lowering the threshold when it
                # becomes bad
                good[j] = 1
                Th -= 0.5

        # we exited the loop because the contamination was too high. We revert to the higher
        # threshold
        Th += 0.5
        st = ss[vexp > Th]  # take spikes above the current threshold

        # compute the refractory violation metrics using the cross-corellogram
        contam_ratio, p_value = ccg_metrics(st, st, nbins, dt)

        est_contam_rate[j] = contam_ratio  # this score will be displayed in Phy

        Ths[j] = Th  # store the threshold for potential debugging

        # any spikes below the threshold get discarded into a 0-th cluster
        st3[ix[vexp <= Th], 1] = -1

    # we sometimes get NaNs, why? replace with full contamination
    # (DEV_NOTES) this seems to occur when both Qi and max(Q00, Q01) are zero thus when dividing
    # the two to get Q the result is a NaN

    est_contam_rate[np.isnan(est_contam_rate)] = 1

    # remove spikes assigned to the -1 cluster
    ix = st3[:, 1] == -1
    st3 = st3[~ix, :]

    # NOTE: to avoid loading everything into memory, we don't export cProj and cProjPC
    # right now, we'll remove the spikes at the rezToPhy stage.

    # if len(cProj) > 0:
    #     ix = cp.asnumpy(ix)
    #     if len(ix) > cProj.shape[0]:
    #         ix = ix[:cProj.shape[0]]
    #     else:
    #         ix = np.pad(ix, (0, cProj.shape[0] - len(ix)), mode='constant')
    #     assert ix.shape[0] == cProj.shape[0] == cProjPC.shape[0]
    #     cProj = cProj[~ix, :]  # remove their template projections too
    #     cProjPC = cProjPC[~ix, :, :]  # and their PC projections
    #     assert st3.shape[0] == cProj.shape[0] == cProjPC.shape[0]

    ctx.save(
        st3=st3,  # the spikes assigned to -1 have been removed here
        spikes_to_remove=ix,
        # cProj_c=cProj,
        # cProjPC_c=cProjPC,

        est_contam_rate=est_contam_rate,
        Ths=Ths,
        good=good,
    )

    if params.save_temp_files:
        np.save(ctx.context_path / 'temp_splits' / 'st3_cutoff.npy', st3)


def checkClusters(ctx):
    # Checks integrity of clusters. Removes clusters with 0 spikes.

    # 1) List all cluster ids for spikes.
    # 2) Find missing ids
    # 3) Remove these indices from every variable that has n_clusters

    ir = ctx.intermediate
    n_templates = ir.Wphy.shape[1]
    ids = cp.asnumpy(np.unique(ir.st3[:, 1]).astype(np.int))
    # Check if the max cluster id is equal to the number of cluster ids assigned to spikes.
    if n_templates != len(ids):  # see which cluster ids are missing
        good_units_mask = np.isin(np.arange(n_templates), ids)
        # Remove clusters from fields in `ir` based on `good_units_mask`
        # ir.dWU = ir.dWU[:, :, good_units_mask]
        ir.iNeigh_s = ir.iNeigh_s[:, good_units_mask]
        ir.iNeighPC_s = ir.iNeighPC_s[:, good_units_mask]
        ir.mu_s = ir.mu_s[good_units_mask]
        ir.simScore_s = ir.simScore_s[good_units_mask][:, good_units_mask]
        ir.U_s = ir.U_s[:, good_units_mask, :]
        # ir.UA = ir.UA[:, good_units_mask, :, :]
        # ir.U_a = ir.U_a[:, :, good_units_mask]
        # ir.U_b = ir.U_b[:, :, good_units_mask]
        ir.W_s = ir.W_s[:, good_units_mask, :]
        # ir.WA = ir.WA[:, good_units_mask, :, :]
        # ir.W_a = ir.W_a[:, :, good_units_mask]
        # ir.W_b = ir.W_b[:, :, good_units_mask]
        ir.Wphy = ir.Wphy[:, good_units_mask, :]
        ir.iList = ir.iList[:, good_units_mask]
        ir.isplit = ir.isplit[good_units_mask][:, good_units_mask]
        ir.est_contam_rate = ir.est_contam_rate[good_units_mask]
        ir.Ths = ir.Ths[good_units_mask]
        ir.good = ir.good[good_units_mask]

        # Find empty cluster ids, and for spikes with cluster ids above those indices, subtract 1.
        empty_cl = np.nonzero(~good_units_mask)[0]
        for cl in empty_cl[::-1]:
            logger.debug("Removing empty cluster %d.", cl)
            mislabeled_cl = np.where(ir.st3[:, 1] > cl)[0]
            ir.st3[mislabeled_cl, 1] -= 1

    ctx.ir = ir
    return ctx


# TODO: design - let's split this out into a different module and a class / a few functions
def rezToPhy(ctx, dat_path=None, output_dir=None):
    # pull out results from kilosort's rez to either return to workspace or to
    # save in the appropriate format for the phy GUI to run on. If you provide
    # a output_dir it should be a folder

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    ctx = checkClusters(ctx)  # check clusters integrity

    probe = ctx.probe
    ir = ctx.intermediate
    params = ctx.params

    # spike_times will be in samples, not seconds
    W = ir.Wphy.astype(np.float32)
    Wrot = ir.Wrot
    est_contam_rate = ir.est_contam_rate
    good = ir.good
    Ths = ir.Ths

    st3 = ir.st3

    U = ir.U_s.astype(np.float32)
    iNeigh = ir.iNeigh_s
    iNeighPC = ir.iNeighPC_s
    simScore = ir.simScore_s

    if st3.shape[1] > 4:
        st3 = st3[:, :4]

    isort = np.argsort(st3[:, 0])
    st3 = st3[isort, :]
    # cProj = ir.cProj_c[cp.asnumpy(isort), :]
    # cProjPC = ir.cProjPC_c[cp.asnumpy(isort), :, :]

    fs = os.listdir(output_dir)
    for file in fs:
        if file.endswith('.npy'):
            os.remove(output_dir / file)
    if (output_dir / '.phy').is_dir():
        shutil.rmtree(output_dir / '.phy')

    spike_times = st3[:, 0].astype(np.uint64)
    spike_templates = st3[:, 1].astype(np.uint32)

    # If multiple datasets were run, output the original dataset each spike came from as well as
    # the spike time within the dataset
    if ctx.raw_data.multiple_datasets:
        dataset_times = ctx.raw_data.n_samples
        spike_datasets = np.searchsorted(dataset_times[1:], spike_times, side='right')
        spike_times_corrected = spike_times - dataset_times[spike_datasets]

    # templateFeatures = cProj
    template_feature_inds = iNeigh.astype(np.uint32)
    # pcFeatures = cProjPC
    pc_feature_inds = iNeighPC.astype(np.uint32)

    whiteningMatrix = cp.asnumpy(Wrot) / params.scaleproc
    whiteningMatrixInv = np.linalg.pinv(whiteningMatrix)

    amplitudes = st3[:, 2]

    Nchan = probe.Nchan

    xcoords = probe.xc
    ycoords = probe.yc
    chanMap = probe.chanMap
    chanMap0ind = chanMap  # - 1

    nt0, Nfilt = W.shape[:2]

    # (DEV_NOTES) 2 lines below can be combined
    # templates = cp.einsum('ikl,jkl->ijk', U, W).astype(cp.float32)
    # templates = cp.zeros((Nchan, nt0, Nfilt), dtype=np.float32, order='F')
    temp_amps_unscaled = np.zeros(Nfilt, dtype=np.float32)
    templates_writer = NpyWriter(join(output_dir, 'templates.npy'), (Nfilt, nt0, Nchan), np.float32)
    for iNN in tqdm(range(Nfilt), desc="Computing templates"):
        t = np.dot(U[:, iNN, :], W[:, iNN, :].T).T
        templates_writer.append(t)
        t_unw = np.dot(t, whiteningMatrixInv)
        assert t_unw.ndim == 2
        temp_chan_amps = t_unw.max(axis=0) - t_unw.min(axis=0)
        temp_amps_unscaled[iNN] = temp_chan_amps.max()

    templates_writer.close()
    # templates = cp.transpose(templates, (2, 1, 0))  # now it's nTemplates x nSamples x nChannels
    # we include all channels so this is trivial
    templatesInds = np.tile(np.arange(Nfilt), (Nchan, 1))

    # here we compute the amplitude of every template...

    # unwhiten all the templates
    # tempsUnW = cp.einsum('ijk,kl->ijl', templates, whiteningMatrixinv)
    # tempsUnW = cp.zeros(templates.shape, dtype=np.float32, order='F')
    # for t in tqdm(range(templates.shape[0]), desc="Unwhitening the templates"):
    #     tempsUnW[t, :, :] = cp.dot(templates[t, :, :], whiteningMatrixInv)

    # The amplitude on each channel is the positive peak minus the negative
    # temp_chan_amps = tempsUnW.max(axis=1) - tempsUnW.min(axis=1)

    # The template amplitude is the amplitude of its largest channel
    # temp_amps_unscaled = temp_chan_amps.max(axis=1)

    # assign all spikes the amplitude of their template multiplied by their
    # scaling amplitudes
    # temp_amps_unscaled = cp.(temp_amps_unscaled, axis=0).astype(np.float32)
    spike_amps = temp_amps_unscaled[spike_templates] * amplitudes

    # take the average of all spike amps to get actual template amps (since
    # tempScalingAmps are equal mean for all templates)
    ta = clusterAverage(spike_templates, spike_amps)
    tids = np.unique(spike_templates).astype(np.int64)
    temp_amps = np.zeros_like(temp_amps_unscaled, order='F')
    temp_amps[tids] = ta  # because ta only has entries for templates that had at least one spike
    temp_amps = params.gain * temp_amps  # for consistency, make first dimension template number

    # PCs
    ix = ir.spikes_to_remove  # length: number of spikes BEFORE -1 cluster removed

    cProj_shape = ir.cProj.shape
    cProj_shape = (st3.shape[0],) + cProj_shape[1:]

    cProjPC_shape = ir.cProjPC.shape
    cProjPC_shape = (st3.shape[0],) + cProjPC_shape[1:]

    #TODO:
    tfw = NpyWriter(output_dir / 'template_features.npy', cProj_shape, np.float32)
    pcw = NpyWriter(output_dir / 'pc_features.npy', cProjPC_shape, np.float32)

    #isort = cp.asnumpy(isort)
    N = len(ix)  # number of spikes including those assigned to -1
    assert ir.cProj.shape[0] == N
    assert ir.cProjPC.shape[0] == N

    spikes_to_keep = np.nonzero(~ix)[0]  # indices of the spikes to keep in the cProj index space

    # if len(ix) > ir.cProj.shape[0]:
    #     ix = ix[:cProj.shape[0]]
    # else:
    #     ix = np.pad(ix, (0, ir.cProj.shape[0] - len(ix)), mode='constant')
    # assert ix.shape[0] == ir.cProj.shape[0] == ir.cProjPC.shape[0]

    k = int(ceil(float(N) / 100))  # 100 chunks
    assert k >= 1
    for i in tqdm(range(0, N, 100000), desc="Saving template and PC features"):
        # NOTE: cProj and cProjPC still have the spikes assigned to -1 that have yet to be removed

        # spike indices in cProj that need to be kept in this chunk
        ind = spikes_to_keep[isort[i:i + 100000]]

        cProj = ir.cProj[ind]
        cProjPC = ir.cProjPC[ind]

        tfw.append(cProj)
        pcw.append(cProjPC)
    tfw.close()
    pcw.close()
    # with open(, 'wb') as fp:
    #     save_large_array(fp, templateFeatures)
    # cProj = ir.cProj_c[cp.asnumpy(isort), :]
    # cProjPC = ir.cProjPC_c[cp.asnumpy(isort), :, :]

    def _save(name, arr, dtype=None):
        cp.save(output_dir / f'{name}.npy', arr.astype(dtype or arr.dtype))

    if output_dir is not None:
        # units um, dimension (ntimes, ndepths)
        if params.perform_drift_registration:
            _save('drift.um', ir.dshift)
            # units um, dimension (1, ndepths)
            _save('drift_depths.um', ir.yblk[np.newaxis, :])
            batch_size = params.NT / params.fs
            # units secs, dimension (ntimes,)
            _save('drift.times',
                  np.arange(ir.dshift.shape[0]) * batch_size + batch_size / 2)
        _save('spike_times', spike_times)

        # Save two copies as spike_clusters gets modified by Phy during manual curation
        _save('spike_templates', spike_templates, np.uint32)
        _save('spike_clusters', spike_templates, np.uint32)

        if ctx.raw_data.multiple_datasets:
            _save('spike_datasets', spike_datasets)
            _save('spike_times_corrected', spike_times_corrected)

        _save('amplitudes', amplitudes)
        # _save('templates', templates)
        _save('templates_ind', templatesInds)

        chanMap0ind = chanMap0ind.astype(np.int32)

        _save('channel_map', chanMap0ind)
        _save('channel_positions', np.c_[xcoords, ycoords], np.float32)

        # _save('template_features', templateFeatures)
        # with open(join(output_dir, 'template_features.npy'), 'wb') as fp:
        #     save_large_array(fp, templateFeatures)
        _save('template_feature_ind', template_feature_inds.T)

        # _save('pc_features', pcFeatures)
        # with open(join(output_dir, 'pc_features.npy'), 'wb') as fp:
        #     save_large_array(fp, pcFeatures)
        _save('pc_feature_ind', pc_feature_inds.T)

        _save('spike_pc_components', ir.wPCA)

        _save('whitening_mat', whiteningMatrix)
        _save('whitening_mat_inv', whiteningMatrixInv)

        _save('thresholds', Ths)

        if 'simScore' in ir:
            similarTemplates = simScore
            _save('similar_templates', similarTemplates)

        est_contam_rate[np.isnan(est_contam_rate)] = 1
        with open(join(output_dir, 'cluster_KSLabel.tsv'), 'w') as f:
            f.write('cluster_id\tKSLabel\n')
            for j in range(len(good)):
                if good[j]:
                    f.write('%d\tgood\n' % j)
                else:
                    f.write('%d\tmua\n' % j)

        # Making a copy with the label group as Phy treats this keyword separately
        with open(join(output_dir, 'cluster_group.tsv'), 'w') as f:
            f.write('cluster_id\tgroup\n')
            for j in range(len(good)):
                if good[j]:
                    f.write('%d\tgood\n' % j)
                else:
                    f.write('%d\tmua\n' % j)

        with open(join(output_dir, 'cluster_ContamPct.tsv'), 'w') as f:
            f.write('cluster_id\tContamPct\n')
            for j in range(len(good)):
                f.write('%d\t%.1f\n' % (j, 100 * est_contam_rate[j]))

        with open(join(output_dir, 'cluster_Amplitude.tsv'), 'w') as f:
            f.write('cluster_id\tAmplitude\n')
            for j in range(len(good)):
                f.write('%d\t%.1f\n' % (j, temp_amps[j]))

        # make params file
        if not os.path.exists(join(output_dir, 'params.py')):
            with open(join(output_dir, 'params.py'), 'w') as f:
                f.write('dat_path = "../%s"\n' % dat_path)
                f.write('n_channels_dat = %d\n' % probe.NchanTOT)
                f.write('dtype = "int16"\n')
                f.write('offset = 0\n')
                f.write('hp_filtered = False\n')
                f.write('sample_rate = %i\n' % params.fs)
                f.write('template_scaling = %.1f\n' % params.templateScaling)
