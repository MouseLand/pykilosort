import numpy as np
import cupy as cp

from pykilosort.learn import mexWtW2


def get_time_lags(array1, array2):
    """
    Takes the dot product of two arrays across all time lags
    :param array1: 1D Numpy array
    :param array2: 1D Numpy array
    :return: 1D Numpy array of dot products
    """
    nt = array1.shape[0]
    n = nt * 2 - 1
    output = np.zeros(n)
    for i in range(n):
        if i < nt:
            output[i] = np.dot(array1[:(i+1)], array2[-(i+1):])
        else:
            output[i] = np.dot(array1[(i+1-nt):], array2[:-(i+1-nt)])
    return output


def mexWtW2_cpu(W1, W2, UtU):
    "Slow cpu version of mex_WtW2 for testing"
    n_times, n_templates = W1.shape
    n = n_times * 2 - 1
    template_corrs = np.zeros((n_templates, n_templates, n))
    for i in range(n_templates):
        for j in range(n_templates):
            template_corrs[i,j] = get_time_lags(W1[:,i], W2[:,j]) * UtU[i,j]
    return template_corrs


def test_mexWtW2():
    n_times = 61
    n_templates = 100

    W1 = np.random.randn(n_times, n_templates)
    W2 = np.random.randn(n_times, n_templates)
    UtU = np.random.random((n_templates, n_templates))

    output_cpu = mexWtW2_cpu(W1, W2, UtU)

    W1 = cp.asarray(W1, dtype=np.float32, order='F')
    W2 = cp.asarray(W2, dtype=np.float32, order='F')
    UtU = cp.asarray(UtU, dtype=np.float32, order='F')

    output = mexWtW2(W1, W2, UtU)
    assert np.allclose(output_cpu, cp.asnumpy(output), atol=1e-05)
