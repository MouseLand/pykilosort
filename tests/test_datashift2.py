import numpy as np

from pykilosort.datashift2 import shift_data

def test_shift_data():
    data = np.asfortranarray(np.random.randint(-100, 100, size=(65000, 384), dtype=np.int16))
    transformation = np.random.normal(size=(384, 384))
    test_result = shift_data(data, transformation)
    assert type(test_result == np.ndarray)
    assert test_result.dtype == np.int16
    assert np.isfortran(test_result)
    assert np.allclose(test_result, np.asfortranarray((data @ transformation.T).astype("int16")))