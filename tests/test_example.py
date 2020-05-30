import pytest

from examples import example_sort_local

@pytest.mark.requires_gpu
def test_example():
    example_sort_local.run_example()
