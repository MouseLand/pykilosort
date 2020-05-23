# Contributing
Contributing to pykilosort

## Installation

The installation process is the same for everyone. See the README.

## Dependencies

Dependencies need to be added to the conda environment files (`pyks2.yml`) and the requirements.txt. Please pin any critical dependencies to at least a minor version. Ideally we should avoid adding dependencies that are not actively maintained or are not widely used (unless we are willing to support them ourselves).

## Testing

### Running the tests

*Full tests (with cupy and the GPU)* - Simply run `pytest` in the root dir.

*CPU-only tests (without cupy)* - Activate the test environment `conda activate pyks2_test.yml` and then run the tests like so: `MOCK_CUPY=True pytest -m "not requires_gpu"`. These are the tests that are run on circleCI.

These tests make extensive use of mocking ([python mock module](https://docs.python.org/3/library/unittest.mock.html) so that we can abstract away the GPU parts and continue to test the surrounding code. 

### Marking tests

Please mark tests that meet any of the following description (copied from setup.cfg). e.g. a test that runs sorting on a large dataset may be useful but we don't want to run it too often so mark it with `@pytest.mark.slow`.

requires\_gpu:
    Tests requires cupy (with gpu & CUDA). Tests that seem
    pointless without a GPU but still work should *not* use this.
slow: marks tests as slow, we skip these on CI too.
