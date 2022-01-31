# Python port of KiloSort2

This is a work-in-progress literal Python port of the original MATLAB version of Kilosort 2, written by Marius Pachitariu.
The code is still being debugged and is not ready for use.

## Scope
### Why an IBL port of pykilosort? (draft)
Preprocessing functions and standardization of outputs

Kush Banga, Cyrille Rossant, Olivier Winter

## Installation 

### System Requirements

The code makes extensive use of the GPU via the CUDA framework. A high-end NVIDIA GPU with at least 8GB of memory is required.

A good CPU and a large amount of RAM (minimum 32GB or 64GB) is also required.

See [the Wiki on the Matlab version](https://github.com/MouseLand/Kilosort2/wiki/8.-Hardware-guide) for more information.

<!-- TODO: What OS's does this work on? I am testing with Ubuntu .04. -->

You will need NVIDIA drivers and cuda-toolkit installed on your computer too. This can be the hardest part of the installation. To test if your is working OK you should be able to run the following:
```
nvidia-smi # Should show how much your GPU is being used right now
nvcc # This is the CUDA compiler
```

### Doing the install using Anaconda

On Linux install 
    
    sudo apt-get install -y libfftw3-dev

Clone the repository:

    git clone -b drift_test_stable https://github.com/kushbanga/pykilosort.git
    cd pykilosort

Create a conda environment

    conda env create -f ./pyks2.yml
    conda activate pyks2
    conda develop .

## Usage

### Example

This is how to run for general users
```python
from pathlib import Path
from pykilosort import run, add_default_handler, np1_probe, np2_probe

# Run standard ks2.5 algorithm for a np1 probe
data_path = Path('path/to/data/data.bin')
dir_path = Path('path/to/output/folder') # by default uses the same folder as the dataset
add_default_handler(level='INFO') # print output as the algorithm runs
run(data_path, dir_path=dir_path, probe=np1_probe())

# Run chronic recordings for a np2 probe
# For now this still uses ks2.5 clustering, chronic clustering algorithm coming soon!
data_paths = [
    Path('path/to/first/dataset/dataset.bin'),
    Path('path/to/second/dataset/dataset.bin'),
    Path('path/to/third/dataset/dataset.bin'),
]
dir_path = Path('path/to/output/folder') # by default uses the same folder as the first dataset
add_default_handler(level='INFO')
run(data_paths, dir_path=dir_path, probe=np2_probe(), low_memory=True)
```

This is how to run for NP1.0 probe (for IBL)
```python
import shutil
from pathlib import Path
import numpy as np

import pykilosort
from pykilosort.ibl import run_spike_sorting_ibl, ibl_pykilosort_params

INTEGRATION_DATA_PATH = Path("/datadisk/Data/spike_sorting/pykilosort_tests")
SCRATCH_DIR = Path.home().joinpath("scratch", 'pykilosort')
shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
SCRATCH_DIR.mkdir(exist_ok=True)
DELETE = True  # delete the intermediate run products, if False they'll be copied over
bin_file = INTEGRATION_DATA_PATH.joinpath("imec_385_100s.ap.bin")
# this is the output of the pykilosort data, unprocessed after the spike sorter
ks_output_dir = INTEGRATION_DATA_PATH.joinpath("results")
ks_output_dir.mkdir(parents=True, exist_ok=True)
# this is the output standardized as per IBL standards (SI units, ALF convention)
alf_path = ks_output_dir.joinpath('alf')


params = ibl_pykilosort_params()
run_spike_sorting_ibl(bin_file, delete=DELETE, scratch_dir=SCRATCH_DIR,
                      ks_output_dir=ks_output_dir, alf_path=alf_path, log_level='DEBUG', params=params)
```

### Disk cache (serialized results & parameter objects)

The MATLAB version used a big `rez` structured object containing the input data, the parameters, intermediate and final results.

The Python version makes the distinction between:

- `raw_data`: a NumPy-like object of shape `(n_channels_total, n_samples)`
- `probe`: a Bunch instance (dictionary) with the channel coordinates, the indices of the "good channels"
- `params`: a Bunch instance (dictionary) with optional user-defined parameters. It can be empty. Any missing parameter is transparently replaced by the default as found in `default_params.py` file in the repository.
- `intermediate`: a Bunch instance (dictionary) with intermediate arrays.

These objects are accessible via the *context* (`ctx`) which replaces the MATLAB `rez` object: `ctx.raw_data`, etc.

This context also stores a special object called `ctx.intermediate` which stores intermediate arrays. This object derives from `Bunch` and implements special methods to save and load arrays in a temporary folder. By default, an intermediate result called `ctx.intermediate.myarray` is stored in `./.kilosort/context/myarray.npy`.

The main `run()` function checks the existence of some of these intermediate arrays to skip some steps that might have run already, for a given dataset.

The suffixes `_m` (merge), `_s` (split), `_c` (cutoff) are used to disambiguate between multiple processing steps for the same arrays (they would be overwritten otherwise).


## Technical notes about the port

The following differences between MATLAB and Python required special care during the port:

* Discrepancy between 0-based and 1-based indexing.
* MATLAB uses Fortran ordering for arrays, whereas NumPy uses C ordering by default. The Python code therefore uses Fortran ordering exclusively so that the custom CUDA kernels can be used with no modification.
* In MATLAB, arrays can be extended transparently with indexing, whereas NumPy/CuPy requires explicit concatenation.
* The MATLAB code used mex C files to launch CUDA kernels, whereas the Python code uses CuPy directly.
* A few workarounds around limitations of CuPy compared to MATLAB: no `cp.median()`, no GPU version of the `lfilter()` LTI filter in CuPy (a custom CUDA kernel had to be written), etc.
