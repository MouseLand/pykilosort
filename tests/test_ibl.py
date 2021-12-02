from pathlib import Path
from ibllib.io import spikeglx
from pykilosort.ibl import probe_geometry
INTEGRATION_DATA_PATH = Path("/datadisk/Data/spike_sorting/pykilosort_tests")
bin_file = INTEGRATION_DATA_PATH.joinpath("imec_385_100s.ap.cbin")


bin_file = "/datadisk/Data/spike_sorting/benchmark/raw/8413c5c6-b42b-4ec6-b751-881a54413628/_spikeglx_ephysData_g0_t0.imec0.ap.cbin"
sr = spikeglx.Reader(bin_file)

trace_header = sr.geometry

probe_geometry(sr)
# Out[10]: dict_keys(['ind', 'row', 'shank', 'col', 'x', 'y', 'sample_shift', 'adc'])
help(probe_geometry)
