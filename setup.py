# -*- coding: utf-8 -*-
# flake8: noqa

"""Installation script."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
from pathlib import Path
import re

from setuptools import setup


#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

if 'CONDA_BUILD' in os.environ and 'RECIPE_DIR' in os.environ:
    # 'conda-build' configuration
    probe_path = 'probes'
else:
    probe_path = os.path.join(os.path.expanduser('~'), '.pykilosort', 'probes')

def _package_tree(pkgroot):
    path = op.dirname(__file__)
    subdirs = [op.relpath(i[0], path).replace(op.sep, '.')
               for i in os.walk(op.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return subdirs


readme = (Path(__file__).parent / 'README.md').read_text()

install_requires = [
    "tqdm",
    "click",
    "mock",
    "cupy",
    "numpy",
    "numba",
    "scipy",
    "matplotlib",
    "pyqtgraph",
    "PyQt5",
    "pydantic",
    "spikeextractors",
    "pytest",
    "pytest-cov",
    "phy==2.0b1",
]

# Find version number from `__init__.py` without executing it.
with (Path(__file__).parent / 'pykilosort/__init__.py').open('r') as f:
    version = re.search(r"__version__ = '([^']+)'", f.read()).group(1)


setup(
    name='pykilosort',
    version=version,
    license="BSD",
    description='Python port of KiloSort 2',
    install_requires=install_requires,
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Cyrille Rossant',
    author_email='cyrille.rossant@gmail.com',
    url='https://github.com/rossant/pykilosort',
    packages=_package_tree('pykilosort'),
    package_dir={'pykilosort': 'pykilosort'},
    package_data={
        'pykilosort': [
            'cuda/*.cu',
        ],
    },
    data_files=[
        (
            probe_path,
            [
                os.path.join('gui', 'probes', 'cortexlab-single-phase-3_chanMap.prb'),
                os.path.join('gui', 'probes', 'Linear16x1_kilosortChanMap.prb'),
                os.path.join('gui', 'probes', 'NeuropixelsPhase3A301_kilosortChanMap.prb'),
                os.path.join('gui', 'probes', 'neuropixPhase3A_kilosortChanMap.prb'),
                os.path.join('gui', 'probes', 'neuropixPhase3B1_kilosortChanMap.prb'),
                os.path.join('gui', 'probes', 'neuropixPhase3B1_kilosortChanMap_all.prb'),
                os.path.join('gui', 'probes', 'neuropixPhase3B2_kilosortChanMap.prb'),
                os.path.join('gui', 'probes', 'NP2_kilosortChanMap.prb'),
            ]
        )
    ],
    include_package_data=True,
    keywords='kilosort,spike sorting,electrophysiology,neuroscience',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Framework :: IPython",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    entry_points={
        'console_scripts': [
            'kilosort = pykilosort.gui.launch:launcher'
        ],
    },
)
