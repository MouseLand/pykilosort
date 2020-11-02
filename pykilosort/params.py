import typing as t
from math import ceil

import numpy as np
from pydantic import BaseModel, Field, validator

from .utils import Bunch

# TODO: design - Let's move all of this to a yaml file with sections so that its easier to read.
#              - We can then just parse the yaml file to generate this.


class Probe(BaseModel):
    NchanTOT: int
    Nchan: t.Optional[int] = Field(
        None, description="Nchan < NchanTOT if some channels should not be used."
    )

    chanMap: np.ndarray  # TODO: add constraints
    kcoords: np.ndarray  # TODO: add constraints
    xc: np.ndarray
    yc: np.ndarray

    @validator("yc")
    def coords_same_length(cls, v, values):
        assert len(values["xc"]) == len(v)
        return v

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def load_from_npy(cls, rootZ, **kwargs):
        return cls(
            chanMap=np.load(f"{rootZ}/channel_map.npy").flatten().astype(int),
            xc=np.load(rootZ + "/channel_positions.npy")[:, 0],
            yc=np.load(rootZ + "/channel_positions.npy")[:, 1],
            **kwargs,
        )


class DatashiftParams(BaseModel):
    sig: float = Field(20.0, description="sigma for the Gaussian process smoothing")
    nblocks: int = Field(
        5, description="blocks for registration. 1 does rigid registration."
    )
    output_filename: t.Optional[str] = Field(
        None, description="optionally save registered data to a new binary file"
    )
    overwrite: bool = Field(True, description="overwrite proc file with shifted data")

    @validator("nblocks")
    def validate_nblocks(v):
        if v < 1:
            raise ValueError(
                "datashift.nblocks must be >= 1, or datashift should be None"
            )
        return v


class KilosortParams(BaseModel):
    fs: float = Field(30000.0, description="sample rate")

    probe: Probe = Field(..., description="recording probe metadata")

    fshigh: float = Field(150.0, description="high pass filter frequency")
    fslow: t.Optional[float] = Field(None, description="low pass filter frequency")
    minfr_goodchannels: float = Field(
        0.1, description="minimum firing rate on a 'good' channel (0 to skip)"
    )

    genericSpkTh: float = Field(
        10.0, description="threshold for crossings with generic templates"
    )
    nblocks: int = Field(
        1,
        description="number of blocks used to segment the probe when tracking drift, 0 == don't track, 1 == rigid, > 1 == non-rigid",
    )

    stablemode_enabled: bool = Field(False, "make output more stable")
    deterministicmode_enabled: bool = Field(False, "make output deterministic by sorting spikes before applying kernels")

    @validator("deterministicmode_enabled")
    def validate_deterministicmode(v, values):
        if values.get("stablemode_enabled"):
            return deterministicmode_enabled
        raise ValueError("stablemode must be enabled for deterministic results")

    datashift: t.Optional[DatashiftParams] = Field(
        None, description="parameters for 'datashift' drift correction. not required"
    )

    Th: t.List[float] = Field(
        [10, 4],
        description="""
        threshold on projections (like in Kilosort1, can be different for last pass like [10 4])
    """,
    )
    ThPre: float = Field(
        8,
        description="threshold crossings for pre-clustering (in PCA projection space)",
    )

    lam: float = Field(
        10,
        description="""
        how important is the amplitude penalty (like in Kilosort1, 0 means not used,
        10 is average, 50 is a lot)
    """,
    )

    AUCsplit: float = Field(
        0.9,
        description="""
        splitting a cluster at the end requires at least this much isolation for each sub-cluster (max=1)
    """,
    )

    minFR: float = Field(
        1.0 / 50,
        description="""
        minimum spike rate (Hz), if a cluster falls below this for too long it gets removed
    """,
    )

    momentum: t.List[float] = Field(
        [20, 400],
        description="""
        number of samples to average over (annealed from first to second value)
    """,
    )

    sigmaMask: float = Field(
        30,
        description="""
        spatial constant in um for computing residual variance of spike
    """,
    )

    # danger, changing these settings can lead to fatal errors
    # options for determining PCs
    spkTh: float = Field(-6, description="spike threshold in standard deviations")
    reorder: int = Field(
        1, description="whether to reorder batches for drift correction."
    )
    nskip: int = Field(
        5, description="how many batches to skip for determining spike PCs"
    )
    nSkipCov: int = Field(
        25, description="compute whitening matrix from every nth batch"
    )

    # GPU = 1  # has to be 1, no CPU version yet, sorry
    # Nfilt = 1024 # max number of clusters
    nfilt_factor: int = Field(
        4, description="max number of clusters per good channel (even temporary ones)"
    )
    ntbuff = Field(
        64,
        description="""
    samples of symmetrical buffer for whitening and spike detection
    
    Must be multiple of 32 + ntbuff. This is the batch size (try decreasing if out of memory).
    """,
    )

    whiteningRange: int = Field(
        32, description="number of channels to use for whitening each channel"
    )
    nSkipCov: int = Field(
        25, description="compute whitening matrix from every N-th batch"
    )
    scaleproc: int = Field(200, description="int16 scaling of whitened data")
    nPCs: int = Field(3, description="how many PCs to project the spikes into")

    nt0: int = 61
    nup: int = 10
    sig: int = 1
    gain: int = 1

    templateScaling: float = 20.0

    loc_range: t.List[int] = [5, 4]
    long_range: t.List[int] = [30, 6]

    Nfilt: t.Optional[
        int
    ] = None  # This should be a computed property once we add the probe to the config

    # Computed properties
    @property
    def NT(self) -> int:
        return 32 * 1024 + self.ntbuff

    @property
    def NTbuff(self) -> int:
        return self.NT + 4 * self.ntbuff

    @property
    def nt0min(self) -> int:
        return int(ceil(20 * self.nt0 / 61))
