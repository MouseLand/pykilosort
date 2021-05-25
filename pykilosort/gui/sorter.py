import cupy as cp
import numpy as np
from numba import jit
from pykilosort.main import run_export, run_preprocess, run_spikesort
from pykilosort.preprocess import gpufilter
from PyQt5 import QtCore


def filter_and_whiten(raw_traces, params, probe, whitening_matrix):
    sample_rate = params.fs
    high_pass_freq = params.fshigh
    low_pass_freq = params.fslow
    scaleproc = params.scaleproc

    whitening_matrix_np = cp.asarray(whitening_matrix, dtype=np.float32) / np.float(
        scaleproc
    )

    filtered_data = gpufilter(
        buff=cp.asarray(raw_traces, dtype=np.float32),
        chanMap=probe.chanMap,
        fs=sample_rate,
        fslow=low_pass_freq,
        fshigh=high_pass_freq,
    )

    whitened_data = cp.dot(filtered_data, whitening_matrix_np)

    array_means = cp.mean(whitened_data, axis=0)
    array_stds = cp.std(whitened_data, axis=0)
    whitened_array = (whitened_data - array_means) / array_stds
    return whitened_array.get()


@jit(nopython=True)
def get_predicted_traces(
        matrix_U: np.ndarray,
        matrix_W: np.ndarray,
        sorting_result: np.ndarray,
        time_limits: tuple,
) -> np.ndarray:
    W = np.ascontiguousarray(matrix_W)
    U = np.ascontiguousarray(matrix_U)

    buffer = W.shape[0]

    predicted_traces = np.zeros(
        (U.shape[0], 4 * buffer + (time_limits[1] - time_limits[0])),
        dtype=np.int16
    )

    all_spike_times = sorting_result[:, 0]
    included_spike_pos = np.where(
            (all_spike_times > time_limits[0] - buffer // 2) &
            (all_spike_times < time_limits[1] + buffer // 2)
        )[0]

    spike_times = all_spike_times[
        included_spike_pos
    ].astype(np.int32)

    spike_templates = sorting_result[
        included_spike_pos, 1
    ].astype(np.int32)

    spike_amplitudes = sorting_result[
        included_spike_pos, 2
    ]

    for s, spike in enumerate(spike_times):
        amplitude = spike_amplitudes[s]
        U_i = np.ascontiguousarray(U[:, spike_templates[s], :])
        W_i = np.ascontiguousarray(W[:, spike_templates[s], :])

        addendum = (U_i @ W_i.T * amplitude).astype(np.int16)

        pred_pos = np.arange(buffer) + spike - time_limits[0] + buffer + buffer // 2
        predicted_traces[:, pred_pos] += addendum

    output = predicted_traces[:, buffer * 2: -buffer * 2]

    return output.T


class KiloSortWorker(QtCore.QThread):
    finishedPreprocess = QtCore.pyqtSignal(object)
    finishedSpikesort = QtCore.pyqtSignal(object)
    finishedAll = QtCore.pyqtSignal(object)

    def __init__(
            self,
            context,
            data_path,
            output_directory,
            steps,
            sanity_plots=False,
            plot_widgets=None,
            *args,
            **kwargs):
        super(KiloSortWorker, self).__init__(*args, **kwargs)
        self.context = context
        self.data_path = data_path
        self.output_directory = output_directory

        self.sanity_plots = sanity_plots
        self.plot_widgets = plot_widgets

        assert isinstance(steps, list) or isinstance(steps, str)
        self.steps = steps if isinstance(steps, list) else [steps]

    def run(self):
        if "preprocess" in self.steps:
            self.context = run_preprocess(self.context)
            self.finishedPreprocess.emit(self.context)

        if "spikesort" in self.steps:
            self.context = run_spikesort(self.context,
                                         sanity_plots=self.sanity_plots,
                                         plot_widgets=self.plot_widgets)
            self.finishedSpikesort.emit(self.context)

        if "export" in self.steps:
            run_export(self.context, self.data_path, self.output_directory)
            self.finishedAll.emit(self.context)
