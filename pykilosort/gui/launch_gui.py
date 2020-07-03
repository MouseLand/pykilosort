import os
import sys
import numpy as np
import logging
import pyqtgraph as pg
from pathlib import Path
# TODO: optimize imports before incorporating into codebase
from phylib.io.traces import FlatEphysReader
from pykilosort.gui import DataViewBox, ProbeViewBox, SettingsBox, RunBox, MessageLogBox, HeaderBox
from pykilosort.gui import DarkPalette
from pykilosort.default_params import default_params, set_dependent_params
from pykilosort.utils import Context
from pykilosort.main import default_probe
from pykilosort.gui import probes
from PyQt5 import QtGui, QtWidgets, QtCore


class KiloSortGUI(QtWidgets.QMainWindow):

    def __init__(self, application, *args, **kwargs):
        super(KiloSortGUI, self).__init__(*args, **kwargs)

        self.app = application

        self.data_path = None
        self.probe_layout = None
        self.params = None
        self.working_directory = None
        self.results_directory = None

        self.probe_files_path = Path(probes.__file__).parent
        assert self.probe_files_path.exists()

        self.time_range = None
        self.num_channels = None

        self.context = None
        self.raw_data = None

        self.content = QtWidgets.QWidget(self)
        self.content_layout = QtWidgets.QVBoxLayout()

        self.header_box = HeaderBox(self)

        self.boxes = QtWidgets.QWidget()
        self.boxes_layout = QtWidgets.QHBoxLayout(self.boxes)
        self.second_boxes_layout = QtWidgets.QVBoxLayout()

        self.settings_box = SettingsBox(self)
        self.probe_view_box = ProbeViewBox(self)
        self.data_view_box = DataViewBox(self)
        self.run_box = RunBox(self)
        self.message_log_box = MessageLogBox(self)

        self.setup()

    def keyPressEvent(self, event):
        QtWidgets.QMainWindow.keyPressEvent(self, event)

        if type(event) == QtGui.QKeyEvent:
            if event.key() == QtCore.Qt.Key_Up:
                self.change_channel(shift=1)
            elif event.key() == QtCore.Qt.Key_Down:
                self.change_channel(shift=-1)
            elif event.key() == QtCore.Qt.Key_C:
                self.toggle_view()
            elif event.key() == QtCore.Qt.Key_1:
                self.toggle_mode("raw")
            elif event.key() == QtCore.Qt.Key_2:
                self.toggle_mode("whitened")
            elif event.key() == QtCore.Qt.Key_3:
                self.toggle_mode("prediction")
            elif event.key() == QtCore.Qt.Key_4:
                self.toggle_mode("residual")
            else:
                pass
            event.accept()
        else:
            event.ignore()

    def setup(self):
        self.setWindowTitle("KiloSort2 GUI")

        self.content_layout.addWidget(self.header_box, 3)

        self.second_boxes_layout.addWidget(self.settings_box, 85)
        self.second_boxes_layout.addWidget(self.run_box, 15)

        self.boxes_layout.addLayout(self.second_boxes_layout, 20)
        self.boxes_layout.addWidget(self.probe_view_box, 15)
        self.boxes_layout.addWidget(self.data_view_box, 65)

        self.boxes.setLayout(self.boxes_layout)
        self.content_layout.addWidget(self.boxes, 90)
        self.content_layout.addWidget(self.message_log_box, 7)

        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content.setLayout(self.content_layout)
        self.setCentralWidget(self.content)

        self.settings_box.settingsUpdated.connect(self.set_parameters)

        self.data_view_box.channelChanged.connect(self.probe_view_box.update_channel)

    def change_channel(self, shift):
        # TODO: shift channel by +1 or -1
        pass

    def toggle_view(self):
        # TODO: toggle between traces view and colormap view
        self.data_view_box.toggle_view()

    def toggle_mode(self, mode):
        if mode == "raw":
            self.data_view_box.raw_button.toggle()
        elif mode == "whitened":
            self.data_view_box.whitened_button.toggle()
        elif mode == "prediction":
            self.data_view_box.prediction_button.toggle()
        elif mode == "residual":
            self.data_view_box.residual_button.toggle()
        else:
            raise ValueError("Invalid mode requested!")

    def set_parameters(self):
        settings = self.settings_box.settings

        self.data_path = settings.pop('data_file_path')
        self.working_directory = settings.pop('working_directory')
        self.results_directory = settings.pop('results_directory')
        self.probe_layout = settings.pop('probe_layout')
        self.time_range = settings.pop('time_range')
        self.num_channels = settings.pop('num_channels')

        params = default_params.copy()
        set_dependent_params(params)
        params.update(settings)

        assert params

        self.params = params

        self.update_data_view()

    def load_context(self):
        context_path = Path(os.path.join(self.working_directory, '.kilosort', self.raw_data.name))

        self.context = Context(context_path=context_path)
        self.context.probe = self.probe_layout
        self.context.params = self.params
        self.context.raw_data = self.raw_data

        self.context.load()

    def update_data_view(self):

        # TODO: account for these temporary hardcoded params
        n_channels = 385
        dtype = np.int16
        sample_rate = 3e4

        raw_data = FlatEphysReader(self.data_path, sample_rate=sample_rate, dtype=dtype, n_channels=n_channels)
        self.raw_data = raw_data

        # TODO: account for this hackish setup of probe
        self.probe_layout = default_probe(self.raw_data)

        if self.context is None:
            self.load_context()

        self.data_view_box.set_seek_range(self.context)
        self.data_view_box.update_plot(self.context)


if __name__ == "__main__":
    kilosort_application = QtWidgets.QApplication(sys.argv)
    kilosort_application.setStyle("Fusion")
    kilosort_application.setPalette(DarkPalette())
    kilosort_application.setStyleSheet("QToolTip { color: #aeadac;"
                                       "background-color: #35322f;"
                                       "border: 1px solid #aeadac; }")

    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')
    pg.setConfigOption('useOpenGL', True)

    kilosort_gui = KiloSortGUI(kilosort_application)
    kilosort_gui.showMaximized()
    kilosort_gui.show()

    sys.exit(kilosort_application.exec_())
