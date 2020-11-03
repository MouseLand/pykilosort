import os
import numpy as np
from pathlib import Path
# TODO: optimize imports before incorporating into codebase
from phylib.io.traces import get_ephys_reader
from pykilosort.gui import DataViewBox, ProbeViewBox, SettingsBox, RunBox, MessageLogBox, HeaderBox, KiloSortWorker
from pykilosort.params import KilosortParams
from pykilosort.utils import Context
from pykilosort.gui import probes
from pykilosort.gui.logger import setup_logger
from pykilosort import __version__
from PyQt5 import QtGui, QtWidgets, QtCore

logger = setup_logger(__name__)


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

        self.local_config_path = Path.home() / ".pykilosort"
        self.local_config_path.mkdir(exist_ok=True)

        self.new_probe_files_path = self.local_config_path / "probes"
        self.new_probe_files_path.mkdir(exist_ok=True)

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

        self.message_log_dock = QtWidgets.QDockWidget("Message Log", self)

        self.setup()

    def keyPressEvent(self, event):
        QtWidgets.QMainWindow.keyPressEvent(self, event)

        if type(event) == QtGui.QKeyEvent:
            modifiers = event.modifiers()

            if event.key() == QtCore.Qt.Key_Up:
                if modifiers:
                    if modifiers == QtCore.Qt.ControlModifier:
                        # logger.debug("Ctrl+Up")
                        self.change_channel_display(1)
                    elif modifiers == QtCore.Qt.AltModifier:
                        # logger.debug("Alt+Up")
                        self.scale_data(1)
                    elif modifiers == QtCore.Qt.ShiftModifier:
                        # logger.debug("Shift+Up")
                        self.zoom_data_view_in_time(1)
                else:
                    # logger.debug("Only Up")
                    self.change_displayed_channel_count(shift=1)
            elif event.key() == QtCore.Qt.Key_Down:
                if modifiers:
                    if modifiers == QtCore.Qt.ControlModifier:
                        # logger.debug("Ctrl+Down")
                        self.change_channel_display(-1)
                    elif modifiers == QtCore.Qt.AltModifier:
                        # logger.debug("Alt+Down")
                        self.scale_data(-1)
                    elif modifiers == QtCore.Qt.ShiftModifier:
                        # logger.debug("Shift+Down")
                        self.zoom_data_view_in_time(-1)
                else:
                    # logger.debug("Only Down")
                    self.change_displayed_channel_count(shift=-1)
            elif event.key() == QtCore.Qt.Key_Left:
                self.shift_data(time_shift=-1)
            elif event.key() == QtCore.Qt.Key_Right:
                self.shift_data(time_shift=1)
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
        self.setWindowTitle(f"Kilosort{__version__}")

        self.message_log_dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        self.message_log_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable | QtWidgets.QDockWidget.DockWidgetMovable)
        self.message_log_dock.setWidget(self.message_log_box)

        self.content_layout.addWidget(self.header_box, 3)

        self.second_boxes_layout.addWidget(self.settings_box, 85)
        self.second_boxes_layout.addWidget(self.run_box, 15)

        self.boxes_layout.addLayout(self.second_boxes_layout, 20)
        self.boxes_layout.addWidget(self.probe_view_box, 10)
        self.boxes_layout.addWidget(self.data_view_box, 70)

        self.boxes.setLayout(self.boxes_layout)
        self.content_layout.addWidget(self.boxes, 90)

        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content.setLayout(self.content_layout)
        self.setCentralWidget(self.content)

        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.message_log_dock)

        self.header_box.reset_gui_button.clicked.connect(self.reset_gui)

        self.settings_box.settingsUpdated.connect(self.set_parameters)
        self.settings_box.previewProbe.connect(self.probe_view_box.preview_probe)

        self.data_view_box.channelChanged.connect(self.probe_view_box.update_probe_view)
        self.data_view_box.modeChanged.connect(self.probe_view_box.synchronize_data_view_mode)

        self.probe_view_box.channelSelected.connect(self.data_view_box.change_primary_channel)

        self.run_box.updateContext.connect(self.update_context)
        self.run_box.sortingStepStatusUpdate.connect(self.update_sorting_status)

    def change_channel_display(self, direction):
        if self.context is not None:
            self.data_view_box.change_channel_display(direction)

    def shift_data(self, time_shift):
        if self.context is not None:
            self.data_view_box.shift_current_time(time_shift)

    def change_displayed_channel_count(self, shift):
        if self.context is not None:
            self.data_view_box.change_displayed_channel_count(shift)

    def zoom_data_view_in_time(self, direction):
        if self.context is not None:
            self.data_view_box.change_plot_range(direction)

    def scale_data(self, direction):
        if self.context is not None:
            self.data_view_box.change_plot_scaling(direction)

    def toggle_view(self):
        self.data_view_box.traces_button.toggle()

    def toggle_mode(self, mode):
        if mode == "raw":
            self.data_view_box.raw_button.click()
        elif mode == "whitened":
            self.data_view_box.whitened_button.click()
        elif mode == "prediction":
            self.data_view_box.prediction_button.click()
        elif mode == "residual":
            self.data_view_box.residual_button.click()
        else:
            raise ValueError("Invalid mode requested!")

    def set_parameters(self):
        settings = self.settings_box.settings
        advanced_options = self.settings_box.advanced_options

        self.data_path = settings.pop('data_file_path')
        self.working_directory = settings.pop('working_directory')
        self.results_directory = settings.pop('results_directory')
        self.probe_layout = settings.pop('probe_layout')
        self.time_range = settings.pop('time_range')
        self.num_channels = settings.pop('num_channels')

        params = KilosortParams()
        params = params.parse_obj(advanced_options)
        params = params.parse_obj(settings)

        assert params

        self.params = params

        self.settings_box.disable_all_buttons()
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))

        self.prepare_for_new_context()
        self.load_raw_data()
        self.setup_context()
        self.setup_probe_view()
        self.setup_data_view()
        self.update_run_box()

        self.settings_box.reenable_all_buttons()
        QtWidgets.QApplication.restoreOverrideCursor()

    def load_raw_data(self):
        # TODO: account for these temporary hardcoded params
        n_channels = self.num_channels
        dtype = np.int16
        sample_rate = self.params.fs

        raw_data = get_ephys_reader(self.data_path, sample_rate=sample_rate, dtype=dtype, n_channels=n_channels)
        self.raw_data = raw_data

    def setup_data_view(self):
        self.data_view_box.setup_seek(self.context)
        self.data_view_box.update_plot(self.context)

    def update_context_with_good_channels(self):
        worker = KiloSortWorker(self.context, self.data_path, self.results_directory, ["goodchannels"])

        worker.foundGoodChannels.connect(self.update_context)

        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        worker.start()
        while worker.isRunning():
            QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.restoreOverrideCursor()

    def setup_context(self):
        context_path = Path(os.path.join(self.working_directory, '.kilosort', self.raw_data.name))

        self.context = Context(context_path=context_path)
        self.context.probe = self.probe_layout
        self.context.params = self.params
        self.context.raw_data = self.raw_data

        self.context.load()

        self.update_context_with_good_channels()

    @QtCore.pyqtSlot(object)
    def update_context(self, context):
        self.context = context

    def setup_probe_view(self):
        self.probe_view_box.set_layout(self.context)

    def update_run_box(self):
        self.run_box.set_data_path(self.data_path)
        self.run_box.set_working_directory(self.working_directory)
        self.run_box.set_results_directory(self.results_directory)

    @QtCore.pyqtSlot(dict)
    def update_sorting_status(self, status_dict):
        self.data_view_box.change_sorting_status(status_dict)

    def get_context(self):
        return self.context

    def get_probe(self):
        return self.probe_layout

    def get_params(self):
        return self.params

    def prepare_for_new_context(self):
        self.data_view_box.prepare_for_new_context()
        self.probe_view_box.prepare_for_new_context()
        self.message_log_box.prepare_for_new_context()

        self.context = None

    def reset_gui(self):
        self.time_range = None
        self.num_channels = None
        self.context = None
        self.raw_data = None

        self.probe_view_box.reset()
        self.data_view_box.reset()
        self.settings_box.reset()
        self.message_log_box.reset()

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.message_log_box.save_log_file()
        event.accept()
