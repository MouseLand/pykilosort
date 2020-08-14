import os
import json
from pathlib import Path
from PyQt5 import QtWidgets, QtGui, QtCore
from scipy.io.matlab.miobase import MatReadError
from pykilosort.utils import load_probe
from pykilosort.default_params import default_params
from pykilosort.gui.minor_gui_elements import ProbeBuilder


class SettingsBox(QtWidgets.QGroupBox):

    settingsUpdated = QtCore.pyqtSignal()

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)

        self.gui = parent

        self.select_data_file = QtWidgets.QPushButton("Select Data File")
        self.data_file_path_input = QtWidgets.QLineEdit("")

        self.select_working_directory = QtWidgets.QPushButton("Select Working Directory")
        self.working_directory_input = QtWidgets.QLineEdit("")

        self.select_results_directory = QtWidgets.QPushButton("Select Results Directory")
        self.results_directory_input = QtWidgets.QLineEdit("")

        self.probe_layout_text = QtWidgets.QLabel("Select Probe Layout")
        self.probe_layout_selector = QtWidgets.QComboBox()
        self._probes = []
        self.populate_probe_selector()

        self.num_channels_text = QtWidgets.QLabel("Number of Channels")
        self.num_channels_input = QtWidgets.QLineEdit()

        self.time_range_text = QtWidgets.QLabel("Time Range (in seconds)")
        self.time_range_min_input = QtWidgets.QLineEdit()
        self.time_range_max_input = QtWidgets.QLineEdit()

        self.min_firing_rate_text = QtWidgets.QLabel("Min. Firing Rate/Channel\n(0 includes all channels)")
        self.min_firing_rate_input = QtWidgets.QLineEdit()

        self.threshold_text = QtWidgets.QLabel("Threshold")
        self.threshold_upper_input = QtWidgets.QLineEdit()
        self.threshold_lower_input = QtWidgets.QLineEdit()

        self.lambda_text = QtWidgets.QLabel("Lambda")
        self.lambda_value_input = QtWidgets.QLineEdit()

        self.auc_splits_text = QtWidgets.QLabel("AUC for Splits")
        self.auc_splits_input = QtWidgets.QLineEdit()

        self.error_label = QtWidgets.QLabel("")
        self.error_label.setText("Invalid inputs!")
        self.error_label.setWordWrap(True)

        self.advanced_options_button = QtWidgets.QPushButton("Advanced Options")

        self.load_settings_button = QtWidgets.QPushButton("Load")

        self.data_file_path = None
        self.working_directory_path = None
        self.results_directory_path = None
        self.probe_layout = None
        self.num_channels = None
        self.time_range_min = None
        self.time_range_max = None
        self.min_firing_rate = None
        self.threshold_lower = None
        self.threshold_upper = None
        self.lambda_value = None
        self.auc_splits = None

        self.setup()

        # set default parameters to trigger an update on settings
        self.num_channels_input.setText(str(1))
        self.time_range_min_input.setText(str(0))
        self.time_range_max_input.setText("inf")
        self.min_firing_rate_input.setText(str(default_params.minfr_goodchannels))
        self.threshold_upper_input.setText(str(default_params.Th[0]))
        self.threshold_lower_input.setText(str(default_params.Th[1]))
        self.lambda_value_input.setText(str(default_params.lam))
        self.auc_splits_input.setText(str(default_params.AUCsplit))

        self.settings = {}

        self.update_settings()

    def setup(self):
        self.setTitle("Settings")

        layout = QtWidgets.QVBoxLayout()

        select_data_file_layout = QtWidgets.QHBoxLayout()
        select_data_file_layout.addWidget(self.select_data_file, 70)
        select_data_file_layout.addWidget(self.data_file_path_input, 30)
        self.select_data_file.clicked.connect(self.on_select_data_file_clicked)
        self.data_file_path_input.textChanged.connect(self.on_data_file_path_changed)
        self.data_file_path_input.editingFinished.connect(self.on_data_file_path_changed)

        select_working_directory_layout = QtWidgets.QHBoxLayout()
        select_working_directory_layout.addWidget(self.select_working_directory, 70)
        select_working_directory_layout.addWidget(self.working_directory_input, 30)
        self.select_working_directory.clicked.connect(self.on_select_working_dir_clicked)
        self.working_directory_input.textChanged.connect(self.on_working_directory_changed)
        self.working_directory_input.editingFinished.connect(self.on_working_directory_changed)

        select_results_directory_layout = QtWidgets.QHBoxLayout()
        select_results_directory_layout.addWidget(self.select_results_directory, 70)
        select_results_directory_layout.addWidget(self.results_directory_input, 30)
        self.select_results_directory.clicked.connect(self.on_select_results_dir_clicked)
        self.results_directory_input.textChanged.connect(self.on_results_directory_changed)
        self.results_directory_input.editingFinished.connect(self.on_results_directory_changed)

        probe_layout_layout = QtWidgets.QHBoxLayout()
        probe_layout_layout.addWidget(self.probe_layout_text, 70)
        probe_layout_layout.addWidget(self.probe_layout_selector, 30)
        self.probe_layout_selector.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLength)
        self.probe_layout_selector.currentTextChanged.connect(self.on_probe_layout_selected)

        num_channels_layout = QtWidgets.QHBoxLayout()
        num_channels_layout.addWidget(self.num_channels_text, 70)
        num_channels_layout.addWidget(self.num_channels_input, 30)
        self.num_channels_input.textChanged.connect(self.on_number_of_channels_changed)

        time_range_layout = QtWidgets.QHBoxLayout()
        time_range_layout.addWidget(self.time_range_text, 70)
        time_range_layout.addWidget(self.time_range_min_input, 15)
        time_range_layout.addWidget(self.time_range_max_input, 15)
        self.time_range_min_input.textChanged.connect(self.on_time_range_changed)
        self.time_range_max_input.textChanged.connect(self.on_time_range_changed)

        min_firing_rate_layout = QtWidgets.QHBoxLayout()
        min_firing_rate_layout.addWidget(self.min_firing_rate_text, 70)
        min_firing_rate_layout.addWidget(self.min_firing_rate_input, 30)
        self.min_firing_rate_input.textChanged.connect(self.on_min_firing_rate_changed)

        threshold_layout = QtWidgets.QHBoxLayout()
        threshold_layout.addWidget(self.threshold_text, 70)
        threshold_layout.addWidget(self.threshold_lower_input, 15)
        threshold_layout.addWidget(self.threshold_upper_input, 15)
        self.threshold_upper_input.textChanged.connect(self.on_thresholds_changed)
        self.threshold_lower_input.textChanged.connect(self.on_thresholds_changed)

        lambda_layout = QtWidgets.QHBoxLayout()
        lambda_layout.addWidget(self.lambda_text, 70)
        lambda_layout.addWidget(self.lambda_value_input, 30)
        self.lambda_value_input.textChanged.connect(self.on_lambda_changed)

        auc_splits_layout = QtWidgets.QHBoxLayout()
        auc_splits_layout.addWidget(self.auc_splits_text, 70)
        auc_splits_layout.addWidget(self.auc_splits_input, 30)
        self.auc_splits_input.textChanged.connect(self.on_auc_splits_changed)

        error_label_layout = QtWidgets.QHBoxLayout()
        error_label_size_policy = self.error_label.sizePolicy()
        error_label_size_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Maximum)
        error_label_size_policy.setRetainSizeWhenHidden(True)
        self.error_label.setSizePolicy(error_label_size_policy)
        error_label_palette = self.error_label.palette()
        error_label_palette.setColor(QtGui.QPalette.Foreground, QtGui.QColor("red"))
        self.error_label.setPalette(error_label_palette)
        error_label_layout.addWidget(self.error_label)
        self.error_label.hide()

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(self.advanced_options_button)
        buttons_layout.addStretch(0)
        buttons_layout.addWidget(self.load_settings_button)
        self.advanced_options_button.clicked.connect(self.on_advanced_options_clicked)
        self.load_settings_button.clicked.connect(self.update_settings)

        layout.addLayout(select_data_file_layout)
        layout.addLayout(select_working_directory_layout)
        layout.addLayout(select_results_directory_layout)
        layout.addLayout(probe_layout_layout)
        layout.addLayout(num_channels_layout)
        layout.addLayout(time_range_layout)
        layout.addLayout(min_firing_rate_layout)
        layout.addLayout(threshold_layout)
        layout.addLayout(auc_splits_layout)
        layout.addLayout(error_label_layout)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)

    def on_select_data_file_clicked(self):
        data_file_name, _ = QtWidgets.QFileDialog.getOpenFileName(parent=self,
                                                                  caption="Choose data file to load...",
                                                                  directory=os.getcwd())
        if data_file_name:
            self.data_file_path_input.setText(data_file_name)

    def on_select_working_dir_clicked(self):
        working_dir_name = QtWidgets.QFileDialog.getExistingDirectoryUrl(parent=self,
                                                                         caption="Choose working directory...",
                                                                         directory=QtCore.QUrl(os.getcwd()))
        if working_dir_name:
            self.working_directory_input.setText(working_dir_name.toLocalFile())

    def on_select_results_dir_clicked(self):
        results_dir_name = QtWidgets.QFileDialog.getExistingDirectoryUrl(parent=self,
                                                                         caption="Choose results directory...",
                                                                         directory=QtCore.QUrl(os.getcwd()))
        if results_dir_name:
            self.results_directory_input.setText(results_dir_name.toLocalFile())

    def on_working_directory_changed(self):
        working_directory = Path(self.working_directory_input.text())
        try:
            assert working_directory.exists()

            self.working_directory_path = working_directory
            self.error_label.hide()
        except AssertionError:
            self.error_label.setText("Please select an existing working directory!")
            self.error_label.show()

    def on_results_directory_changed(self):
        results_directory = Path(self.results_directory_input.text())
        try:
            assert results_directory.exists()

            self.results_directory_path = results_directory
            self.error_label.hide()
        except AssertionError:
            self.error_label.setText("Please select an existing directory for results!")
            self.error_label.show()

    def on_data_file_path_changed(self):
        data_file_path = Path(self.data_file_path_input.text())
        try:
            assert data_file_path.exists()

            parent_folder = data_file_path.parent
            self.working_directory_input.setText(parent_folder.as_posix())
            self.results_directory_input.setText(parent_folder.as_posix())
            self.error_label.hide()

            self.data_file_path = data_file_path
            self.working_directory_path = parent_folder
            self.results_directory_path = parent_folder
        except AssertionError:
            self.error_label.setText("Please select a valid file path!")
            self.error_label.show()

    def update_settings(self):
        self.settings = {
            'data_file_path': self.data_file_path,
            'working_directory': self.working_directory_path,
            'results_directory': self.working_directory_path,
            'probe_layout': self.probe_layout,
            'num_channels': self.num_channels,
            'lam': self.lambda_value,
            'time_range': [self.time_range_min, self.time_range_max],
            'minfr_goodchannels': self.min_firing_rate,
            'Th': [self.threshold_upper, self.threshold_lower],
            'AUCsplit': self.auc_splits
        }

        if None not in self.settings.values():
            self.settingsUpdated.emit()

    def on_advanced_options_clicked(self):
        advanced_options_dialog = QtWidgets.QMessageBox(self)
        advanced_options_dialog.setIcon(QtWidgets.QMessageBox.Information)

        advanced_options_dialog.setText("This information will be updated soon!")
        advanced_options_dialog.setWindowTitle("Setting advanced options")
        advanced_options_dialog.setStandardButtons(QtWidgets.QMessageBox.Ok)
        advanced_options_dialog.exec_()

    def on_probe_layout_selected(self, name):
        if name not in ["", "[new]", "other..."]:
            probe_path = Path(self.gui.probe_files_path).joinpath(name)
            try:
                probe_layout = load_probe(probe_path)
                self.error_label.hide()

                self.probe_layout = probe_layout
                total_channels = self.probe_layout.NchanTOT

                self.num_channels_input.setText(str(total_channels))
            except MatReadError:
                self.error_label.setText("Invalid probe file!")
                self.error_label.show()

        elif name == "[new]":
            probe_layout, probe_name, okay = ProbeBuilder(parent=self).exec_()

            if okay:
                probe_path = Path(self.gui.probe_files_path).joinpath(probe_name + ".prb")
                with open(probe_path, 'w+') as probe_file:
                    probe_dumps = json.dumps(probe_layout)
                    probe_file.write(probe_dumps)
                assert probe_path.exists()

                self.populate_probe_selector()

                self.probe_layout = probe_layout

                total_channels = self.probe_layout.NchanTOT
                self.num_channels_input.setText(str(total_channels))
                self.error_label.hide()
            else:
                self.probe_layout_selector.setCurrentIndex(0)

    def on_number_of_channels_changed(self):
        try:
            number_of_channels = int(self.num_channels_input.text())
            assert number_of_channels > 0
            self.error_label.hide()

            self.num_channels = number_of_channels
        except ValueError:
            self.error_label.setText("Invalid input!\nNo. of channels must be an integer!")
            self.error_label.show()
        except AssertionError:
            self.error_label.setText("Invalid input!\nNo. of channels must be > 0!")
            self.error_label.show()

    def on_time_range_changed(self):
        try:
            time_range_low = float(self.time_range_min_input.text())
            time_range_high = self.time_range_max_input.text()
            if not time_range_high == "inf":
                time_range_high = float(time_range_high)
                assert 0 <= time_range_low < time_range_high
            else:
                assert 0 <= time_range_low
            self.error_label.hide()

            self.time_range_min = time_range_low
            self.time_range_max = time_range_high
        except ValueError:
            self.error_label.setText("Invalid inputs!\nTime range values must be floats!"
                                     "\n(`inf` accepted as upper limit)")
            self.error_label.show()
        except AssertionError:
            self.error_label.setText("Invalid inputs!\nCheck that 0 <= lower limit < upper limit!")
            self.error_label.show()

    def on_min_firing_rate_changed(self):
        try:
            min_firing_rate = float(self.min_firing_rate_input.text())
            assert min_firing_rate >= 0

            self.min_firing_rate = min_firing_rate
            self.error_label.hide()
        except ValueError:
            self.error_label.setText("Invalid input!\nMin. firing rate value must be a float!")
            self.error_label.show()
        except AssertionError:
            self.error_label.setText("Invalid input!\nMin. firing rate must be >= 0.0 Hz!")
            self.error_label.show()

    def on_thresholds_changed(self):
        try:
            threshold_upper = float(self.threshold_upper_input.text())
            threshold_lower = float(self.threshold_lower_input.text())
            assert 0 < threshold_lower < threshold_upper

            self.threshold_upper = threshold_upper
            self.threshold_lower = threshold_lower
            self.error_label.hide()
        except ValueError:
            self.error_label.setText("Invalid inputs!\nThreshold values must be floats!")
            self.error_label.show()
        except AssertionError:
            self.error_label.setText("Invalid inputs!\nCheck that 0 < lower threshold < upper threshold!")
            self.error_label.show()

    def on_lambda_changed(self):
        try:
            lambda_value = float(self.lambda_value_input.text())
            assert 0 < lambda_value
            self.error_label.hide()

            self.lambda_value = lambda_value
        except ValueError:
            self.error_label.setText("Invalid input!\nLambda value must be a float!")
            self.error_label.show()
        except AssertionError:
            self.error_label.setText("Invalid input!\nLambda value must be > 0!")
            self.error_label.show()

    def on_auc_splits_changed(self):
        try:
            auc_split = float(self.auc_splits_input.text())
            assert 0 <= auc_split <= 1
            self.error_label.hide()

            self.auc_splits = auc_split
        except ValueError:
            self.error_label.setText("Invalid input!\nAUC split value must be a float!")
            self.error_label.show()
        except AssertionError:
            self.error_label.setText("Invalid input!\nCheck that 0 <= AUC split <= 1!")
            self.error_label.show()

    def populate_probe_selector(self):
        self.probe_layout_selector.clear()

        probe_folder = self.gui.probe_files_path
        probes = os.listdir(probe_folder)
        probes = [probe for probe in probes if probe.endswith(".mat") or probe.endswith(".prb")]

        self.probe_layout_selector.addItems([""] + probes + ["[new]", "other..."])
        self._probes = probes
