import os
import json
import pprint
from pathlib import Path
import numpy as np
from PyQt5 import QtWidgets, QtCore
from scipy.io.matlab.miobase import MatReadError
from pykilosort.utils import load_probe, create_prb
from pykilosort.params import KilosortParams
from pykilosort.gui.logger import setup_logger
from pykilosort.gui.minor_gui_elements import ProbeBuilder, AdvancedOptionsEditor


logger = setup_logger(__name__)


class SettingsBox(QtWidgets.QGroupBox):
    settingsUpdated = QtCore.pyqtSignal()
    previewProbe = QtCore.pyqtSignal(object)

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

        self.advanced_options_button = QtWidgets.QPushButton("Advanced Options...")

        self.load_settings_button = QtWidgets.QPushButton("LOAD")
        self.probe_preview_button = QtWidgets.QPushButton("Preview Probe")

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

        default_params = KilosortParams()
        self.settings = {}
        self.advanced_options = default_params.parse_obj(self.get_default_advanced_options()).dict()

        self.setup()

    def setup(self):
        self.setTitle("Settings")

        layout = QtWidgets.QVBoxLayout()

        font = self.load_settings_button.font()
        font.setPointSize(20)
        self.load_settings_button.setFont(font)
        self.load_settings_button.setDisabled(True)
        self.load_settings_button.clicked.connect(self.update_settings)

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

        probe_preview_layout = QtWidgets.QHBoxLayout()
        self.probe_preview_button.setDisabled(True)
        self.probe_preview_button.clicked.connect(self.show_probe_layout)
        probe_preview_layout.addWidget(self.probe_preview_button, 30, alignment=QtCore.Qt.AlignRight)

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

        self.advanced_options_button.clicked.connect(self.on_advanced_options_clicked)

        layout.addWidget(self.load_settings_button)
        layout.addLayout(select_data_file_layout)
        layout.addLayout(select_working_directory_layout)
        layout.addLayout(select_results_directory_layout)
        layout.addLayout(probe_layout_layout)
        layout.addLayout(probe_preview_layout)
        layout.addLayout(num_channels_layout)
        layout.addLayout(time_range_layout)
        layout.addLayout(min_firing_rate_layout)
        layout.addLayout(threshold_layout)
        layout.addLayout(auc_splits_layout)
        layout.addWidget(self.advanced_options_button)

        self.setLayout(layout)

        self.set_default_field_values(KilosortParams().parse_obj(self.advanced_options))

        self.update_settings()

    def set_default_field_values(self, default_params):
        if default_params is None:
            default_params = KilosortParams()

        self.num_channels_input.setText(str(1))
        self.time_range_min_input.setText(str(0))
        self.time_range_max_input.setText("inf")
        self.min_firing_rate_input.setText(str(default_params.minfr_goodchannels))
        self.threshold_upper_input.setText(str(default_params.Th[0]))
        self.threshold_lower_input.setText(str(default_params.Th[1]))
        self.lambda_value_input.setText(str(default_params.lam))
        self.auc_splits_input.setText(str(default_params.AUCsplit))

    def get_default_advanced_options(self):
        advanced_options_path = self.gui.local_config_path / "advanced_options.json"
        default_advanced_options_path = self.gui.local_config_path / "default_advanced_options.json"

        if advanced_options_path.exists():
            with open(advanced_options_path, "r") as advanced_options_file:
                advanced_options = json.load(advanced_options_file)

        elif default_advanced_options_path.exists():
            with open(default_advanced_options_path, "r") as default_advanced_options_file:
                advanced_options = json.load(default_advanced_options_file)

        else:
            advanced_options = KilosortParams().dict()

            with open(default_advanced_options_path, "w+") as default_advanced_options_file:
                advanced_options_dump = json.dumps(advanced_options)
                default_advanced_options_file.write(advanced_options_dump)

        return advanced_options

    def on_select_data_file_clicked(self):
        file_dialog_options = QtWidgets.QFileDialog.DontUseNativeDialog
        data_file_name, _ = QtWidgets.QFileDialog.getOpenFileName(parent=self,
                                                                  caption="Choose data file to load...",
                                                                  directory=os.path.expanduser("~"),
                                                                  options=file_dialog_options)
        if data_file_name:
            self.data_file_path_input.setText(data_file_name)

    def on_select_working_dir_clicked(self):
        file_dialog_options = QtWidgets.QFileDialog.DontUseNativeDialog
        working_dir_name = QtWidgets.QFileDialog.getExistingDirectoryUrl(parent=self,
                                                                         caption="Choose working directory...",
                                                                         directory=QtCore.QUrl(os.path.expanduser("~")),
                                                                         options=file_dialog_options)
        if working_dir_name:
            self.working_directory_input.setText(working_dir_name.toLocalFile())

    def on_select_results_dir_clicked(self):
        file_dialog_options = QtWidgets.QFileDialog.DontUseNativeDialog
        results_dir_name = QtWidgets.QFileDialog.getExistingDirectoryUrl(parent=self,
                                                                         caption="Choose results directory...",
                                                                         directory=QtCore.QUrl(os.path.expanduser("~")),
                                                                         options=file_dialog_options)
        if results_dir_name:
            self.results_directory_input.setText(results_dir_name.toLocalFile())

    def on_working_directory_changed(self):
        working_directory = Path(self.working_directory_input.text())
        try:
            assert working_directory.exists()

            self.working_directory_path = working_directory
            if self.check_settings():
                self.enable_load()
        except AssertionError:
            logger.exception("Please select an existing working directory!")
            self.disable_load()

    def on_results_directory_changed(self):
        results_directory = Path(self.results_directory_input.text())
        try:
            assert results_directory.exists()

            self.results_directory_path = results_directory
            if self.check_settings():
                self.enable_load()
        except AssertionError:
            logger.exception("Please select an existing directory for results!")
            self.disable_load()

    def on_data_file_path_changed(self):
        data_file_path = Path(self.data_file_path_input.text())
        try:
            assert data_file_path.exists()

            parent_folder = data_file_path.parent
            self.working_directory_input.setText(parent_folder.as_posix())
            self.results_directory_input.setText(parent_folder.as_posix())

            self.data_file_path = data_file_path
            self.working_directory_path = parent_folder
            self.results_directory_path = parent_folder

            if self.check_settings():
                self.enable_load()
        except AssertionError:
            logger.exception("Please select a valid file path!")
            self.disable_load()

    def disable_all_buttons(self):
        self.load_settings_button.setDisabled(True)
        self.probe_preview_button.setDisabled(True)
        self.advanced_options_button.setDisabled(True)

    def reenable_all_buttons(self):
        self.load_settings_button.setDisabled(False)
        self.probe_preview_button.setDisabled(False)
        self.advanced_options_button.setDisabled(False)

    def enable_load(self):
        self.load_settings_button.setEnabled(True)

    def disable_load(self):
        self.load_settings_button.setDisabled(True)

    def enable_preview_probe(self):
        self.probe_preview_button.setEnabled(True)

    def disable_preview_probe(self):
        self.probe_preview_button.setDisabled(True)

    def check_settings(self):
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

        return None not in self.settings.values()

    @QtCore.pyqtSlot()
    def update_settings(self):
        if self.check_settings():
            self.settingsUpdated.emit()

    @QtCore.pyqtSlot()
    def show_probe_layout(self):
        self.previewProbe.emit(self.probe_layout)

    @QtCore.pyqtSlot()
    def on_advanced_options_clicked(self):
        dialog = AdvancedOptionsEditor(parent=self)

        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self.advanced_options = dialog.get_parameters()

            save_advanced_options = QtWidgets.QMessageBox.question(
                self, "Save as defaults?",
                "Would you like to save these as the default advanced options?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )

            if save_advanced_options:
                advanced_options_dumps = json.dumps(self.advanced_options, indent=4)
                advanced_options_path = self.gui.local_config_path / "advanced_options.json"

                with open(advanced_options_path, 'w+') as advanced_options_file:
                    advanced_options_file.write(advanced_options_dumps)

    @QtCore.pyqtSlot(str)
    def on_probe_layout_selected(self, name):
        if name not in ["", "[new]", "other..."]:
            probe_path = Path(self.gui.probe_files_path).joinpath(name)
            if not probe_path.exists():
                probe_path = Path(self.gui.new_probe_files_path).joinpath(name)
            try:
                probe_layout = load_probe(probe_path)

                self.probe_layout = probe_layout
                total_channels = self.probe_layout.NchanTOT

                total_channels = self.estimate_total_channels(total_channels)

                self.num_channels_input.setText(str(total_channels))

                self.enable_preview_probe()

                if self.check_settings():
                    self.enable_load()
            except MatReadError:
                logger.exception("Invalid probe file!")
                self.disable_load()
                self.disable_preview_probe()

        elif name == "[new]":
            dialog = ProbeBuilder(parent=self)

            if dialog.exec() == QtWidgets.QDialog.Accepted:
                probe_name = dialog.get_map_name()
                probe_layout = dialog.get_probe()

                probe_name = probe_name + ".prb"
                probe_prb = create_prb(probe_layout)
                probe_path = Path(self.gui.new_probe_files_path).joinpath(probe_name)
                with open(probe_path, 'w+') as probe_file:
                    # TODO: pretty print output
                    str_prb = f"""channel_groups = {probe_prb}"""
                    probe_file.write(str_prb)
                assert probe_path.exists()

                self.populate_probe_selector()

                self.probe_layout_selector.setCurrentText(probe_name)
            else:
                self.probe_layout_selector.setCurrentIndex(0)
                self.disable_load()
                self.disable_preview_probe()

        elif name == "other...":
            file_dialog_options = QtWidgets.QFileDialog.DontUseNativeDialog
            probe_path, _ = QtWidgets.QFileDialog.getOpenFileName(parent=self,
                                                                  caption="Choose probe file...",
                                                                  filter="Probe Files (*.mat *.prb)",
                                                                  directory=os.path.expanduser("~"),
                                                                  options=file_dialog_options
                                                                  )
            if probe_path:
                try:
                    probe_path = Path(probe_path)
                    assert probe_path.exists()

                    probe_layout = load_probe(probe_path)

                    save_probe_file = QtWidgets.QMessageBox.question(
                        self, "Save probe layout?",
                        "Would you like this probe layout to appear in the list of probe layouts next time?",
                        QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Yes,
                        QtWidgets.QMessageBox.Yes)

                    if save_probe_file == QtWidgets.QMessageBox.Yes:
                        probe_prb = create_prb(probe_layout)

                        probe_name = probe_path.with_suffix(".prb").name
                        def_probe_path = Path(self.gui.probe_files_path) / probe_name
                        new_probe_path = Path(self.gui.new_probe_files_path) / probe_name

                        if not new_probe_path.exists() and not def_probe_path.exists():
                            with open(new_probe_path, 'w+') as probe_file:
                                str_dict = pprint.pformat(probe_prb, indent=4, compact=False)
                                str_prb = f"""channel_groups = {str_dict}"""
                                probe_file.write(str_prb)

                            self.populate_probe_selector()
                            self.probe_layout_selector.setCurrentText(probe_name)

                        else:
                            logger.exception("Probe with the same name already exists.")

                    else:
                        self.probe_layout = probe_layout

                        total_channels = self.probe_layout.NchanTOT
                        total_channels = self.estimate_total_channels(total_channels)
                        self.num_channels_input.setText(str(total_channels))

                        self.enable_preview_probe()

                        if self.check_settings():
                            self.enable_load()

                except AssertionError:
                    logger.exception("Please select a valid probe file (accepted types: *.prb, *.mat)!")
                    self.disable_load()
                    self.disable_preview_probe()
            else:
                self.probe_layout_selector.setCurrentIndex(0)
                self.disable_load()
                self.disable_preview_probe()

    def on_number_of_channels_changed(self):
        try:
            number_of_channels = int(self.num_channels_input.text())
            assert number_of_channels > 0

            self.num_channels = number_of_channels

            if self.check_settings():
                self.enable_load()
        except ValueError:
            logger.exception("Invalid input!\nNo. of channels must be an integer!")
            self.disable_load()
        except AssertionError:
            logger.exception("Invalid input!\nNo. of channels must be > 0!")
            self.disable_load()

    def on_time_range_changed(self):
        try:
            time_range_low = float(self.time_range_min_input.text())
            time_range_high = self.time_range_max_input.text()
            if not time_range_high == "inf":
                time_range_high = float(time_range_high)
                assert 0 <= time_range_low < time_range_high
            else:
                assert 0 <= time_range_low

            self.time_range_min = time_range_low
            self.time_range_max = time_range_high

            if self.check_settings():
                self.enable_load()
        except ValueError:
            logger.exception("Invalid inputs!\nTime range values must be floats!"
                             "\n(`inf` accepted as upper limit)")
            self.disable_load()
        except AssertionError:
            logger.exception("Invalid inputs!\nCheck that 0 <= lower limit < upper limit!")
            self.disable_load()

    def on_min_firing_rate_changed(self):
        try:
            min_firing_rate = float(self.min_firing_rate_input.text())
            assert min_firing_rate >= 0

            self.min_firing_rate = min_firing_rate

            if self.check_settings():
                self.enable_load()
        except ValueError:
            logger.exception("Invalid input!\nMin. firing rate value must be a float!")
            self.disable_load()
        except AssertionError:
            logger.exception("Invalid input!\nMin. firing rate must be >= 0.0 Hz!")
            self.disable_load()

    def on_thresholds_changed(self):
        try:
            threshold_upper = float(self.threshold_upper_input.text())
            threshold_lower = float(self.threshold_lower_input.text())
            assert 0 < threshold_lower < threshold_upper

            self.threshold_upper = threshold_upper
            self.threshold_lower = threshold_lower

            if self.check_settings():
                self.enable_load()
        except ValueError:
            logger.exception("Invalid inputs!\nThreshold values must be floats!")
            self.disable_load()
        except AssertionError:
            logger.exception("Invalid inputs!\nCheck that 0 < lower threshold < upper threshold!")
            self.disable_load()

    def on_lambda_changed(self):
        try:
            lambda_value = float(self.lambda_value_input.text())
            assert 0 < lambda_value

            self.lambda_value = lambda_value

            if self.check_settings():
                self.enable_load()
        except ValueError:
            logger.exception("Invalid input!\nLambda value must be a float!")
            self.disable_load()
        except AssertionError:
            logger.exception("Invalid input!\nLambda value must be > 0!")
            self.disable_load()

    def on_auc_splits_changed(self):
        try:
            auc_split = float(self.auc_splits_input.text())
            assert 0 <= auc_split <= 1

            self.auc_splits = auc_split

            if self.check_settings():
                self.enable_load()
        except ValueError:
            logger.exception("Invalid input!\nAUC split value must be a float!")
            self.disable_load()
        except AssertionError:
            logger.exception("Invalid input!\nCheck that 0 <= AUC split <= 1!")
            self.disable_load()

    def populate_probe_selector(self):
        self.probe_layout_selector.clear()

        probe_folders = [self.gui.probe_files_path, self.gui.new_probe_files_path]

        probes_list = []
        for probe_folder in probe_folders:
            probes = os.listdir(probe_folder)
            probes = [probe for probe in probes if probe.endswith(".mat") or probe.endswith(".prb")]
            probes_list.extend(probes)

        self.probe_layout_selector.addItems([""] + probes_list + ["[new]", "other..."])
        self._probes = probes_list

    def estimate_total_channels(self, num_channels):
        if self.data_file_path is not None:
            memmap_data = np.memmap(self.data_file_path, dtype=np.int16)
            data_size = memmap_data.size

            test_n_channels = np.arange(num_channels, num_channels+31)
            remainders = np.remainder(data_size, test_n_channels)

            possible_results = test_n_channels[np.where(remainders == 0)]

            del memmap_data

            if possible_results.size == 0:
                return num_channels

            else:
                result = possible_results[0]
                text_message = f"The correct number of channels has been estimated to be {possible_results[0]}."
                if possible_results.size > 1:
                    text_message += f" Other possibilities could be {possible_results[1:]}"

                logger.info(text_message)

                return result

        else:
            return num_channels

    def preapre_for_new_context(self):
        pass

    def reset(self):
        self.data_file_path_input.clear()
        self.working_directory_input.clear()
        self.results_directory_input.clear()
        self.probe_layout_selector.setCurrentIndex(0)
        self.set_default_field_values(None)
        self.disable_preview_probe()
        self.disable_load()

