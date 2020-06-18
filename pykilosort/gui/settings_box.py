import os
from PyQt5 import QtWidgets, QtGui

from pykilosort.default_params import default_params


class SettingsBox(QtWidgets.QGroupBox):

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)

        self.select_data_file = QtWidgets.QPushButton("Select Data File")
        self.data_file_path = QtWidgets.QLineEdit("")

        self.select_working_directory = QtWidgets.QPushButton("Select Working Directory")
        self.working_directory = QtWidgets.QLineEdit("")

        self.select_results_directory = QtWidgets.QPushButton("Select Results Directory")
        self.results_directory = QtWidgets.QLineEdit("")

        self.probe_layout_text = QtWidgets.QLabel("Select Probe Layout")
        self.probe_layout_selector = QtWidgets.QComboBox()

        self.num_channels_text = QtWidgets.QLabel("Number of Channels")
        self.num_channels = QtWidgets.QLineEdit()

        self.time_range_text = QtWidgets.QLabel("Time Range (in seconds)")
        self.time_range_min = QtWidgets.QLineEdit(str(0))
        self.time_range_max = QtWidgets.QLineEdit("inf")

        self.min_firing_rate_text = QtWidgets.QLabel("Min. Firing Rate/Channel\n(0 includes all channels)")
        self.min_firing_rate = QtWidgets.QLineEdit(str(default_params.minfr_goodchannels))

        self.lambda_text = QtWidgets.QLabel("Lambda")
        self.lambda_value = QtWidgets.QLineEdit(str(default_params.lam))

        self.auc_splits_text = QtWidgets.QLabel("AUC for Splits")
        self.auc_splits = QtWidgets.QLineEdit(str(default_params.AUCsplit))

        self.advanced_options_button = QtWidgets.QPushButton("Advanced Options")
        self.error_label = QtWidgets.QLabel("")
        self.error_label.setText("Invalid inputs!")
        self.error_label.setWordWrap(True)

        self.setup()

    def setup(self):
        self.setTitle("Settings")

        layout = QtWidgets.QVBoxLayout()

        select_data_file_layout = QtWidgets.QHBoxLayout()
        select_data_file_layout.addWidget(self.select_data_file, 70)
        select_data_file_layout.addWidget(self.data_file_path, 30)
        self.select_data_file.clicked.connect(self.on_select_data_file_clicked)
        self.data_file_path.textChanged.connect(self.on_data_file_path_changed)
        self.data_file_path.editingFinished.connect(self.on_data_file_path_changed)

        select_working_directory_layout = QtWidgets.QHBoxLayout()
        select_working_directory_layout.addWidget(self.select_working_directory, 70)
        select_working_directory_layout.addWidget(self.working_directory, 30)
        self.select_working_directory.clicked.connect(self.on_select_working_dir_clicked)
        self.working_directory.textChanged.connect(self.on_file_paths_changed)
        self.working_directory.editingFinished.connect(self.on_file_paths_changed)

        select_results_directory_layout = QtWidgets.QHBoxLayout()
        select_results_directory_layout.addWidget(self.select_results_directory, 70)
        select_results_directory_layout.addWidget(self.results_directory, 30)
        self.select_results_directory.clicked.connect(self.on_select_results_dir_clicked)
        self.results_directory.textChanged.connect(self.on_file_paths_changed)
        self.results_directory.editingFinished.connect(self.on_file_paths_changed)

        probe_layout_layout = QtWidgets.QHBoxLayout()
        probe_layout_layout.addWidget(self.probe_layout_text, 70)
        probe_layout_layout.addWidget(self.probe_layout_selector, 30)
        self.probe_layout_selector.currentIndexChanged.connect(self.on_probe_layout_selected)

        num_channels_layout = QtWidgets.QHBoxLayout()
        num_channels_layout.addWidget(self.num_channels_text, 70)
        num_channels_layout.addWidget(self.num_channels, 30)
        self.num_channels.textEdited.connect(self.on_number_of_channels_changed)

        time_range_layout = QtWidgets.QHBoxLayout()
        time_range_layout.addWidget(self.time_range_text, 70)
        time_range_layout.addWidget(self.time_range_min, 15)
        time_range_layout.addWidget(self.time_range_max, 15)
        self.time_range_min.textEdited.connect(self.on_time_range_changed)
        self.time_range_max.textEdited.connect(self.on_time_range_changed)

        min_firing_rate_layout = QtWidgets.QHBoxLayout()
        min_firing_rate_layout.addWidget(self.min_firing_rate_text, 70)
        min_firing_rate_layout.addWidget(self.min_firing_rate, 30)
        self.min_firing_rate.textEdited.connect(self.on_min_firing_rate_changed)

        threshold_layout = QtWidgets.QHBoxLayout()
        self.threshold_text = QtWidgets.QLabel("Threshold")
        self.threshold_upper = QtWidgets.QLineEdit(str(default_params.Th[0]))
        self.threshold_lower = QtWidgets.QLineEdit(str(default_params.Th[1]))
        threshold_layout.addWidget(self.threshold_text, 70)
        threshold_layout.addWidget(self.threshold_lower, 15)
        threshold_layout.addWidget(self.threshold_upper, 15)
        self.threshold_upper.textEdited.connect(self.on_thresholds_changed)
        self.threshold_lower.textEdited.connect(self.on_thresholds_changed)

        lambda_layout = QtWidgets.QHBoxLayout()
        lambda_layout.addWidget(self.lambda_text, 70)
        lambda_layout.addWidget(self.lambda_value, 30)
        self.lambda_value.textEdited.connect(self.on_lambda_changed)

        auc_splits_layout = QtWidgets.QHBoxLayout()
        auc_splits_layout.addWidget(self.auc_splits_text, 70)
        auc_splits_layout.addWidget(self.auc_splits, 30)
        self.auc_splits.textEdited.connect(self.on_auc_splits_changed)

        advanced_options_layout = QtWidgets.QHBoxLayout()
        error_label_palette = self.error_label.palette()
        error_label_palette.setColor(QtGui.QPalette.Foreground, QtGui.QColor("red"))
        self.error_label.setPalette(error_label_palette)
        advanced_options_layout.addWidget(self.error_label)
        advanced_options_layout.addStretch(0)
        advanced_options_layout.addWidget(self.advanced_options_button)
        self.advanced_options_button.clicked.connect(self.on_advanced_options_clicked)
        self.error_label.hide()

        layout.addLayout(select_data_file_layout)
        layout.addLayout(select_working_directory_layout)
        layout.addLayout(select_results_directory_layout)
        layout.addLayout(probe_layout_layout)
        layout.addLayout(num_channels_layout)
        layout.addLayout(time_range_layout)
        layout.addLayout(min_firing_rate_layout)
        layout.addLayout(threshold_layout)
        layout.addLayout(auc_splits_layout)
        layout.addLayout(advanced_options_layout)

        self.setLayout(layout)

    def on_select_data_file_clicked(self):
        data_file_name, _ = QtWidgets.QFileDialog.getOpenFileName(parent=self,
                                                                  caption="Choose data file to load...",
                                                                  directory=os.getcwd())
        if data_file_name:
            self.data_file_path.setText(data_file_name)
            # TODO: pass onto plotting

    def on_select_working_dir_clicked(self):
        working_dir_name, _ = QtWidgets.QFileDialog.getExistingDirectory(parent=self,
                                                                         caption="Choose working directory...",
                                                                         directory=os.getcwd())
        if working_dir_name:
            self.working_directory.setText(working_dir_name)

    def on_select_results_dir_clicked(self):
        results_dir_name, _ = QtWidgets.QFileDialog.getExistingDirectory(parent=self,
                                                                         caption="Choose results directory...",
                                                                         directory=os.getcwd())

        if results_dir_name:
            self.results_directory.setText(results_dir_name)

    def on_data_file_path_changed(self):
        data_file_path = self.data_file_path.text()
        try:
            assert os.path.exists(data_file_path)

            parent_folder = os.path.dirname(data_file_path)
            self.working_directory.setText(parent_folder)
            self.results_directory.setText(parent_folder)
            self.error_label.hide()
        except AssertionError:
            self.error_label.setText("Please select a valid file path!")
            self.error_label.show()

    def on_file_paths_changed(self):
        pass

    def on_advanced_options_clicked(self):
        advanced_options_dialog = QtWidgets.QMessageBox(self)
        advanced_options_dialog.setIcon(QtWidgets.QMessageBox.Information)

        advanced_options_dialog.setText("This information will be updated soon!")
        advanced_options_dialog.setWindowTitle("Setting advanced options")
        advanced_options_dialog.setStandardButtons(QtWidgets.QMessageBox.Ok)
        advanced_options_dialog.exec_()

    def on_probe_layout_selected(self):
        pass

    def on_number_of_channels_changed(self):
        try:
            number_of_channels = int(self.num_channels.text())
            assert number_of_channels > 0
            self.error_label.hide()
            # TODO: pass onto plotting
            # TODO: specific error messages
        except ValueError:
            self.error_label.setText("Invalid input!\nNo. of channels must be an integer!")
            self.error_label.show()
        except AssertionError:
            self.error_label.setText("Invalid input!\nNo. of channels must be > 0!")
            self.error_label.show()

    def on_time_range_changed(self):
        try:
            time_range_low = float(self.time_range_min.text())
            time_range_high = self.time_range_max.text()
            if not time_range_high == "inf":
                time_range_high = float(time_range_high)
                assert 0 <= time_range_low < time_range_high
            else:
                assert 0 <= time_range_low
            self.error_label.hide()
            # TODO: pass onto plotting
            # TODO: specific error messages
        except ValueError:
            self.error_label.setText("Invalid inputs!\nTime range values must be floats!"
                                     "\n(`inf` accepted as upper limit)")
            self.error_label.show()
        except AssertionError:
            self.error_label.setText("Invalid inputs!\nCheck that 0 <= lower limit < upper limit!")
            self.error_label.show()

    def on_min_firing_rate_changed(self):
        try:
            min_firing_rate = float(self.min_firing_rate.text())
            assert min_firing_rate >= 0
            self.error_label.hide()
            # TODO: pass onto plotting
            # TODO: specific error messages
        except ValueError:
            self.error_label.setText("Invalid input!\nMin. firing rate value must be a float!")
            self.error_label.show()
        except AssertionError:
            self.error_label.setText("Invalid input!\nMin. firing rate must be >= 0.0 Hz!")
            self.error_label.show()

    def on_thresholds_changed(self):
        try:
            threshold_upper = float(self.threshold_upper.text())
            threshold_lower = float(self.threshold_lower.text())
            assert 0 < threshold_lower < threshold_upper
            self.error_label.hide()
            # TODO: pass onto plotting
            # TODO: specific error messages
        except ValueError:
            self.error_label.setText("Invalid inputs!\nThreshold values must be floats!")
            self.error_label.show()
        except AssertionError:
            self.error_label.setText("Invalid inputs!\nCheck that 0 < lower threshold < upper threshold!")
            self.error_label.show()

    def on_lambda_changed(self):
        try:
            lambda_value = float(self.lambda_value.text())
            assert 0 < lambda_value
            self.error_label.hide()
            # TODO: pass onto plotting
            # TODO: specific error messages
        except ValueError:
            self.error_label.setText("Invalid input!\nLambda value must be a float!")
            self.error_label.show()
        except AssertionError:
            self.error_label.setText("Invalid input!\nLambda value must be > 0!")
            self.error_label.show()

    def on_auc_splits_changed(self):
        try:
            auc_split = float(self.auc_splits.text())
            assert 0 <= auc_split <= 1
            self.error_label.hide()
            # TODO: pass onto plotting
            # TODO: specific error messages
        except ValueError:
            self.error_label.setText("Invalid input!\nAUC split value must be a float!")
            self.error_label.show()
        except AssertionError:
            self.error_label.setText("Invalid input!\nCheck that 0 <= AUC split <= 1!")
            self.error_label.show()
