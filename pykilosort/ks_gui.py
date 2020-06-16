import os
import sys
import pyqtgraph as pg
# TODO: optimize imports before incorporating into codebase
from pykilosort.default_params import default_params
from PyQt5 import QtGui, QtWidgets, QtCore


class HeaderBox(QtWidgets.QWidget):

    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent=parent)

        self.layout = QtWidgets.QHBoxLayout()

        self.kilosort_text = QtWidgets.QLabel()
        self.kilosort_text.setText("KiloSort")
        self.kilosort_text.setFont(QtGui.QFont("Arial", 20, QtGui.QFont.Black))
        self.help_button = QtWidgets.QPushButton("Help")
        self.reset_gui_button = QtWidgets.QPushButton("Reset GUI")

        self.layout.addWidget(self.kilosort_text)
        self.layout.addStretch(0)
        self.layout.addWidget(self.help_button)
        self.layout.addWidget(self.reset_gui_button)

        self.setLayout(self.layout)


class SettingsBox(QtWidgets.QGroupBox):

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.setTitle("Settings")

        self.layout = QtWidgets.QVBoxLayout()

        self.select_data_file_layout = QtWidgets.QHBoxLayout()
        self.select_data_file = QtWidgets.QPushButton("Select Data File")
        self.data_file_path = QtWidgets.QLineEdit("")
        self.select_data_file_layout.addWidget(self.select_data_file, 70)
        self.select_data_file_layout.addWidget(self.data_file_path, 30)
        self.select_data_file.clicked.connect(self.on_select_data_file_clicked)

        self.select_working_directory_layout = QtWidgets.QHBoxLayout()
        self.select_working_directory = QtWidgets.QPushButton("Select Working Directory")
        self.working_directory = QtWidgets.QLineEdit("")
        self.select_working_directory_layout.addWidget(self.select_working_directory, 70)
        self.select_working_directory_layout.addWidget(self.working_directory, 30)
        self.select_working_directory.clicked.connect(self.on_select_working_dir_clicked)

        self.select_results_directory_layout = QtWidgets.QHBoxLayout()
        self.select_results_directory = QtWidgets.QPushButton("Select Results Directory")
        self.results_directory = QtWidgets.QLineEdit("")
        self.select_results_directory_layout.addWidget(self.select_results_directory, 70)
        self.select_results_directory_layout.addWidget(self.results_directory, 30)
        self.select_results_directory.clicked.connect(self.on_select_results_dir_clicked)

        self.probe_layout_layout = QtWidgets.QHBoxLayout()
        self.probe_layout_text = QtWidgets.QLabel("Select Probe Layout")
        self.probe_layout_selector = QtWidgets.QComboBox()
        self.probe_layout_layout.addWidget(self.probe_layout_text, 70)
        self.probe_layout_layout.addWidget(self.probe_layout_selector, 30)
        self.probe_layout_selector.currentIndexChanged.connect(self.on_probe_layout_selected)

        self.num_channels_layout = QtWidgets.QHBoxLayout()
        self.num_channels_text = QtWidgets.QLabel("Number of Channels")
        self.num_channels = QtWidgets.QLineEdit()
        self.num_channels_layout.addWidget(self.num_channels_text, 70)
        self.num_channels_layout.addWidget(self.num_channels, 30)
        self.num_channels.textEdited.connect(self.on_number_of_channels_changed)

        self.time_range_layout = QtWidgets.QHBoxLayout()
        self.time_range_text = QtWidgets.QLabel("Time Range (in seconds)")
        self.time_range_min = QtWidgets.QLineEdit(str(0))
        self.time_range_max = QtWidgets.QLineEdit("inf")
        self.time_range_layout.addWidget(self.time_range_text, 70)
        self.time_range_layout.addWidget(self.time_range_min, 15)
        self.time_range_layout.addWidget(self.time_range_max, 15)
        self.time_range_min.textEdited.connect(self.on_time_range_changed)
        self.time_range_max.textEdited.connect(self.on_time_range_changed)

        self.min_firing_rate_layout = QtWidgets.QHBoxLayout()
        self.min_firing_rate_text = QtWidgets.QLabel("Min. Firing Rate/Channel\n(0 includes all channels)")
        self.min_firing_rate = QtWidgets.QLineEdit(str(default_params.minfr_goodchannels))
        self.min_firing_rate_layout.addWidget(self.min_firing_rate_text, 70)
        self.min_firing_rate_layout.addWidget(self.min_firing_rate, 30)
        self.min_firing_rate.textEdited.connect(self.on_min_firing_rate_changed)

        self.threshold_layout = QtWidgets.QHBoxLayout()
        self.threshold_text = QtWidgets.QLabel("Threshold")
        self.threshold_upper = QtWidgets.QLineEdit(str(default_params.Th[0]))
        self.threshold_lower = QtWidgets.QLineEdit(str(default_params.Th[1]))
        self.threshold_layout.addWidget(self.threshold_text, 70)
        self.threshold_layout.addWidget(self.threshold_lower, 15)
        self.threshold_layout.addWidget(self.threshold_upper, 15)
        self.threshold_upper.textEdited.connect(self.on_thresholds_changed)
        self.threshold_lower.textEdited.connect(self.on_thresholds_changed)

        self.lambda_layout = QtWidgets.QHBoxLayout()
        self.lambda_text = QtWidgets.QLabel("Lambda")
        self.lambda_value = QtWidgets.QLineEdit(str(default_params.lam))
        self.lambda_layout.addWidget(self.lambda_text, 70)
        self.lambda_layout.addWidget(self.lambda_value, 30)
        self.lambda_value.textEdited.connect(self.on_lambda_changed)

        self.auc_splits_layout = QtWidgets.QHBoxLayout()
        self.auc_splits_text = QtWidgets.QLabel("AUC for Splits")
        self.auc_splits = QtWidgets.QLineEdit(str(default_params.AUCsplit))
        self.auc_splits_layout.addWidget(self.auc_splits_text, 70)
        self.auc_splits_layout.addWidget(self.auc_splits, 30)
        self.auc_splits.textEdited.connect(self.on_auc_splits_changed)

        self.advanced_options_layout = QtWidgets.QHBoxLayout()
        self.advanced_options_button = QtWidgets.QPushButton("Advanced Options")
        self.error_label = QtWidgets.QLabel("")
        self.error_label.setText("Invalid inputs!")
        self.error_label.setWordWrap(True)
        error_label_palette = self.error_label.palette()
        error_label_palette.setColor(QtGui.QPalette.Foreground, QtGui.QColor("red"))
        self.error_label.setPalette(error_label_palette)
        self.advanced_options_layout.addWidget(self.error_label)
        self.advanced_options_layout.addStretch(0)
        self.advanced_options_layout.addWidget(self.advanced_options_button)
        self.advanced_options_button.clicked.connect(self.on_advanced_options_clicked)
        self.error_label.hide()

        self.layout.addLayout(self.select_data_file_layout)
        self.layout.addLayout(self.select_working_directory_layout)
        self.layout.addLayout(self.select_results_directory_layout)
        self.layout.addLayout(self.probe_layout_layout)
        self.layout.addLayout(self.num_channels_layout)
        self.layout.addLayout(self.time_range_layout)
        self.layout.addLayout(self.min_firing_rate_layout)
        self.layout.addLayout(self.threshold_layout)
        self.layout.addLayout(self.auc_splits_layout)
        self.layout.addLayout(self.advanced_options_layout)

        self.setLayout(self.layout)

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


class ProbeViewBox(QtWidgets.QGroupBox):

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.setTitle("Probe View")

        self.layout = QtWidgets.QVBoxLayout()

        self.info_message = QtWidgets.QLabel("scroll to zoom, click to view channel,\nright click to disable channel")
        self.info_message.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Black))
        self.info_message.setAlignment(QtCore.Qt.AlignCenter)

        self.probe_view = pg.PlotWidget(background='w')

        self.layout.addWidget(self.info_message, 5)
        self.layout.addWidget(self.probe_view, 95)

        self.setLayout(self.layout)


class DataViewBox(QtWidgets.QGroupBox):

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.setTitle("Data View")

        self.layout = QtWidgets.QVBoxLayout()

        self.controls_button_layout = QtWidgets.QHBoxLayout()
        self.controls_button = QtWidgets.QPushButton("Controls")
        self.controls_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.controls_button_layout.addWidget(self.controls_button)

        self.data_view_layout = QtWidgets.QHBoxLayout()
        self.data_view_widget = pg.PlotWidget(background='w')
        self.data_view_layout.addWidget(self.data_view_widget)

        self.data_seek_layout = QtWidgets.QHBoxLayout()
        self.data_seek_widget = pg.PlotWidget(background='w')
        self.data_seek_layout.addWidget(self.data_seek_widget)

        self.layout.addLayout(self.controls_button_layout, 3)
        self.layout.addLayout(self.data_view_layout, 82)
        self.layout.addLayout(self.data_seek_layout, 15)

        self.setLayout(self.layout)


class RunBox(QtWidgets.QGroupBox):

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.setTitle("Run")

        self.layout = QtWidgets.QGridLayout()

        self.run_all_button = QtWidgets.QPushButton("Run All")
        self.run_all_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.preprocess_button = QtWidgets.QPushButton("Preprocess")
        self.spike_sort_button = QtWidgets.QPushButton("Spikesort")
        self.export_button = QtWidgets.QPushButton("Export for Phy")

        self.layout.addWidget(self.run_all_button, 0, 0, 3, 2)
        self.layout.addWidget(self.preprocess_button, 0, 2, 1, 2)
        self.layout.addWidget(self.spike_sort_button, 1, 2, 1, 2)
        self.layout.addWidget(self.export_button, 2, 2, 1, 2)

        self.setLayout(self.layout)


class MessageLogBox(QtWidgets.QGroupBox):

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.setTitle("Message Log")

        self.layout = QtWidgets.QHBoxLayout()
        self.log_box = QtWidgets.QPlainTextEdit()
        self.layout.addWidget(self.log_box)

        self.setLayout(self.layout)


class KiloSortGUI(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)

        self.setWindowTitle("KiloSort2 GUI")
        self.content = QtWidgets.QWidget(self)
        self.content_layout = QtWidgets.QVBoxLayout()

        self.header_box = HeaderBox(self)
        self.content_layout.addWidget(self.header_box, 3)

        self.boxes = QtWidgets.QWidget()
        self.boxes_layout = QtWidgets.QHBoxLayout(self.boxes)
        self.second_boxes_layout = QtWidgets.QVBoxLayout()

        self.settings_box = SettingsBox(self)
        self.probe_view_box = ProbeViewBox(self)
        self.data_view_box = DataViewBox(self)
        self.run_box = RunBox(self)
        self.message_log_box = MessageLogBox(self)

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


if __name__ == "__main__":
    kilosort_application = QtWidgets.QApplication(sys.argv)
    kilosort_application.setStyle("Fusion")

    kilosort_gui = KiloSortGUI()
    kilosort_gui.showMaximized()
    kilosort_gui.show()

    sys.exit(kilosort_application.exec_())
