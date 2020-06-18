from PyQt5 import QtWidgets, QtGui
import pyqtgraph as pg


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
        self.data_view_widget = pg.PlotWidget()
        self.data_view_layout.addWidget(self.data_view_widget)

        self.data_controls_layout = QtWidgets.QHBoxLayout()

        self.traces_view_button = QtWidgets.QPushButton("Traces")
        self.traces_view_button.setCheckable(True)

        self.colormap_view_button = QtWidgets.QPushButton("Colormap")
        self.colormap_view_button.setCheckable(True)
        self.colormap_view_button.setChecked(True)

        self.raw_button = QtWidgets.QPushButton("Raw")
        self.raw_button.setCheckable(True)
        self.raw_button.setStyleSheet("QPushButton {background-color: black; color: white;}")
        self.raw_button.toggled.connect(self.on_raw_button_toggled)
        self.raw_button.setChecked(True)

        self.whitened_button = QtWidgets.QPushButton("Whitened")
        self.whitened_button.setCheckable(True)
        self.whitened_button.setStyleSheet("QPushButton {background-color: black; color: white;}")
        self.whitened_button.toggled.connect(self.on_whitened_button_toggled)

        self.prediction_button = QtWidgets.QPushButton("Prediction")
        self.prediction_button.setCheckable(True)
        self.prediction_button.setStyleSheet("QPushButton {background-color: black; color: white;}")
        self.prediction_button.toggled.connect(self.on_prediction_button_toggled)

        self.residual_button = QtWidgets.QPushButton("Residual")
        self.residual_button.setCheckable(True)
        self.residual_button.setStyleSheet("QPushButton {background-color: black; color: white;}")
        self.residual_button.toggled.connect(self.on_residual_button_toggled)

        self.data_controls_layout.addWidget(self.traces_view_button)
        self.data_controls_layout.addWidget(self.colormap_view_button)
        self.data_controls_layout.addStretch(1)
        self.data_controls_layout.addWidget(self.raw_button)
        self.data_controls_layout.addWidget(self.whitened_button)
        self.data_controls_layout.addWidget(self.prediction_button)
        self.data_controls_layout.addWidget(self.residual_button)

        self.data_seek_layout = QtWidgets.QHBoxLayout()
        self.data_seek_widget = pg.PlotWidget()
        self.data_seek_layout.addWidget(self.data_seek_widget)

        self.layout.addLayout(self.controls_button_layout, 2)
        self.layout.addLayout(self.data_view_layout, 80)
        self.layout.addLayout(self.data_controls_layout, 3)
        self.layout.addLayout(self.data_seek_layout, 15)

        self.setLayout(self.layout)

    def on_raw_button_toggled(self, state):
        if state:
            self.raw_button.setStyleSheet("QPushButton {background-color: white; color: black;}")
        else:
            self.raw_button.setStyleSheet("QPushButton {background-color: black; color: white;}")

    def on_whitened_button_toggled(self, state):
        if state:
            self.whitened_button.setStyleSheet("QPushButton {background-color: lightblue; color: black;}")
        else:
            self.whitened_button.setStyleSheet("QPushButton {background-color: black; color: white;}")

    def on_prediction_button_toggled(self, state):
        if state:
            self.prediction_button.setStyleSheet("QPushButton {background-color: green; color: black;}")
        else:
            self.prediction_button.setStyleSheet("QPushButton {background-color: black; color: white;}")

    def on_residual_button_toggled(self, state):
        if state:
            self.residual_button.setStyleSheet("QPushButton {background-color: red; color: black;}")
        else:
            self.residual_button.setStyleSheet("QPushButton {background-color: black; color: white;}")

    def toggle_view(self):
        traces_state, colormap_state = self.traces_view_button.isChecked(), self.colormap_view_button.isChecked()

        if traces_state and not colormap_state:
            self.traces_view_button.setChecked(False)
            self.colormap_view_button.setChecked(True)
        elif colormap_state and not traces_state:
            self.traces_view_button.setChecked(True)
            self.colormap_view_button.setChecked(False)
        else:
            print("Something is wrong!")
