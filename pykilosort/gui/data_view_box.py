from PyQt5 import QtWidgets, QtGui
import pyqtgraph as pg
import numpy as np


class DataViewBox(QtWidgets.QGroupBox):

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)

        self.gui = parent

        self.controls_button = QtWidgets.QPushButton("Controls")

        self.data_view_widget = pg.PlotWidget()
        self.plot_item = self.data_view_widget.getPlotItem()

        self.data_seek_widget = pg.PlotWidget()

        self.traces_view_button = QtWidgets.QPushButton("Traces")
        self.colormap_view_button = QtWidgets.QPushButton("Colormap")
        self.raw_button = QtWidgets.QPushButton("Raw")
        self.whitened_button = QtWidgets.QPushButton("Whitened")
        self.prediction_button = QtWidgets.QPushButton("Prediction")
        self.residual_button = QtWidgets.QPushButton("Residual")

        self.mode_buttons = [self.raw_button, self.whitened_button, self.prediction_button, self.residual_button]
        self.view_buttons = [self.traces_view_button, self.colormap_view_button]

        self.central_channel = 80

        self.traces_view = None
        self.colormap_view = None

        colors = [(240, 228, 66), (0, 0, 0), (86, 180, 233)]
        positions = np.linspace(0.0, 1.0, 3)
        color_map = pg.ColorMap(pos=positions, color=colors)
        self.lookup_table = color_map.getLookupTable(nPts=4196, alpha=True)

        self.setup()

    def setup(self):
        self.setTitle("Data View")

        layout = QtWidgets.QVBoxLayout()

        controls_button_layout = QtWidgets.QHBoxLayout()
        self.controls_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        controls_button_layout.addWidget(self.controls_button)

        data_view_layout = QtWidgets.QHBoxLayout()
        data_view_layout.addWidget(self.data_view_widget)

        self.data_view_widget.setMenuEnabled(False)
        self.data_view_widget.setMouseEnabled(False, False)
        self.data_view_widget.hideAxis("left")

        self.data_seek_widget.setMenuEnabled(False)
        self.data_seek_widget.hideAxis("left")

        data_controls_layout = QtWidgets.QHBoxLayout()

        self.traces_view_button.setCheckable(True)
        self.colormap_view_button.setCheckable(True)
        self.traces_view_button.setChecked(True)

        self.traces_view_button.clicked.connect(self.toggle_view)
        self.colormap_view_button.clicked.connect(self.toggle_view)

        self.raw_button.setCheckable(True)
        self.raw_button.setStyleSheet("QPushButton {background-color: black; color: white;}")
        self.raw_button.toggled.connect(self.on_raw_button_toggled)
        self.raw_button.setChecked(True)

        self.whitened_button.setCheckable(True)
        self.whitened_button.setStyleSheet("QPushButton {background-color: black; color: white;}")
        self.whitened_button.toggled.connect(self.on_whitened_button_toggled)

        self.prediction_button.setCheckable(True)
        self.prediction_button.setStyleSheet("QPushButton {background-color: black; color: white;}")
        self.prediction_button.toggled.connect(self.on_prediction_button_toggled)

        self.residual_button.setCheckable(True)
        self.residual_button.setStyleSheet("QPushButton {background-color: black; color: white;}")
        self.residual_button.toggled.connect(self.on_residual_button_toggled)

        data_controls_layout.addWidget(self.traces_view_button)
        data_controls_layout.addWidget(self.colormap_view_button)
        data_controls_layout.addStretch(1)
        data_controls_layout.addWidget(self.raw_button)
        data_controls_layout.addWidget(self.whitened_button)
        data_controls_layout.addWidget(self.prediction_button)
        data_controls_layout.addWidget(self.residual_button)

        data_seek_layout = QtWidgets.QHBoxLayout()
        data_seek_layout.addWidget(self.data_seek_widget)

        layout.addLayout(controls_button_layout, 2)
        layout.addLayout(data_view_layout, 80)
        layout.addLayout(data_controls_layout, 3)
        layout.addLayout(data_seek_layout, 15)

        self.setLayout(layout)

    def on_raw_button_toggled(self, state):
        if state:
            self.raw_button.setStyleSheet("QPushButton {background-color: white; color: black;}")
        else:
            self.raw_button.setStyleSheet("QPushButton {background-color: black; color: white;}")

        self.update_plot()

    def on_whitened_button_toggled(self, state):
        if state:
            self.whitened_button.setStyleSheet("QPushButton {background-color: lightblue; color: black;}")
        else:
            self.whitened_button.setStyleSheet("QPushButton {background-color: black; color: white;}")

        self.update_plot()

    def on_prediction_button_toggled(self, state):
        if state:
            self.prediction_button.setStyleSheet("QPushButton {background-color: green; color: black;}")
        else:
            self.prediction_button.setStyleSheet("QPushButton {background-color: black; color: white;}")

        self.update_plot()

    def on_residual_button_toggled(self, state):
        if state:
            self.residual_button.setStyleSheet("QPushButton {background-color: red; color: black;}")
        else:
            self.residual_button.setStyleSheet("QPushButton {background-color: black; color: white;}")

        self.update_plot()

    def enforce_single_mode(self):
        def only_one_true(iterable):
            true_found = False
            for item in iterable:
                if item:
                    if true_found:
                        return False
                    else:
                        true_found = True
            return true_found

        if not only_one_true([self.raw_button.isChecked(), self.prediction_button.isChecked(),
                              self.residual_button.isChecked(), self.whitened_button.isChecked()]):
            for button in self.mode_buttons:
                button.setChecked(False)

            self.raw_button.setChecked(True)

    def toggle_view(self):
        traces_state, colormap_state = self.traces_view_button.isChecked(), self.colormap_view_button.isChecked()

        # if traces_state and not colormap_state:
        #     self.traces_view_button.setChecked(False)
        #     self.colormap_view_button.setChecked(True)
        # elif colormap_state and not traces_state:
        #     self.traces_view_button.setChecked(True)
        #     self.colormap_view_button.setChecked(False)
        # else:
        #     print("Something is wrong!")

        self.update_plot()

    def update_plot(self, context=None):
        if context is None:
            context = self.gui.context

        if context is not None:

            raw_data = context.raw_data

            single_view = self.traces_view_button.isChecked() ^ self.colormap_view_button.isChecked()

            self.plot_item.clear()

            if single_view:
                if self.traces_view_button.isChecked():
                    if self.raw_button.isChecked():
                        raw_traces = raw_data[:3000].T
                        for i in range(self.central_channel, self.central_channel+32):
                            data_item = pg.PlotDataItem(raw_traces[i] + 200*i, pen=pg.mkPen(color='w', width=1))
                            self.plot_item.addItem(data_item)
                            self.data_view_widget.setXRange(0, 3000, padding=0.0)
                            self.data_view_widget.setLimits(xMin=0, xMax=3000, minXRange=0, maxXRange=3000)

                if self.colormap_view_button.isChecked():
                    self.enforce_single_mode()

                    if self.raw_button.isChecked():
                        raw_traces = raw_data[:3000].T
                        image_item = pg.ImageItem(setPxMode=False)
                        image_item.setImage(raw_traces.T, autoLevels=True, autoDownsample=True)
                        image_item.setLookupTable(self.lookup_table)
                        self.plot_item.addItem(image_item)
                        self.data_view_widget.setXRange(0, 3000, padding=0.02)
                        self.data_view_widget.setLimits(xMin=0, xMax=3000, minXRange=0, maxXRange=3000)
            else:
                print("Invalid option!")
