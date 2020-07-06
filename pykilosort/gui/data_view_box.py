from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np


class DataViewBox(QtWidgets.QGroupBox):
    channelChanged = QtCore.pyqtSignal(int)
    modeChanged = QtCore.pyqtSignal(str)

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)

        self.gui = parent

        self.controls_button = QtWidgets.QPushButton("Controls")

        self.data_view_widget = pg.PlotWidget(useOpenGL=True)
        self.plot_item = self.data_view_widget.getPlotItem()

        self.data_seek_widget = pg.PlotWidget(useOpenGL=True)
        self.time_seek = pg.InfiniteLine(pen=pg.mkPen((255, 0, 0, 128)), movable=True, name="indicator")
        self.time_label = pg.TextItem(color=(180, 180, 180))

        self.traces_view_button = QtWidgets.QPushButton("Traces")
        self.colormap_view_button = QtWidgets.QPushButton("Colormap")
        self.raw_button = QtWidgets.QPushButton("Raw")
        self.whitened_button = QtWidgets.QPushButton("Whitened")
        self.prediction_button = QtWidgets.QPushButton("Prediction")
        self.residual_button = QtWidgets.QPushButton("Residual")

        self.mode_buttons_group = QtWidgets.QButtonGroup(self)
        self.view_buttons_group = QtWidgets.QButtonGroup(self)

        self.mode_buttons = [self.raw_button, self.whitened_button, self.prediction_button, self.residual_button]
        self.view_buttons = [self.traces_view_button, self.colormap_view_button]

        self.primary_channel = 0
        self.current_time = 0
        self.plot_range = 0.1  # seconds

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

        data_seek_view_box = self.data_seek_widget.getViewBox()
        self.time_label.setParentItem(data_seek_view_box)
        self.time_label.setPos(0, 0)
        self.data_seek_widget.addItem(self.time_seek)

        self.time_seek.sigPositionChanged.connect(self.update_seek_text)
        self.time_seek.sigPositionChangeFinished.connect(self.update_seek_position)

        self.data_view_widget.setMenuEnabled(False)
        self.data_view_widget.setMouseEnabled(False, True)
        self.data_view_widget.hideAxis("left")

        self.data_seek_widget.setMenuEnabled(False)
        self.data_seek_widget.setMouseEnabled(False, False)
        self.data_seek_widget.hideAxis("left")

        data_controls_layout = QtWidgets.QHBoxLayout()

        self.traces_view_button.setCheckable(True)
        self.colormap_view_button.setCheckable(True)
        self.traces_view_button.setChecked(True)

        self.mode_buttons_group.addButton(self.traces_view_button)
        self.mode_buttons_group.addButton(self.colormap_view_button)
        self.mode_buttons_group.setExclusive(True)

        self.traces_view_button.toggled.connect(self.toggle_view)

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

        self.view_buttons_group.addButton(self.raw_button)
        self.view_buttons_group.addButton(self.whitened_button)
        self.view_buttons_group.addButton(self.prediction_button)
        self.view_buttons_group.addButton(self.residual_button)
        self.view_buttons_group.setExclusive(False)

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
        layout.addLayout(data_view_layout, 85)
        layout.addLayout(data_controls_layout, 3)
        layout.addLayout(data_seek_layout, 10)

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

    def toggle_view(self, toggled):
        if toggled:
            self.modeChanged.emit("traces")
        else:
            self.modeChanged.emit("colormap")

        self.update_plot()

    def change_primary_channel(self, channel):
        self.primary_channel = channel
        self.channelChanged.emit(channel)
        self.update_plot()

    def trace_clicked(self, curve):
        label = curve.label
        self.primary_channel = label

        self.channelChanged.emit(label)
        self.update_plot()

    def set_seek_range(self, context):
        raw_data = context.raw_data
        sample_rate = raw_data.sample_rate

        timepoints = raw_data.shape[0]
        max_time = timepoints/sample_rate

        self.data_seek_widget.setXRange(min=0, max=max_time, padding=0.02)
        self.time_seek.setPos(0)
        self.time_seek.setBounds((0, max_time))

    def update_seek_text(self, seek):
        position = seek.pos()[0]
        self.time_label.setText("t={0:.2f} s".format(position))

    def update_seek_position(self, seek):
        position = seek.pos()[0]
        self.current_time = position
        self.update_plot()

    def update_plot(self, context=None):
        if context is None:
            context = self.gui.context

        if context is not None:

            raw_data = context.raw_data
            sample_rate = raw_data.sample_rate

            start_time = int(self.current_time * sample_rate)
            time_range = int(self.plot_range * sample_rate)
            end_time = start_time + time_range

            self.plot_item.clear()

            if self.traces_view_button.isChecked():

                if self.raw_button.isChecked():
                    raw_traces = raw_data[start_time:end_time].T
                    for i in range(self.primary_channel + 32, self.primary_channel, -1):
                        curve = pg.PlotCurveItem(parent=self.plot_item, clickable=True,
                                                 pen=pg.mkPen(color='w', width=1))
                        curve.label = i
                        try:
                            curve.setData(raw_traces[i] + 200*i)
                            self.plot_item.addItem(curve)
                            curve.sigClicked.connect(self.trace_clicked)
                            self.data_view_widget.setXRange(start_time, end_time, padding=0.0)
                            self.data_view_widget.setLimits(xMin=0, xMax=3000, minXRange=0, maxXRange=3000)
                        except IndexError:
                            continue

            if self.colormap_view_button.isChecked():

                if self.raw_button.isChecked():
                    raw_traces = raw_data[start_time:end_time]
                    image_item = pg.ImageItem(setPxMode=False)

                    image_item.setImage(raw_traces, autoLevels=True, autoDownsample=True)
                    image_item.setLookupTable(self.lookup_table)
                    self.plot_item.addItem(image_item)
                    self.data_view_widget.setXRange(start_time, end_time, padding=0.0)
                    self.data_view_widget.setLimits(xMin=0, xMax=3000, minXRange=0, maxXRange=3000)
