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

        self.central_channel = 0

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
            assert context is not None

        raw_data = context.raw_data

        single_view = self.traces_view_button.isChecked() ^ self.colormap_view_button.isChecked()

        self.plot_item.clear()

        if single_view:
            if self.traces_view_button.isChecked():
                if self.raw_button.isChecked():
                    raw_traces = raw_data._mmaps[0].T
                    for i in range(self.central_channel, self.central_channel+32):
                        data_item = pg.PlotDataItem(raw_traces[i, :3000] + 200*i, pen=pg.mkPen(color='w', width=1))
                        self.plot_item.addItem(data_item)

            if self.colormap_view_button.isChecked():
                self.enforce_single_mode()

                if self.raw_button.isChecked():
                    raw_traces = raw_data._mmaps[0].T
                    image_item = pg.ImageItem()
                    image_item.setImage(raw_traces[:, :3000].T, autoLevels=True)
                    self.plot_item.addItem(image_item)
        else:
            print("Invalid option!")


class Heatmap(pg.ImageItem):
    def __init__(self, image=None):

        if image is not None:
            self.image = image
        else:
            self.image = np.zeros((500, 500))

        pg.ImageItem.__init__(self, self.image)

    def update_image(self, image, image_rect):
        self.image[image_rect.y(): image_rect.y()+image.shape[0], image_rect.x(): image_rect.x()+image.shape[1]] = image
        self.render()
        self.update()
