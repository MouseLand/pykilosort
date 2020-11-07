from PyQt5 import QtWidgets, QtCore
from pykilosort.gui.palettes import COLORMAP_COLORS
from pykilosort.gui.sorter import get_predicted_traces, filter_and_whiten
from pykilosort.preprocess import get_whitening_matrix
from pykilosort.gui.minor_gui_elements import controls_popup_text
from pykilosort.gui.logger import setup_logger
import pyqtgraph as pg
import numpy as np

logger = setup_logger(__name__)


class DataViewBox(QtWidgets.QGroupBox):
    channelChanged = QtCore.pyqtSignal()
    modeChanged = QtCore.pyqtSignal(str)

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)

        self.gui = parent

        self.controls_button = QtWidgets.QPushButton("Controls")

        self.data_view_widget = KSPlotWidget(useOpenGL=True)
        self.data_x_axis = self.data_view_widget.getAxis("bottom")
        self.plot_item = self.data_view_widget.getPlotItem()
        self.colormap_image = None

        self.data_seek_widget = pg.PlotWidget(useOpenGL=True)
        self.seek_view_box = self.data_seek_widget.getViewBox()
        self.time_seek = pg.InfiniteLine(pen=pg.mkPen((255, 0, 0, 128)), movable=True, name="indicator")
        self.time_label = pg.TextItem(color=(180, 180, 180))

        self.traces_button = QtWidgets.QPushButton("Traces")
        self.colormap_button = QtWidgets.QPushButton("Colormap")
        self.raw_button = QtWidgets.QPushButton("Raw")
        self.whitened_button = QtWidgets.QPushButton("Whitened")
        self.prediction_button = QtWidgets.QPushButton("Prediction")
        self.residual_button = QtWidgets.QPushButton("Residual")

        self.mode_buttons_group = QtWidgets.QButtonGroup(self)
        self.view_buttons_group = QtWidgets.QButtonGroup(self)

        self.view_buttons = {'raw': self.raw_button,
                             'whitened': self.whitened_button,
                             'prediction': self.prediction_button,
                             'residual': self.residual_button}

        self.mode_buttons = [self.traces_button, self.colormap_button]

        self._view_button_checked_bg_colors = {'raw': 'white',
                                               'whitened': 'lightblue',
                                               'prediction': 'green',
                                               'residual': 'red'}

        self._keys = ['raw', 'whitened', 'prediction', 'residual']

        self.view_buttons_state = {key: False for key in self._keys}

        self.primary_channel = 0
        self.current_time = 0
        self.plot_range = 0.1  # seconds

        self.whitening_matrix = None
        self.whitened_traces = None
        self.prediction_traces = None
        self.residual_traces = None

        self.sorting_status = {'preprocess': False,
                               'spikesort': False,
                               'export': False}

        # traces settings
        self.good_channel_color = (255, 255, 255)
        self.bad_channel_color = (100, 100, 100)
        self.channels_displayed_traces = 32
        self.channels_displayed_colormap = None
        self.data_range = (0, 3000)
        self.seek_range = (0, 100)
        self.scale_factor = 1.0
        self.processed_traces_scaling = 15
        self.traces_curve_color = {'raw': 'w',
                                   'whitened': 'c',
                                   'predicted': 'g',
                                   'residual': 'r'}

        # colormap settings
        self._colors = COLORMAP_COLORS

        self.colormap_min = 0.0
        self.colormap_max = 1.0
        self.lookup_table = self.generate_lookup_table(self.colormap_min, self.colormap_max)

        self.setup()

    def setup(self):
        self.setTitle("Data View")

        layout = QtWidgets.QVBoxLayout()

        controls_button_layout = QtWidgets.QHBoxLayout()
        self.controls_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        controls_button_layout.addWidget(self.controls_button)
        self.controls_button.clicked.connect(self.show_controls_popup)

        data_view_layout = QtWidgets.QHBoxLayout()
        data_view_layout.addWidget(self.data_view_widget)

        self.time_label.setParentItem(self.seek_view_box)
        self.time_label.setPos(0, 0)
        self.data_seek_widget.addItem(self.time_seek)

        self.time_seek.sigPositionChanged.connect(self.update_seek_text)
        self.time_seek.sigPositionChanged.connect(self.update_seek_position)
        # self.time_seek.sigPositionChangeFinished.connect(self.update_seek_position)

        self.data_view_widget.setMenuEnabled(False)
        self.data_view_widget.setMouseEnabled(True)
        self.data_view_widget.mouseEnabled = True
        self.data_view_widget.hideAxis("left")
        self.data_view_widget.sceneObj.sigMouseClicked.connect(self.scene_clicked)

        self.data_view_widget.signalChangeChannel.connect(self.on_wheel_scroll_plus_control)
        self.data_view_widget.signalChangeTimeRange.connect(self.on_wheel_scroll_plus_shift)
        self.data_view_widget.signalChangeScaling.connect(self.on_wheel_scroll_plus_alt)
        self.data_view_widget.signalChangeTimePoint.connect(self.on_wheel_scroll)

        self.data_seek_widget.setMenuEnabled(False)
        self.data_seek_widget.setMouseEnabled(False, False)
        self.data_seek_widget.hideAxis("left")
        self.data_seek_widget.sceneObj.sigMouseClicked.connect(self.seek_clicked)

        data_controls_layout = QtWidgets.QHBoxLayout()

        self.traces_button.setCheckable(True)
        self.colormap_button.setCheckable(True)
        self.colormap_button.setChecked(True)

        for mode_button in self.mode_buttons:
            self.mode_buttons_group.addButton(mode_button)
        self.mode_buttons_group.setExclusive(True)

        self.traces_button.toggled.connect(self.toggle_view)

        for key in self._keys:
            button = self.view_buttons[key]
            button.setCheckable(True)
            button.setStyleSheet("QPushButton {background-color: black; color: white;}")
            button.clicked.connect(self.on_views_clicked)
            self.view_buttons_group.addButton(button)

        self.raw_button.setChecked(True)
        self.raw_button.setStyleSheet("QPushButton {background-color: white; color: black;}")

        self.enable_view_buttons()

        self.view_buttons_group.setExclusive(True)

        self.view_buttons_state = [self.view_buttons[key].isChecked() for key in self._keys]

        data_controls_layout.addWidget(self.traces_button)
        data_controls_layout.addWidget(self.colormap_button)
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

    @QtCore.pyqtSlot()
    def on_views_clicked(self):
        current_state = {key: self.view_buttons[key].isChecked() for key in self._keys}

        if current_state != self.view_buttons_state:
            self.view_buttons_state = current_state

            for view, state in self.view_buttons_state.items():
                button = self.view_buttons[view]
                if state:
                    color = self._view_button_checked_bg_colors[view]
                    button.setStyleSheet(f"QPushButton {{background-color: {color}; color: black;}}")
                else:
                    if button.isEnabled():
                        button.setStyleSheet("QPushButton {background-color: black; color: white;}")
                    else:
                        button.setStyleSheet("QPushButton {background-color: black; color: gray;}")

            self.update_plot()

    @QtCore.pyqtSlot(int)
    def on_wheel_scroll(self, direction):
        if self.gui.context is not None:
            self.shift_current_time(direction)

    @QtCore.pyqtSlot(int)
    def on_wheel_scroll_plus_control(self, direction):
        if self.gui.context is not None:
            if self.traces_button.isChecked():
                self.shift_primary_channel(direction)
            else:
                self.change_displayed_channel_count(direction)

    @QtCore.pyqtSlot(int)
    def on_wheel_scroll_plus_shift(self, direction):
        if self.gui.context is not None:
            self.change_plot_range(direction)

    @QtCore.pyqtSlot(int)
    def on_wheel_scroll_plus_alt(self, direction):
        if self.gui.context is not None:
            self.change_plot_scaling(direction)

    def toggle_view(self, toggled):
        if toggled:
            self.modeChanged.emit("traces")
            self.view_buttons_group.setExclusive(False)
        else:
            self.modeChanged.emit("colormap")
            for button in self.view_buttons.values():
                if button.isChecked():
                    button.setChecked(False)
            self.view_buttons['raw'].setChecked(True)
            self.view_buttons_group.setExclusive(True)

        self.update_plot()

    def change_primary_channel(self, channel):
        self.primary_channel = channel
        self.channelChanged.emit()
        self.update_plot()

    def shift_primary_channel(self, shift):
        primary_channel = self.primary_channel
        primary_channel += shift * 5
        total_channels = self.get_total_channels()
        if (0 <= primary_channel < total_channels) and total_channels is not None:
            self.primary_channel = primary_channel
            self.channelChanged.emit()
            self.update_plot()

    def get_currently_displayed_channel_count(self):
        if self.traces_button.isChecked():
            return self.channels_displayed_traces
        else:
            count = self.channels_displayed_colormap
            if count is None:
                count = self.get_total_channels()
            return count

    def set_currently_displayed_channel_count(self, count):
        if self.traces_button.isChecked():
            self.channels_displayed_traces = count
        else:
            self.channels_displayed_colormap = count

    def get_total_channels(self):
        return self.gui.probe_view_box.total_channels

    def change_displayed_channel_count(self, direction):
        total_channels = self.get_total_channels()
        current_count = self.get_currently_displayed_channel_count()

        new_count = current_count + (direction * 5)
        if 0 < new_count <= total_channels:
            self.set_currently_displayed_channel_count(new_count)
            self.channelChanged.emit()
            self.update_plot()
        elif new_count <= 0 and current_count != 1:
            self.set_currently_displayed_channel_count(1)
            self.channelChanged.emit()
            self.update_plot()
        elif new_count > total_channels:
            self.set_currently_displayed_channel_count(total_channels)
            self.channelChanged.emit()
            self.update_plot()

    def shift_current_time(self, direction):
        time_shift = direction * self.plot_range / 2  # seconds
        current_time = self.current_time
        new_time = current_time + time_shift
        seek_range_min = self.seek_range[0]
        seek_range_max = self.seek_range[1]
        if seek_range_min <= new_time <= seek_range_max:
            self.time_seek.setPos(new_time)

    def change_plot_range(self, direction):
        plot_range = self.plot_range + 0.1 * direction
        if 0.01 < plot_range < 2.0:
            self.plot_range = plot_range
            self.clear_cached_traces()
            self.update_plot()

    def change_plot_scaling(self, direction):
        if self.traces_button.isChecked():
            scale_factor = self.scale_factor * (1.1 ** direction)
            if 0.1 < scale_factor < 10.0:
                self.scale_factor = scale_factor

                self.update_plot()

        if self.colormap_button.isChecked():
            colormap_min = self.colormap_min + (direction * 0.05)
            colormap_max = self.colormap_max - (direction * 0.05)
            if 0.0 <= colormap_min < colormap_max <= 1.0:
                self.colormap_min = colormap_min
                self.colormap_max = colormap_max
                self.lookup_table = self.generate_lookup_table(self.colormap_min, self.colormap_max)

                self.update_plot()

    def change_channel_display(self, direction):
        if self.traces_button.isChecked():
            self.shift_primary_channel(direction)
        else:
            self.change_displayed_channel_count(direction)

    def scene_clicked(self, ev):
        if self.gui.context is not None:
            if self.traces_button.isChecked():
                x_pos = ev.pos().x()
            else:
                x_pos = self.colormap_image.mapFromScene(ev.pos()).x()
            range_min = self.data_range[0]
            range_max = self.data_range[1]
            fraction = (x_pos - range_min) / range_max
            if fraction > 0.5:
                self.shift_current_time(direction=1)
            else:
                self.shift_current_time(direction=-1)

    def seek_clicked(self, ev):
        if self.gui.context is not None:
            new_time = self.seek_view_box.mapSceneToView(ev.pos()).x()
            seek_range_min = self.seek_range[0]
            seek_range_max = self.seek_range[1]
            if seek_range_min <= new_time <= seek_range_max:
                self.time_seek.setPos(new_time)

    def setup_seek(self, context):
        raw_data = context.raw_data
        sample_rate = raw_data.sample_rate

        timepoints = raw_data.shape[0]
        max_time = timepoints / sample_rate

        self.data_seek_widget.setXRange(min=0, max=max_time, padding=0.02)
        self.time_seek.setPos(0)
        self.time_seek.setBounds((0, max_time))
        self.seek_range = (0, max_time)

    def update_seek_text(self, seek):
        position = seek.pos()[0]
        self.time_label.setText("t={0:.2f} s".format(position))

    def update_seek_position(self, seek):
        position = seek.pos()[0]
        self.current_time = position
        self.clear_cached_traces()
        self.update_plot()

    def change_sorting_status(self, status_dict):
        self.sorting_status = status_dict
        self.enable_view_buttons()

    def enable_view_buttons(self):
        if self.colormap_button.isChecked():
            if self.prediction_button.isChecked() or self.residual_button.isChecked():
                self.raw_button.click()
        else:
            if self.prediction_button.isChecked():
                self.prediction_button.click()
            if self.residual_button.isChecked():
                self.residual_button.click()

        if self.sorting_status['preprocess'] and self.sorting_status['spikesort']:
            self.prediction_button.setEnabled(True)
            self.prediction_button.setStyleSheet("QPushButton {background-color: black; color: white;}")
            self.residual_button.setEnabled(True)
            self.residual_button.setStyleSheet("QPushButton {background-color: black; color: white;}")
        else:
            self.prediction_button.setDisabled(True)
            self.prediction_button.setStyleSheet("QPushButton {background-color: black; color: gray;}")
            self.residual_button.setDisabled(True)
            self.residual_button.setStyleSheet("QPushButton {background-color: black; color: gray;}")

    def show_controls_popup(self):
        QtWidgets.QMessageBox.information(self, "Controls", controls_popup_text,
                                          QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def reset(self):
        self.plot_item.clear()
        self.clear_cached_traces()
        self.clear_cached_whitening_matrix()

    def prepare_for_new_context(self):
        self.plot_item.clear()
        self.clear_cached_traces()
        self.clear_cached_whitening_matrix()

    def clear_cached_traces(self):
        self.whitened_traces = None
        self.residual_traces = None
        self.prediction_traces = None

    def clear_cached_whitening_matrix(self):
        self.whitening_matrix = None

    def generate_lookup_table(self, colormap_min, colormap_max, num_points=8192):
        assert colormap_min >= 0.0 and colormap_max <= 1.0
        positions = np.linspace(colormap_min, colormap_max, len(self._colors))
        color_map = pg.ColorMap(pos=positions, color=self._colors)
        return color_map.getLookupTable(nPts=num_points)

    def add_curve_to_plot(self, trace, color, label):
        curve = pg.PlotCurveItem(parent=self.plot_item, clickable=True,
                                 pen=pg.mkPen(color=color, width=1))

        curve.label = label
        curve.setData(trace * self.scale_factor + 200 * label)
        self.plot_item.addItem(curve)

    def add_image_to_plot(self, raw_traces, level_min, level_max):
        image_item = pg.ImageItem(setPxMode=False)
        image_item.setImage(raw_traces, autoLevels=False, lut=self.lookup_table,
                            levels=(level_min, level_max), autoDownsample=True)
        self.colormap_image = image_item
        self.plot_item.addItem(image_item)

    def get_whitened_traces(self, raw_data, raw_traces, intermediate, params, probe, nSkipCov=None):
        if "Wrot" in intermediate and self.whitening_matrix is None:
            self.whitening_matrix = intermediate.Wrot

        elif self.whitening_matrix is None:
            self.whitening_matrix = get_whitening_matrix(raw_data=raw_data,
                                                         params=params,
                                                         probe=probe,
                                                         nSkipCov=nSkipCov)

        whitened_traces = filter_and_whiten(raw_traces=raw_traces,
                                            params=params,
                                            probe=probe,
                                            whitening_matrix=self.whitening_matrix)

        return whitened_traces

    def update_plot(self, context=None):
        if context is None:
            context = self.gui.context

        if context is not None:
            params = context.params
            probe = context.probe
            raw_data = context.raw_data
            intermediate = context.intermediate
            good_channels = intermediate.igood.ravel()

            sample_rate = raw_data.sample_rate

            start_time = int(self.current_time * sample_rate)
            time_range = int(self.plot_range * sample_rate)
            end_time = start_time + time_range

            self.plot_item.clear()
            self.colormap_image = None

            if self.traces_button.isChecked():
                raw_traces = raw_data[start_time:end_time]

                if self.raw_button.isChecked():
                    for i in range(self.primary_channel + self.channels_displayed_traces, self.primary_channel, -1):
                        try:
                            color = self.traces_curve_color['raw'] if good_channels[i] else self.bad_channel_color
                            self.add_curve_to_plot(raw_traces.T[i], color, i)
                        except IndexError:
                            continue

                if self.whitened_button.isChecked():
                    if self.whitened_traces is None:
                        whitened_traces = self.get_whitened_traces(raw_data=raw_data,
                                                                   raw_traces=raw_traces,
                                                                   intermediate=intermediate,
                                                                   params=params,
                                                                   probe=probe,
                                                                   nSkipCov=100)

                        self.whitened_traces = whitened_traces
                    else:
                        whitened_traces = self.whitened_traces
                    for i in range(self.primary_channel + self.channels_displayed_traces, self.primary_channel, -1):
                        try:
                            color = self.traces_curve_color['whitened'] if good_channels[i] else self.bad_channel_color
                            self.add_curve_to_plot(whitened_traces.T[i] * self.processed_traces_scaling, color, i)
                        except IndexError:
                            continue

                if self.prediction_button.isChecked():
                    if self.prediction_traces is None:
                        prediction_traces = get_predicted_traces(matrix_U=intermediate.U_s,
                                                                 matrix_W=intermediate.Wphy,
                                                                 sorting_result=intermediate.st3,
                                                                 time_limits=(start_time, end_time))
                        self.prediction_traces = prediction_traces
                    else:
                        prediction_traces = self.prediction_traces

                    for i in range(self.primary_channel + self.channels_displayed_traces, self.primary_channel, -1):
                        try:
                            color = self.traces_curve_color['predicted'] if good_channels[i] \
                                else self.bad_channel_color
                            self.add_curve_to_plot(prediction_traces.T[i] * self.processed_traces_scaling, color, i)
                        except IndexError:
                            continue

                if self.residual_button.isChecked():
                    if self.residual_traces is None:
                        if self.whitened_traces is None:
                            whitened_traces = self.get_whitened_traces(raw_data=raw_data,
                                                                       raw_traces=raw_traces,
                                                                       intermediate=intermediate,
                                                                       params=params,
                                                                       probe=probe,
                                                                       nSkipCov=100)

                            self.whitened_traces = whitened_traces

                        else:
                            whitened_traces = self.whitened_traces

                        if self.prediction_traces is None:
                            prediction_traces = get_predicted_traces(matrix_U=intermediate.U_s,
                                                                     matrix_W=intermediate.Wphy,
                                                                     sorting_result=intermediate.st3,
                                                                     time_limits=(start_time, end_time))

                            self.prediction_traces = prediction_traces

                        else:
                            prediction_traces = self.prediction_traces

                        residual_traces = whitened_traces - prediction_traces

                        self.residual_traces = residual_traces
                    else:
                        residual_traces = self.residual_traces

                    for i in range(self.primary_channel + self.channels_displayed_traces, self.primary_channel, -1):
                        try:
                            color = self.traces_curve_color['residual'] if good_channels[i] else self.bad_channel_color
                            self.add_curve_to_plot(residual_traces.T[i] * self.processed_traces_scaling, color, i)
                        except IndexError:
                            continue

            if self.colormap_button.isChecked():
                raw_traces = raw_data[start_time:end_time]
                start_channel = self.primary_channel
                displayed_channels = self.channels_displayed_colormap
                if displayed_channels is None:
                    displayed_channels = self.get_total_channels()
                end_channel = start_channel + displayed_channels

                if self.raw_button.isChecked():
                    colormap_min, colormap_max = -32.0, 32.0
                    self.add_image_to_plot(raw_traces[:, start_channel:end_channel], colormap_min, colormap_max)

                elif self.whitened_button.isChecked():
                    if self.whitened_traces is None:
                        whitened_traces = self.get_whitened_traces(raw_data=raw_data,
                                                                   raw_traces=raw_traces,
                                                                   intermediate=intermediate,
                                                                   params=params,
                                                                   probe=probe,
                                                                   nSkipCov=100)

                        self.whitened_traces = whitened_traces
                    else:
                        whitened_traces = self.whitened_traces
                    colormap_min, colormap_max = -4.0, 4.0
                    self.add_image_to_plot(whitened_traces[:, start_channel:end_channel], colormap_min, colormap_max)

                elif self.prediction_button.isChecked():
                    if self.prediction_traces is None:
                        prediction_traces = get_predicted_traces(matrix_U=intermediate.U_s,
                                                                 matrix_W=intermediate.Wphy,
                                                                 sorting_result=intermediate.st3,
                                                                 time_limits=(start_time, end_time))
                        self.prediction_traces = prediction_traces
                    else:
                        prediction_traces = self.prediction_traces
                    colormap_min, colormap_max = -4.0, 4.0
                    self.add_image_to_plot(prediction_traces[:, start_channel:end_channel], colormap_min, colormap_max)

                elif self.residual_button.isChecked():
                    if self.residual_traces is None:
                        if self.whitened_traces is None:
                            whitened_traces = self.get_whitened_traces(raw_data=raw_data,
                                                                       raw_traces=raw_traces,
                                                                       intermediate=intermediate,
                                                                       params=params,
                                                                       probe=probe,
                                                                       nSkipCov=100)

                            self.whitened_traces = whitened_traces

                        else:
                            whitened_traces = self.whitened_traces

                        if self.prediction_traces is None:
                            prediction_traces = get_predicted_traces(matrix_U=intermediate.U_s,
                                                                     matrix_W=intermediate.Wphy,
                                                                     sorting_result=intermediate.st3,
                                                                     time_limits=(start_time, end_time))

                            self.prediction_traces = prediction_traces

                        else:
                            prediction_traces = self.prediction_traces

                        residual_traces = whitened_traces - prediction_traces

                        self.residual_traces = residual_traces
                    else:
                        residual_traces = self.residual_traces
                    colormap_min, colormap_max = -4.0, 4.0
                    self.add_image_to_plot(residual_traces[:, start_channel:end_channel], colormap_min, colormap_max)

            self.data_view_widget.setXRange(0, time_range, padding=0.0)
            self.data_view_widget.setLimits(xMin=0, xMax=time_range)
            self.data_x_axis.setTicks([[(pos, f"{(start_time + pos) / sample_rate:.3f}")
                                        for pos in np.linspace(0, time_range, 20)]])


class KSPlotWidget(pg.PlotWidget):
    signalChangeTimePoint = QtCore.pyqtSignal(int)
    signalChangeChannel = QtCore.pyqtSignal(int)
    signalChangeTimeRange = QtCore.pyqtSignal(int)
    signalChangeScaling = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super(KSPlotWidget, self).__init__(*args, **kwargs)

    def wheelEvent(self, ev):
        # QtWidgets.QGraphicsView.wheelEvent(self, ev)
        # if not self.mouseEnabled:
        #     ev.ignore()
        #     return

        delta = ev.angleDelta().y()
        if delta == 0:
            delta = ev.angleDelta().x()

        direction = delta / np.abs(delta)
        modifiers = QtWidgets.QApplication.keyboardModifiers()

        if modifiers == QtCore.Qt.ControlModifier:
            # control pressed while scrolling
            self.signalChangeChannel.emit(direction)

        elif modifiers == QtCore.Qt.AltModifier:
            # alt pressed while scrolling
            self.signalChangeScaling.emit(direction)

        elif modifiers == QtCore.Qt.ShiftModifier:
            # shift pressed while scrolling
            self.signalChangeTimeRange.emit(direction)

        else:
            # other key / no key pressed while scrolling
            self.signalChangeTimePoint.emit(direction)

        ev.accept()
        return

    def mouseMoveEvent(self, ev):
        pass
