from PyQt5 import QtCore, QtWidgets, QtGui
import numpy as np
import pyqtgraph as pg


class ProbeViewBox(QtWidgets.QGroupBox):

    channelSelected = QtCore.pyqtSignal(int)

    def __init__(self, parent):
        super(ProbeViewBox, self).__init__(parent=parent)
        self.setTitle("Probe View")

        self.gui = parent

        self.probe_view = pg.PlotWidget()

        self.info_message = QtWidgets.QLabel("scroll to zoom, click to view channel,\nright click to disable channel")

        self.setup()

        self.active_layout = None
        self.kcoords = None
        self.xc = None
        self.yc = None
        self.total_channels = None
        self.channel_map = None
        self.channel_map_dict = {}
        self.good_channels = None

        self.configuration = {'active_channel': 'g',
                              'good_channel': 'b',
                              'bad_channel': 'r'}

        self.active_data_view_mode = "colormap"
        self.primary_channel = None
        self.active_channels = []

    def setup(self):
        layout = QtWidgets.QVBoxLayout()

        self.probe_view.hideAxis("left")
        self.probe_view.hideAxis("bottom")
        self.probe_view.setMouseEnabled(False, True)

        self.info_message.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Black))
        self.info_message.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.info_message, 5)
        layout.addWidget(self.probe_view, 95)
        self.setLayout(layout)

    def set_active_channels(self):
        if self.active_data_view_mode == "traces":
            displayed_channels = self.gui.data_view_box.channels_displayed_traces
        else:
            displayed_channels = self.gui.data_view_box.channels_displayed_colormap
            if displayed_channels is None:
                displayed_channels = self.total_channels

        primary_channel = self.primary_channel
        channel_map = np.array(self.channel_map)

        primary_channel_position = int(np.where(channel_map == primary_channel)[0])
        end_channel_position = np.where(channel_map == primary_channel + displayed_channels)[0]
        # prevent the last displayed channel would be set as the end channel in the case that
        # `primary_channel + displayed_channels` exceeds the total number of channels in the channel map
        if end_channel_position.size == 0:
            end_channel_position = np.argmax(channel_map)
        else:
            end_channel_position = int(end_channel_position)
        self.active_channels = channel_map[primary_channel_position:end_channel_position].tolist()

    def set_layout(self, context):
        self.probe_view.clear()
        self.set_active_layout(context.probe, context.intermediate.igood)

        self.update_probe_view()

    def set_active_layout(self, probe, good_channels=None):
        self.active_layout = probe
        self.kcoords = self.active_layout.kcoords
        self.xc, self.yc = self.active_layout.xc, self.active_layout.yc
        self.channel_map_dict = {}
        for ind, (xc, yc) in enumerate(zip(self.xc, self.yc)):
            self.channel_map_dict[(xc, yc)] = ind
        self.total_channels = self.active_layout.NchanTOT
        self.channel_map = self.active_layout.chanMap
        if good_channels is None:
            self.good_channels = np.ones_like(self.channel_map, dtype=bool)
        else:
            self.good_channels = good_channels

    def on_points_clicked(self, points):
        selected_point = points.ptsClicked[0]
        x_pos = int(selected_point.pos().x())
        y_pos = int(selected_point.pos().y())

        index = self.channel_map_dict[(x_pos, y_pos)]
        channel = self.channel_map[index]
        self.channelSelected.emit(channel)

    def synchronize_data_view_mode(self, string):
        old_mode = self.active_data_view_mode
        self.active_data_view_mode = string

        if old_mode != self.active_data_view_mode and self.primary_channel is not None:
            self.probe_view.clear()
            self.update_probe_view()

    def synchronize_primary_channel(self):
        self.primary_channel = self.gui.data_view_box.primary_channel

    def generate_spots_list(self):
        spots = []
        size = 10
        symbol = 's'

        for ind, (x_pos, y_pos) in enumerate(zip(self.xc, self.yc)):
            pos = (x_pos, y_pos)
            good_channel = self.good_channels[ind]
            is_active = np.isin(ind, self.active_channels)
            if not good_channel:
                color = self.configuration['bad_channel']
            elif good_channel and is_active:
                color = self.configuration['active_channel']
            elif good_channel and not is_active:
                color = self.configuration['good_channel']
            else:
                # TODO: logger.error
                print("Logical error!")
            pen = pg.mkPen(0.5)
            brush = pg.mkBrush(color)
            spots.append(dict(pos=pos, size=size, pen=pen, brush=brush, symbol=symbol))

        return spots

    @QtCore.pyqtSlot()
    def update_probe_view(self):
        self.synchronize_primary_channel()
        self.set_active_channels()
        self.create_plot()

    @QtCore.pyqtSlot(object)
    def preview_probe(self, probe):
        self.probe_view.clear()
        self.set_active_layout(probe)
        self.create_plot(connect=False)

    def create_plot(self, connect=True):
        spots = self.generate_spots_list()

        scatter_plot = pg.ScatterPlotItem(spots)
        if connect:
            scatter_plot.sigClicked.connect(self.on_points_clicked)
        self.probe_view.addItem(scatter_plot)

    def reset(self):
        self.clear_plot()
        self.reset_current_probe_layout()
        self.reset_active_data_view_mode()
        self.primary_channel = None

    def reset_active_data_view_mode(self):
        self.active_data_view_mode = "colormap"

    def reset_current_probe_layout(self):
        self.active_layout = None
        self.kcoords = None
        self.xc = None
        self.yc = None
        self.total_channels = None
        self.channel_map = None
        self.channel_map_dict = {}
        self.good_channels = None
        self.active_channels = []

    def prepare_for_new_context(self):
        self.clear_plot()
        self.reset_current_probe_layout()

    def clear_plot(self):
        self.probe_view.getPlotItem().clear()
