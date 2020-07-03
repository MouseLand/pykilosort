from PyQt5 import QtCore, QtWidgets, QtGui
import numpy as np
import pyqtgraph as pg


class ProbeViewBox(QtWidgets.QGroupBox):

    def __init__(self, parent):
        super(ProbeViewBox, self).__init__(parent=parent)
        self.setTitle("Probe View")

        self.probe_view = pg.PlotWidget()

        self.info_message = QtWidgets.QLabel("scroll to zoom, click to view channel,\nright click to disable channel")

        self.setup()

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

    def set_layout(self, probe_layout):
        self.probe_view.clear()

        scatter_plot = pg.ScatterPlotItem(x=probe_layout.xc, y=probe_layout.yc, symbol="s")
        self.probe_view.addItem(scatter_plot)

    def update_channel(self, channel):
        pass


def debug_trace():
    """Set a tracepoint in the Python debugger that works with Qt"""
    from PyQt5.QtCore import pyqtRemoveInputHook

    from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace()
