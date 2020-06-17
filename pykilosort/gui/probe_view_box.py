from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg


class ProbeViewBox(QtWidgets.QGroupBox):

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.setTitle("Probe View")

        self.layout = QtWidgets.QVBoxLayout()

        self.info_message = QtWidgets.QLabel("scroll to zoom, click to view channel,\nright click to disable channel")
        self.info_message.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Black))
        self.info_message.setAlignment(QtCore.Qt.AlignCenter)

        self.probe_view = pg.PlotWidget()

        self.layout.addWidget(self.info_message, 5)
        self.layout.addWidget(self.probe_view, 95)

        self.setLayout(self.layout)
