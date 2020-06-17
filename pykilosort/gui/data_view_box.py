from PyQt5 import QtWidgets
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

        self.data_seek_layout = QtWidgets.QHBoxLayout()
        self.data_seek_widget = pg.PlotWidget()
        self.data_seek_layout.addWidget(self.data_seek_widget)

        self.layout.addLayout(self.controls_button_layout, 3)
        self.layout.addLayout(self.data_view_layout, 82)
        self.layout.addLayout(self.data_seek_layout, 15)

        self.setLayout(self.layout)
