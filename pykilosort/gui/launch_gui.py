import os
import sys
import pyqtgraph as pg
# TODO: optimize imports before incorporating into codebase

from pykilosort.gui import DataViewBox, ProbeViewBox, SettingsBox, RunBox, MessageLogBox, HeaderBox
from pykilosort.gui import DarkPalette
from PyQt5 import QtGui, QtWidgets, QtCore


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
    kilosort_application.setPalette(DarkPalette())
    kilosort_application.setStyleSheet("QToolTip { color: #aeadac;"
                                       "background-color: #35322f;"
                                       "border: 1px solid #aeadac; }")

    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')
    pg.setConfigOption('useOpenGL', True)

    kilosort_gui = KiloSortGUI()
    kilosort_gui.showMaximized()
    kilosort_gui.show()

    sys.exit(kilosort_application.exec_())
