from pykilosort import __version__
from PyQt5 import QtWidgets, QtGui


class HeaderBox(QtWidgets.QWidget):

    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent=parent)

        self.layout = QtWidgets.QHBoxLayout()

        self.kilosort_text = QtWidgets.QLabel()
        self.kilosort_text.setText(f"Kilosort {__version__[:3]}")
        self.kilosort_text.setFont(QtGui.QFont("Arial", 20, QtGui.QFont.Black))
        self.help_button = QtWidgets.QPushButton("Help")
        self.reset_gui_button = QtWidgets.QPushButton("Reset GUI")

        self.layout.addWidget(self.kilosort_text)
        self.layout.addStretch(0)
        self.layout.addWidget(self.help_button)
        self.layout.addWidget(self.reset_gui_button)

        self.setLayout(self.layout)
