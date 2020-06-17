from PyQt5 import QtWidgets


class RunBox(QtWidgets.QGroupBox):

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.setTitle("Run")

        self.layout = QtWidgets.QGridLayout()

        self.run_all_button = QtWidgets.QPushButton("Run All")
        self.run_all_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.preprocess_button = QtWidgets.QPushButton("Preprocess")
        self.spike_sort_button = QtWidgets.QPushButton("Spikesort")
        self.export_button = QtWidgets.QPushButton("Export for Phy")

        self.layout.addWidget(self.run_all_button, 0, 0, 3, 2)
        self.layout.addWidget(self.preprocess_button, 0, 2, 1, 2)
        self.layout.addWidget(self.spike_sort_button, 1, 2, 1, 2)
        self.layout.addWidget(self.export_button, 2, 2, 1, 2)

        self.setLayout(self.layout)
