from PyQt5 import QtWidgets


class MessageLogBox(QtWidgets.QGroupBox):

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.setTitle("Message Log")

        self.layout = QtWidgets.QHBoxLayout()
        self.log_box = QtWidgets.QPlainTextEdit()
        self.layout.addWidget(self.log_box)

        self.setLayout(self.layout)
