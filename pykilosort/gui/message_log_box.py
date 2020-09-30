from pykilosort.gui.logger import XStream
from PyQt5 import QtWidgets, QtCore, QtGui


class MessageLogBox(QtWidgets.QGroupBox):

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.setTitle("Message Log")

        self.layout = QtWidgets.QHBoxLayout()
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.log_box.setFont(QtGui.QFont("Monospace"))
        self.layout.addWidget(self.log_box)

        XStream.stdout().messageWritten.connect(self.update_text)
        XStream.stderr().messageWritten.connect(self.update_text)

        self.setLayout(self.layout)

    @QtCore.pyqtSlot(str)
    def update_text(self, text):
        self.log_box.moveCursor(QtGui.QTextCursor.End)
        self.log_box.appendPlainText(text)
        self.log_box.ensureCursorVisible()
