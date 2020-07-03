import sys
import logging
from PyQt5 import QtWidgets, QtCore, QtGui

logger = logging.getLogger(__name__)


class MessageLogBox(QtWidgets.QGroupBox):

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.setTitle("Message Log")

        self.layout = QtWidgets.QHBoxLayout()
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.layout.addWidget(self.log_box)

        self.setLayout(self.layout)

    def update_text(self, text):
        self.log_box.moveCursor(QtGui.QTextCursor.End)
        self.log_box.appendPlainText(text)
        self.log_box.ensureCursorVisible()
