import numpy as np
import json
from pykilosort.utils import Bunch
from pykilosort.params import KilosortParams
from PyQt5 import QtWidgets, QtGui, QtCore


class ProbeBuilder(QtWidgets.QDialog):
    def __init__(self, parent, *args, **kwargs):
        super(ProbeBuilder, self).__init__(parent=parent, *args, **kwargs)
        self.parent = parent

        self.map_name_value = QtWidgets.QLineEdit()
        self.map_name_label = QtWidgets.QLabel("Name for new channel map:")

        self.x_coords_value = QtWidgets.QLineEdit()
        self.x_coords_label = QtWidgets.QLabel("X-coordinates for each site:")

        self.y_coords_value = QtWidgets.QLineEdit()
        self.y_coords_label = QtWidgets.QLabel("Y-coordinates for each site:")

        self.k_coords_value = QtWidgets.QLineEdit()
        self.k_coords_label = QtWidgets.QLabel("Shrank index (\'kcoords\') for each "
                                               "site (leave blank for single shank):")

        self.channel_map_value = QtWidgets.QLineEdit()
        self.channel_map_label = QtWidgets.QLabel("Channel map (list of rows in the data file for each site):")

        self.bad_channels_value = QtWidgets.QLineEdit()
        self.bad_channels_label = QtWidgets.QLabel("List of disconnected/bad site numbers (blank for none):")

        self.input_list = [self.map_name_value, self.x_coords_value, self.y_coords_value,
                           self.k_coords_value, self.channel_map_value, self.bad_channels_value]

        self.error_label = QtWidgets.QLabel()
        self.error_label.setText("Invalid inputs!")
        self.error_label.setWordWrap(True)

        self.okay_button = QtWidgets.QPushButton("OK", parent=self)
        self.cancel_button = QtWidgets.QPushButton("Cancel", parent=self)
        self.check_button = QtWidgets.QPushButton("Check", parent=self)

        self.map_name = None
        self.x_coords = None
        self.y_coords = None
        self.k_coords = None
        self.channel_map = None
        self.bad_channels = None

        self.probe = None

        self.values_checked = False

        self.setup()

    def setup(self):
        layout = QtWidgets.QVBoxLayout()

        info_label = QtWidgets.QLabel("Valid inputs: lists, or numpy expressions (use np for numpy)")

        self.cancel_button.clicked.connect(self.reject)
        self.okay_button.setIcon(self.parent.style().standardIcon(QtWidgets.QStyle.SP_DialogCancelButton))
        self.okay_button.clicked.connect(self.accept)
        self.okay_button.setIcon(self.parent.style().standardIcon(QtWidgets.QStyle.SP_DialogOkButton))
        self.check_button.clicked.connect(self.check_inputs)

        buttons = [self.check_button, self.okay_button, self.cancel_button]

        error_label_size_policy = self.error_label.sizePolicy()
        error_label_size_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Maximum)
        error_label_size_policy.setRetainSizeWhenHidden(True)
        self.error_label.setSizePolicy(error_label_size_policy)
        error_label_palette = self.error_label.palette()
        error_label_palette.setColor(QtGui.QPalette.Foreground, QtGui.QColor("red"))
        self.error_label.setPalette(error_label_palette)
        self.error_label.hide()

        for field in self.input_list:
            field.textChanged.connect(self.set_values_as_unchecked)

        widget_list = [self.map_name_label, self.map_name_value,
                       info_label,
                       self.x_coords_label, self.x_coords_value,
                       self.y_coords_label, self.y_coords_value,
                       self.k_coords_label, self.k_coords_value,
                       self.channel_map_label, self.channel_map_value,
                       self.bad_channels_label, self.bad_channels_value,
                       self.error_label]

        button_layout = QtWidgets.QHBoxLayout()
        for button in buttons:
            button_layout.addWidget(button)

        self.okay_button.setDisabled(True)

        for widget in widget_list:
            layout.addWidget(widget)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def set_values_as_unchecked(self):
        self.values_checked = False
        self.okay_button.setDisabled(True)

    def set_values_as_checked(self):
        self.values_checked = True
        self.okay_button.setDisabled(False)

    def check_inputs(self):
        try:
            map_name = self.map_name_value.text()
            assert len(map_name.split()) == 1

            x_coords = eval(self.x_coords_value.text())
            y_coords = eval(self.y_coords_value.text())

            x_coords = np.array(x_coords, dtype=np.float64)
            y_coords = np.array(y_coords, dtype=np.float64)

            assert len(x_coords) == len(y_coords)

            k_coords = self.k_coords_value.text()
            if k_coords == "":

                k_coords = np.ones_like(x_coords, dtype=np.float64)
            else:
                k_coords = np.array(eval(k_coords), dtype=np.float64)
                assert x_coords.size == k_coords.size

            channel_map = self.channel_map_value.text()
            if channel_map == "":
                channel_map = np.arange(x_coords.size)
            else:
                channel_map = np.array(eval(channel_map), dtype=np.int32)
                assert x_coords.size == channel_map.size
                assert channel_map.size == np.unique(channel_map).size
                assert np.amax(channel_map) < channel_map.size

            bad_channels = self.bad_channels_value.text()
            if bad_channels == "":
                bad_channels = np.array([], dtype=np.int32)
            else:
                bad_channels = np.array(eval(bad_channels), dtype=np.int32)
                assert bad_channels.size < x_coords.size

        except Exception as e:
            self.error_label.setText(str(e))
            self.error_label.show()

        else:
            self.map_name = map_name
            self.x_coords = x_coords.tolist()
            self.y_coords = y_coords.tolist()
            self.k_coords = k_coords.tolist()
            self.channel_map = channel_map.tolist()
            self.bad_channels = bad_channels.tolist()

            self.set_values_as_checked()
            self.error_label.hide()

            self.construct_probe()

    # TODO: use `Probe` from pykilosort/params.py
    def construct_probe(self):
        probe = Bunch()

        probe.xc = self.x_coords
        probe.yc = self.y_coords
        probe.kcoords = self.k_coords
        probe.chanMap = self.channel_map
        probe.bad_channels = self.bad_channels
        probe.NchanTOT = len(self.x_coords)

        self.probe = probe

    def get_map_name(self):
        return self.map_name

    def get_probe(self):
        return self.probe


class AdvancedOptionsEditor(QtWidgets.QDialog):

    def __init__(self, parent):
        super(AdvancedOptionsEditor, self).__init__(parent=parent)
        self.parent = parent

        self._default_parameters = KilosortParams().parse_obj(self.parent.get_default_advanced_options())
        self.current_parameters = self._default_parameters.dict()

        self.parameter_edit_box = JsonEditor(self)
        self.parameter_edit_box.setFont(QtGui.QFont("Monospace"))

        self.error_label = QtWidgets.QLabel()
        self.error_label.setText("Please check json syntax!")
        self.error_label.setWordWrap(True)

        self.okay_button = QtWidgets.QPushButton("Ok", parent=self)
        self.cancel_button = QtWidgets.QPushButton("Cancel", parent=self)
        self.check_button = QtWidgets.QPushButton("Check", parent=self)
        self.reset_to_original = QtWidgets.QPushButton("Reset to original", parent=self)
        self.reset_to_saved = QtWidgets.QPushButton("Reset to last saved", parent=self)

        self.values_checked = False

        self.setup()

    def setup(self):
        self.setMinimumSize(300, 600)

        layout = QtWidgets.QVBoxLayout()

        self.parameter_edit_box.textChanged.connect(self.set_values_as_unchecked)

        parameter_edit_label = QtWidgets.QLabel("Modify the advanced parameters by changing this json file:")

        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setIcon(self.parent.style().standardIcon(QtWidgets.QStyle.SP_DialogCancelButton))
        self.okay_button.clicked.connect(self.accept)
        self.okay_button.setIcon(self.parent.style().standardIcon(QtWidgets.QStyle.SP_DialogOkButton))
        self.check_button.clicked.connect(self.check_json)
        self.reset_to_original.clicked.connect(self.reset_to_original_defaults)
        self.reset_to_saved.clicked.connect(self.reset_to_saved_defaults)

        main_buttons = [self.check_button, self.okay_button, self.cancel_button]
        reset_buttons = [self.reset_to_saved, self.reset_to_original]

        error_label_size_policy = self.error_label.sizePolicy()
        error_label_size_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Maximum)
        error_label_size_policy.setRetainSizeWhenHidden(True)
        self.error_label.setSizePolicy(error_label_size_policy)
        error_label_palette = self.error_label.palette()
        error_label_palette.setColor(QtGui.QPalette.Foreground, QtGui.QColor("red"))
        self.error_label.setPalette(error_label_palette)
        self.error_label.hide()

        main_button_layout = QtWidgets.QHBoxLayout()
        for button in main_buttons:
            main_button_layout.addWidget(button)

        reset_button_layout = QtWidgets.QHBoxLayout()
        for button in reset_buttons:
            reset_button_layout.addWidget(button)

        self.okay_button.setDisabled(True)

        layout.addWidget(parameter_edit_label, 1)
        layout.addWidget(self.error_label, 1)
        layout.addWidget(self.parameter_edit_box, 6)
        layout.addLayout(main_button_layout, 1)
        layout.addLayout(reset_button_layout, 1)

        self.setLayout(layout)

        self.set_json_text()

    @QtCore.pyqtSlot()
    def set_values_as_unchecked(self):
        self.values_checked = False
        self.okay_button.setDisabled(True)

    @QtCore.pyqtSlot()
    def set_values_as_checked(self):
        self.values_checked = True
        self.okay_button.setDisabled(False)

    @QtCore.pyqtSlot()
    def check_json(self):
        try:
            param_dict = json.loads(self.parameter_edit_box.toPlainText())
            self.current_parameters = self._default_parameters.parse_obj(param_dict).dict()

            self.set_values_as_checked()

            self.error_label.setText("")
            self.error_label.hide()

            self.okay_button.setDisabled(False)
        except Exception as e:
            self.error_label.setText("Invalid syntax! Refer to terminal for error message.")
            self.error_label.show()
            print(e)

    def reset_to_original_defaults(self):
        self.current_parameters = KilosortParams().dict()
        json_dump = json.dumps(self.current_parameters, indent=4)
        self.set_text(json_dump)

    def reset_to_saved_defaults(self):
        self.current_parameters = self._default_parameters.dict()
        json_dump = json.dumps(self.current_parameters, indent=4)
        self.set_text(json_dump)

    def set_json_text(self):
        json_dump = json.dumps(self.current_parameters, indent=4)
        self.set_text(json_dump)

    def set_text(self, text):
        self.parameter_edit_box.setPlainText(text)

    def get_parameters(self):
        return self.current_parameters


class QLineNumberArea(QtWidgets.QWidget):
    """
    Adapted from: https://stackoverflow.com/a/49790764/7126611
    """
    def __init__(self, editor):
        super().__init__(editor)
        self.codeEditor = editor

    def sizeHint(self):
        return QtCore.QSize(self.editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.codeEditor.line_number_area_paint_event(event)


class JsonEditor(QtWidgets.QPlainTextEdit):
    """
    Adapted from: https://stackoverflow.com/a/49790764/7126611
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.lineNumberArea = QLineNumberArea(self)
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.cursorPositionChanged.connect(self.highlight_current_line)
        self.update_line_number_area_width(0)

    def line_number_area_width(self):
        digits = 1
        max_value = max(1, self.blockCount())
        while max_value >= 10:
            max_value /= 10
            digits += 1
        space = 3 + self.fontMetrics().width('9') * digits + 5
        return space

    def update_line_number_area_width(self, null):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect, dy):
        if dy:
            self.lineNumberArea.scroll(0, dy)
        else:
            self.lineNumberArea.update(0, rect.y(), self.lineNumberArea.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.lineNumberArea.setGeometry(QtCore.QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height()))

    def highlight_current_line(self):
        extra_selections = []
        if not self.isReadOnly():
            selection = QtWidgets.QTextEdit.ExtraSelection()
            line_color = QtGui.QColor(53, 50, 47)
            selection.format.setBackground(line_color)
            selection.format.setProperty(QtGui.QTextFormat.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            extra_selections.append(selection)
        self.setExtraSelections(extra_selections)

    def line_number_area_paint_event(self, event):
        painter = QtGui.QPainter(self.lineNumberArea)

        painter.fillRect(event.rect(), QtGui.QColor(53, 50, 47))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()

        # Just to make sure I use the right font
        height = self.fontMetrics().height()
        while block.isValid() and (top <= event.rect().bottom()):
            if block.isVisible() and (bottom >= event.rect().top()):
                number = str(block_number + 1)
                painter.setPen(QtCore.Qt.white)
                painter.drawText(0, top, self.lineNumberArea.width(), height, QtCore.Qt.AlignRight, number)

            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            block_number += 1


controls_popup_text = """
<font style="font-family:Monospace">
Controls <br>
-------- <br>
<br>
[1 2 3 4]        - activate/deactivate raw/filtered/prediction/residual views of the dataset <br>
[c]              - toggle between colormap mode and traces mode <br>
[up/down]        - move through channels in traces mode <br>
[scroll]         - move forward/backward in time <br>
[ctrl + scroll]  - add/remove channels in colormap mode; slide up/down probe in traces mode <br>
[alt + scroll]   - change data/colormap scaling <br>
[shift + scroll] - zoom in/out in time <br>
[left click]     - move forward/backward in time <br>
[right click]    - enable/disable channel for analysis <br>
</font>
"""
