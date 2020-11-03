from PyQt5 import QtWidgets, QtCore, QtGui
from pykilosort.gui.sorter import KiloSortWorker


class RunBox(QtWidgets.QGroupBox):

    updateContext = QtCore.pyqtSignal(object)
    sortingStepStatusUpdate = QtCore.pyqtSignal(dict)

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.setTitle("Run")

        self.parent = parent

        self.layout = QtWidgets.QGridLayout()

        self.run_all_button = QtWidgets.QPushButton("Run All")
        self.preprocess_button = QtWidgets.QPushButton("Preprocess")
        self.spike_sort_button = QtWidgets.QPushButton("Spikesort")
        self.export_button = QtWidgets.QPushButton("Export for Phy")

        self.buttons = [self.run_all_button, self.preprocess_button, self.spike_sort_button, self.export_button]

        self.data_path = None
        self.working_directory = None
        self.results_directory = None

        self.sorting_step_status = {'preprocess': False,
                                    'spikesort': False,
                                    'export': False}

        self.preprocess_done = False
        self.spikesort_done = False

        self.setup()

    def setup(self):
        self.run_all_button.clicked.connect(self.run_all)
        self.preprocess_button.clicked.connect(self.preprocess)
        self.spike_sort_button.clicked.connect(self.spikesort)
        self.export_button.clicked.connect(self.export)

        self.spike_sort_button.setEnabled(False)
        self.export_button.setEnabled(False)

        self.run_all_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.layout.addWidget(self.run_all_button, 0, 0, 3, 2)
        self.layout.addWidget(self.preprocess_button, 0, 2, 1, 2)
        self.layout.addWidget(self.spike_sort_button, 1, 2, 1, 2)
        self.layout.addWidget(self.export_button, 2, 2, 1, 2)

        self.setLayout(self.layout)

        self.disable_all_buttons()
        self.reenable_buttons()

    def disable_all_buttons(self):
        for button in self.buttons:
            button.setEnabled(False)

    def reenable_buttons(self):
        self.run_all_button.setEnabled(True)
        self.preprocess_button.setEnabled(True)
        if self.sorting_step_status['preprocess']:
            self.spike_sort_button.setEnabled(True)
        else:
            self.spike_sort_button.setEnabled(False)
        if self.sorting_step_status['spikesort']:
            self.export_button.setEnabled(True)
        else:
            self.export_button.setEnabled(False)

    def set_data_path(self, data_path):
        self.data_path = data_path

    def set_working_directory(self, working_directory_path):
        self.working_directory = working_directory_path

    def set_results_directory(self, results_directory_path):
        self.results_directory = results_directory_path

    def get_current_context(self):
        return self.parent.get_context()

    def set_sorting_step_status(self, step, status):
        self.sorting_step_status[step] = status
        self.sortingStepStatusUpdate.emit(self.sorting_step_status)

    @QtCore.pyqtSlot(object)
    def finished_preprocess(self, context):
        self.updateContext.emit(context)
        self.set_sorting_step_status('preprocess', True)

    @QtCore.pyqtSlot(object)
    def finished_spikesort(self, context):
        self.updateContext.emit(context)
        self.set_sorting_step_status('spikesort', True)

    @QtCore.pyqtSlot(object)
    def finished_export(self, context):
        self.updateContext.emit(context)
        self.set_sorting_step_status('export', True)

    @QtCore.pyqtSlot()
    def preprocess(self):
        self.set_sorting_step_status('preprocess', False)
        self.set_sorting_step_status('spikesort', False)
        self.reenable_buttons()
        if self.get_current_context() is not None:
            self.run_steps("preprocess")

    @QtCore.pyqtSlot()
    def spikesort(self):
        if self.get_current_context() is not None:
            self.run_steps("spikesort")

    @QtCore.pyqtSlot()
    def export(self):
        if self.get_current_context() is not None:
            self.run_steps("export")

    @QtCore.pyqtSlot()
    def run_all(self):
        if self.get_current_context() is not None:
            self.run_steps(["preprocess", "spikesort", "export"])

        self.set_sorting_step_status('preprocess', True)
        self.set_sorting_step_status('spikesort', True)
        self.reenable_buttons()

    def run_steps(self, steps):
        worker = KiloSortWorker(self.get_current_context(), self.data_path, self.results_directory, steps)

        worker.finishedPreprocess.connect(self.finished_preprocess)
        worker.finishedSpikesort.connect(self.finished_spikesort)
        worker.finishedAll.connect(self.finished_export)

        self.disable_all_buttons()
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        worker.start()
        while worker.isRunning():
            QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.restoreOverrideCursor()
        self.reenable_buttons()

    def prepare_for_new_context(self):
        self.set_sorting_step_status('preprocess', False)
        self.set_sorting_step_status('spikesort', False)
        self.set_sorting_step_status('export', False)
