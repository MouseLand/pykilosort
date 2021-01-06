import typing as t
import numpy as np
from pykilosort.gui.sorter import KiloSortWorker
from pykilosort.gui.palettes import SANITY_PLOT_COLORS
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import RemoteGraphicsView, LayoutWidget, PlotItem, ColorMap


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
        self.sanity_plot_option = QtWidgets.QCheckBox("Sanity Plots")

        self.buttons = [
            self.run_all_button,
            self.preprocess_button,
            self.spike_sort_button,
            self.export_button,
            self.sanity_plot_option,
        ]

        self.data_path = None
        self.working_directory = None
        self.results_directory = None

        self.sorting_step_status = {
            "preprocess": False,
            "spikesort": False,
            "export": False,
        }

        self.preprocess_done = False
        self.spikesort_done = False

        self.remote_widgets = None

        self.setup()

    def setup(self):
        self.run_all_button.clicked.connect(self.run_all)
        self.preprocess_button.clicked.connect(self.preprocess)
        self.spike_sort_button.clicked.connect(self.spikesort)
        self.export_button.clicked.connect(self.export)

        self.spike_sort_button.setEnabled(False)
        self.export_button.setEnabled(False)

        self.run_all_button.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        self.layout.addWidget(self.run_all_button, 0, 0, 2, 2)
        self.layout.addWidget(self.preprocess_button, 0, 2, 1, 2)
        self.layout.addWidget(self.spike_sort_button, 1, 2, 1, 2)
        self.layout.addWidget(self.export_button, 2, 2, 1, 2)
        self.layout.addWidget(self.sanity_plot_option, 2, 1, 1, 1)

        self.setLayout(self.layout)

        self.disable_all_buttons()
        self.reenable_buttons()

    def disable_all_buttons(self):
        for button in self.buttons:
            button.setEnabled(False)

    def reenable_buttons(self):
        self.run_all_button.setEnabled(True)
        self.preprocess_button.setEnabled(True)
        if self.sorting_step_status["preprocess"]:
            self.spike_sort_button.setEnabled(True)
        else:
            self.spike_sort_button.setEnabled(False)
        if self.sorting_step_status["spikesort"]:
            self.export_button.setEnabled(True)
        else:
            self.export_button.setEnabled(False)
        self.sanity_plot_option.setEnabled(True)

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
        self.set_sorting_step_status("preprocess", True)

    @QtCore.pyqtSlot(object)
    def finished_spikesort(self, context):
        self.updateContext.emit(context)
        self.set_sorting_step_status("spikesort", True)

    @QtCore.pyqtSlot(object)
    def finished_export(self, context):
        self.updateContext.emit(context)
        self.set_sorting_step_status("export", True)

    @QtCore.pyqtSlot()
    def preprocess(self):
        self.set_sorting_step_status("preprocess", False)
        self.set_sorting_step_status("spikesort", False)
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

        self.set_sorting_step_status("preprocess", True)
        self.set_sorting_step_status("spikesort", True)
        self.reenable_buttons()

    def run_steps(self, steps):
        self.disable_all_buttons()
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))

        if self.sanity_plot_option.isChecked() and "spikesort" in steps:
            dissimilarity_plot = SanityPlotWidget(parent=None,
                                                  num_remote_plots=2,
                                                  title="Dissimilarity Matrices")
            dissimilarity_plot.resize(600, 400)

            diagnostic_plot = SanityPlotWidget(parent=None,
                                               num_remote_plots=4,
                                               title="Diagnostic Plots")
            diagnostic_plot.resize(750, 600)
            sanity_plots = True
            self.remote_widgets = [dissimilarity_plot, diagnostic_plot]

        else:
            sanity_plots = False
            self.remote_widgets = None

        worker = KiloSortWorker(
            context=self.get_current_context(),
            data_path=self.data_path,
            output_directory=self.results_directory,
            steps=steps,
            sanity_plots=sanity_plots,
            plot_widgets=self.remote_widgets,
        )

        worker.finishedPreprocess.connect(self.finished_preprocess)
        worker.finishedSpikesort.connect(self.finished_spikesort)
        worker.finishedAll.connect(self.finished_export)

        worker.start()
        while worker.isRunning():
            QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.restoreOverrideCursor()
        self.reenable_buttons()

    def prepare_for_new_context(self):
        self.set_sorting_step_status("preprocess", False)
        self.set_sorting_step_status("spikesort", False)
        self.set_sorting_step_status("export", False)


class SanityPlotWidget(LayoutWidget):
    def __init__(self, parent, num_remote_plots, title, links=None):
        super(SanityPlotWidget, self).__init__(parent=parent)
        self.num_remote_plots = num_remote_plots
        self.remote_plots = []

        self.colormap = ColorMap(pos=np.linspace(0, 1, len(SANITY_PLOT_COLORS)),
                                 color=np.array(SANITY_PLOT_COLORS) * 255)
        self.lookup_table = self.colormap.getLookupTable(start=-1, stop=1, nPts=1024)

        self.addLabel(title, colspan=2)
        self.nextRow()

        self.create_remote_views()
        self.arrange_views()

        self.hide()

    def create_remote_views(self):
        for i in range(self.num_remote_plots):
            remote_plot = RemoteGraphicsView(useOpenGL=True)
            remote_plot_item = remote_plot.pg.PlotItem()
            remote_plot_item._setProxyOptions(deferGetattr=True)  # noqa
            remote_plot.setCentralItem(remote_plot_item)

            self.remote_plots.append(
                remote_plot
            )

    def arrange_views(self):
        for i, remote_plot in enumerate(self.remote_plots):
            self.addWidget(remote_plot)

            if (i + 1) % 2 == 0:
                self.nextRow()

    @staticmethod
    def _set_labels_on_plot(plot_item, labels):
        plot_item.setLabels(left=labels.get("left", ""),
                            right=labels.get("right", ""),
                            top=labels.get("top", ""),
                            bottom=labels.get("bottom", ""),
                            title=labels.get("title", ""))

    def add_scatter(self,
                    x_data: np.ndarray,
                    y_data: np.ndarray,
                    plot_pos: int,
                    labels: dict,
                    x_lim: t.Union[tuple, None] = None,
                    y_lim: t.Union[tuple, None] = None,
                    semi_log_x: t.Union[bool, None] = None,
                    semi_log_y: t.Union[bool, None] = None,
                    ) -> PlotItem:
        remote_plot = self.get_remote_plots()[plot_pos]
        remote_plot_item = remote_plot._view.centralWidget  # noqa

        remote_plot_item.clear()

        remote_plot_item.setLogMode(x=semi_log_x, y=semi_log_y)

        scatter_plot = remote_plot.pg.ScatterPlotItem(x=x_data, y=y_data, pxMode=True, symbol="o")
        remote_plot_item.addItem(scatter_plot)

        remote_plot_item.setRange(xRange=x_lim, yRange=y_lim)

        self._set_labels_on_plot(remote_plot_item, labels)
        remote_plot_item.hideAxis('top')
        remote_plot_item.hideAxis('right')

        return remote_plot_item

    def add_curve(self,
                  x_data: np.ndarray,
                  y_data: np.ndarray,
                  plot_pos: int,
                  labels: dict,
                  x_lim: t.Union[tuple, None] = None,
                  y_lim: t.Union[tuple, None] = None,
                  semi_log_x: t.Union[bool, None] = None,
                  semi_log_y: t.Union[bool, None] = None,
                  ) -> PlotItem:
        remote_plot = self.get_remote_plots()[plot_pos]
        remote_plot_item = remote_plot._view.centralWidget  # noqa

        remote_plot_item.setLogMode(x=semi_log_x, y=semi_log_y)

        remote_plot_item.plot(x=x_data, y=y_data, clear=True)

        remote_plot_item.setRange(xRange=x_lim, yRange=y_lim)

        self._set_labels_on_plot(remote_plot_item, labels)
        remote_plot_item.hideAxis('top')
        remote_plot_item.hideAxis('right')

        return remote_plot_item

    def add_image(self,
                  array: np.ndarray,
                  plot_pos: int,
                  labels: dict,
                  normalize: bool = True,
                  invert_y: bool = True,
                  ) -> PlotItem:
        remote_plot = self.get_remote_plots()[plot_pos]
        remote_plot_item = remote_plot._view.centralWidget  # noqa

        remote_plot_item.clear()

        if normalize:
            array = self.normalize_array(array)

        image_item = remote_plot.pg.ImageItem(image=array,
                                              lut=self.lookup_table)
        remote_plot_item.addItem(image_item)

        self._set_labels_on_plot(remote_plot_item, labels)
        remote_plot_item.hideAxis('top')
        remote_plot_item.hideAxis('right')

        remote_plot_item.invertY(invert_y)

        return remote_plot_item

    @staticmethod
    def normalize_array(array):
        return 2. * (array - np.amin(array)) / np.ptp(array) - 1

    def get_remote_plots(self):
        return self.remote_plots

    def close_all_plots(self):
        for plot in self.remote_plots:
            plot.close()
