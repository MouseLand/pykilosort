import numpy as np
import typing as t
from pyqtgraph import LayoutWidget, ColorMap, RemoteGraphicsView, PlotItem

from pykilosort.gui import SANITY_PLOT_COLORS


class SanityPlotWidget(LayoutWidget):
    def __init__(self, parent, num_remote_plots, title):
        super(SanityPlotWidget, self).__init__(parent=parent)
        self.num_remote_plots = num_remote_plots
        self.remote_plots = []

        self.colormap = ColorMap(pos=np.linspace(0, 1, len(SANITY_PLOT_COLORS)),
                                 color=np.array(SANITY_PLOT_COLORS) * 255)
        self.lookup_table = self.colormap.getLookupTable(start=-1, stop=1, nPts=1024)

        self.setWindowTitle(title)

        self.create_remote_views()
        self.arrange_views()

        self.hide()

    def create_remote_views(self):
        for _ in range(self.num_remote_plots):
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
                    x_lim: t.Optional[tuple] = None,
                    y_lim: t.Optional[tuple] = None,
                    semi_log_x: t.Optional[bool] = None,
                    semi_log_y: t.Optional[bool] = None,
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
                  x_lim: t.Optional[tuple] = None,
                  y_lim: t.Optional[tuple] = None,
                  semi_log_x: t.Optional[bool] = None,
                  semi_log_y: t.Optional[bool] = None,
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
