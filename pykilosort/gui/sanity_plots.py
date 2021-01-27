import numpy as np
import typing as t
from pyqtgraph import LayoutWidget, ColorMap, RemoteGraphicsView, PlotItem

from pykilosort.gui import SANITY_PLOT_COLORS


class SanityPlotWidget(LayoutWidget):
    def __init__(self, parent, num_remote_plots, title):
        super(SanityPlotWidget, self).__init__(parent=parent)
        self.num_remote_plots = num_remote_plots
        self.remote_plots = []

        self.seq_colormap = ColorMap(pos=np.linspace(0, 1, len(SANITY_PLOT_COLORS["sequential"])),
                                     color=np.array(SANITY_PLOT_COLORS["sequential"]))
        self.div_colormap = ColorMap(pos=np.linspace(0, 1, len(SANITY_PLOT_COLORS["diverging"])),
                                     color=np.array(SANITY_PLOT_COLORS["diverging"]))

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
                    **kwargs: t.Optional[dict],
                    ) -> PlotItem:
        remote_plot = self.get_remote_plots()[plot_pos]
        remote_plot_item = remote_plot._view.centralWidget  # noqa
        remote_view = remote_plot_item.getViewBox()

        remote_plot_item.clear()

        remote_plot_item.setLogMode(x=semi_log_x, y=semi_log_y)

        scatter_plot = remote_plot.pg.ScatterPlotItem(x=x_data, y=y_data, **kwargs)
        remote_plot_item.addItem(scatter_plot)

        self._set_labels_on_plot(remote_plot_item, labels)
        remote_plot_item.hideAxis('top')
        remote_plot_item.hideAxis('right')

        if not (semi_log_x or semi_log_y):
            remote_plot_item.setRange(xRange=x_lim, yRange=y_lim)
        else:
            if (x_lim is not None) and (y_lim is not None):
                remote_view.setLimits(
                    xMin=x_lim[0],
                    xMax=x_lim[1],
                    yMin=y_lim[0],
                    yMax=y_lim[1],
                )
            elif (x_lim is not None) and (y_lim is None):
                remote_view.setLimits(
                    xMin=x_lim[0],
                    xMax=x_lim[1],
                )
            elif (y_lim is not None) and (x_lim is None):
                remote_view.setLimits(
                    yMin=y_lim[0],
                    yMax=y_lim[1],
                ),
            else:
                # if both x_lim and y_lim are None, enable autoRange
                remote_plot_item.enableAutoRange()

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
                  cmap_style: str = "diverging",
                  limits: t.Optional[tuple] = None,
                  **kwargs: t.Optional[dict],
                  ) -> PlotItem:
        remote_plot = self.get_remote_plots()[plot_pos]
        remote_plot_item = remote_plot._view.centralWidget  # noqa

        remote_plot_item.clear()

        if limits is None:
            limits = (-1, 1)

        if "levels" in kwargs.keys():
            levels = kwargs.pop("levels")
            auto_levels = False
        else:
            levels = None
            auto_levels = True

        if normalize:
            array = self.normalize_array(array)

        if cmap_style == "sequential":
            colormap = self.seq_colormap
        elif cmap_style == "diverging":
            colormap = self.div_colormap
        else:
            raise ValueError("Invalid colormap style requested.")

        lut = colormap.getLookupTable(
            start=limits[0],
            stop=limits[1],
            nPts=1024,
        )

        if auto_levels:
            image_item = remote_plot.pg.ImageItem(image=array,
                                                  lut=lut)
        else:
            image_item = remote_plot.pg.ImageItem(image=array,
                                                  lut=lut,
                                                  autoLevels=auto_levels,
                                                  levels=levels)

        if not auto_levels:
            image_item.setLevels(levels)

        remote_plot_item.addItem(image_item, **kwargs)

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
