"""
This module contains all files required by the pykilosort GUI.
"""
from .data_view_box import DataViewBox
from .probe_view_box import ProbeViewBox
from .settings_box import SettingsBox
from .message_log_box import MessageLogBox
from .run_box import RunBox
from .header_box import HeaderBox
from .palettes import DarkPalette, COLORMAP_COLORS
from .sorter import find_good_channels, filter_and_whiten, KiloSortWorker
from .minor_gui_elements import ProbeBuilder
from . import probes

from .main import KiloSortGUI
from .launch import launcher
