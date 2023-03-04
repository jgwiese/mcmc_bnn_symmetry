from dataclasses import dataclass
import numpy as np
from visualization import settings


@dataclass
class SettingsSequenceFigure:
    settings_scatter: settings.SettingsScatter = settings.SettingsScatter()
    settings_plot: settings.SettingsPlot = settings.SettingsPlot()
    ax_width: float = 4.0
    ax_height: float = 4.0
    cols: int = 4
