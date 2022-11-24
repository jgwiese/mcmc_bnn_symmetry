from dataclasses import dataclass
from utils import settings


@dataclass
class SettingsRegressionFigure:
    settings_scatter: settings.SettingsScatter = settings.SettingsScatter()
    settings_plot: settings.SettingsPlot = settings.SettingsPlot()
    ax_width: float = 12.0
    ax_height: float = 4.0
    label_size: float = None
