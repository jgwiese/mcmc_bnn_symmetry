from dataclasses import dataclass
from visualization import settings
import matplotlib
import numpy as np


@dataclass
class SettingsPaperFigureBivariate:
    settings_plot: settings.SettingsPlot = settings.SettingsPlot()
    settings_scatter: settings.SettingsScatter = settings.SettingsScatter()
    ax_width: float = np.sqrt(2.0)
    ax_height: float = np.sqrt(2.0)
    prefix: str = r"\theta"
    univariate: bool = True
    shift: bool = False
    cmap: matplotlib.colors.Colormap = matplotlib.cm.get_cmap("gist_rainbow") # "jet"
    label_size: int = 10
