from dataclasses import dataclass
import numpy as np
from typing import Any
import matplotlib


@dataclass
class SettingsPlot:
    xlim: Any = np.array([-1.0, 1.0])
    ylim: Any = np.array([-1.0, 1.0])
    aspect: str = "equal"
    linewidth: float = 1.0
    color: str = "black"
    alpha: float = 1.0
    alpha_std: float = 0.2
    aleatoric: bool = False
    epistemic: bool = False
    cmap: matplotlib.colors.Colormap = matplotlib.cm.get_cmap("gist_rainbow") # "jet"


"""
@dataclass
class PlotSettings:
    xlim: Any = np.array([-1.0, 1.0])
    ylim: Any = np.array([-1.0, 1.0])
    aspect: str = "equal"
    color: str = "black"
    alpha: float = 1.0
    linewidth: float = 1.0
"""