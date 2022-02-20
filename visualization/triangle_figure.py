from dataclasses import dataclass
import copy
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class PlotSettings:
    xlim: Any = np.array([-1.0, 1.0])
    ylim: Any = np.array([-1.0, 1.0])
    aspect: str = "equal"
    color: str = "black"
    alpha: float = 1.0
    linewidth: float = 1.0


@dataclass
class ScatterSettings:
    xlim: Any = np.array([-1.0, 1.0])
    ylim: Any = np.array([-1.0, 1.0])
    aspect: str = "equal"
    color: str = "black"
    alpha: float = 1.0
    size: float = 1.0


@dataclass
class TriangleSettings:
    plot_settings: PlotSettings = PlotSettings()
    scatter_settings: ScatterSettings = ScatterSettings()
    univariate: bool = True


class UnivariatePlot:
    def __init__(self, ax, scale, settings: PlotSettings):
        self._ax = ax
        self._scale = scale
        self._settings = settings
    
    def plot(self, data):
        sns.kdeplot(x=data, ax=self._ax, fill=False, color=self._settings.color, alpha=self._settings.alpha, linewidth=1.0)
        self._ax.set_xlim(self._settings.xlim * self._scale)
        self._ax.set_ylabel("")
        self._ax.tick_params(direction="in")


class BivariatePlot:
    def __init__(self, ax, scale, settings: ScatterSettings):
        self._ax = ax
        self._scale = scale
        self._settings = settings
    
    def plot(self, data):
        self._ax.grid(visible=True)
        self._ax.scatter(data[0], data[1], s=self._settings.size, alpha=self._settings.alpha, color=self._settings.color)
        self._ax.set_xlim(self._settings.xlim * self._scale)
        self._ax.set_ylim(self._settings.ylim * self._scale)
        self._ax.set_aspect(self._settings.aspect)
        self._ax.tick_params(direction="in")


class TriangleFigure:
    def __init__(self, ax_width: float = np.sqrt(2.0), ax_height: float = np.sqrt(2.0), prefix=r"\theta", settings: TriangleSettings = TriangleSettings()):
        self._settings = settings
        self._ax_width = ax_width
        self._ax_height = ax_height
        self.figure = None
        self.prefix = prefix
    
    def __del__(self):
        plt.close(self.figure)
    
    def plot(self, data, data_2=None):
        rows = cols = data.shape[-1]
        if self.figure is not None:
            self.figure.clf()
        self.figure = plt.figure(figsize=(self._ax_width * rows, self._ax_height * cols))
        scale = np.std(data) * 3
        
        for row in range(rows):
            for col in range(cols):
                if col > row:
                    continue
                elif col == row and not self._settings.univariate:
                    continue

                i = row * cols + col
                ax = self.figure.add_subplot(cols, rows, i + 1)

                plot = None
                if col == row:
                    plot = UnivariatePlot(ax, scale, self._settings.plot_settings)
                    plot.plot(data.T[row])
                    if data_2 is not None:
                        settings_2 = copy.deepcopy(self._settings.plot_settings)
                        settings_2.color = "red"
                        plot = UnivariatePlot(ax, scale, settings_2)
                        plot.plot(data_2.T[row])
                if col != row:
                    plot = BivariatePlot(ax, scale, self._settings.scatter_settings)
                    plot.plot(data.T[np.array([col, row])])
                    if data_2 is not None:
                        settings_2 = copy.deepcopy(self._settings.scatter_settings)
                        settings_2.color = "red"
                        plot = BivariatePlot(ax, scale, settings_2)
                        plot.plot(data_2.T[np.array([col, row])])
                
                # labels
                if row == rows - 1:
                    ax.set_xlabel(f"${self.prefix}_{{{col}}}$")
                else:
                    ax.set_xticklabels([])
                if col == 0:
                    ax.set_ylabel(f"${self.prefix}_{{{row}}}$")
                else:
                    ax.set_yticklabels([])
        
        return self.figure

