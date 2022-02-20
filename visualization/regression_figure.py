from dataclasses import dataclass
from typing import Any
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RegressionPlotSettings:
    xlim: Any = np.array([-1.0, 1.0])
    ylim: Any = np.array([-1.0, 1.0])
    aspect: str = "equal"
    linewidth: float = 1.0
    plot_color: str = "black"
    plot_alpha: float = 0.01
    scatter_color: str = "blue"
    scatter_alpha: float = 1.0
    scatter_size: float = 4.0
    aleatoric: bool = False


@dataclass
class RegressionSettings:
    plot_settings: RegressionPlotSettings = RegressionPlotSettings()


class RegressionPlot:
    def __init__(self, ax, scale, settings: RegressionPlotSettings = RegressionPlotSettings()):
        self._ax = ax
        self._scale = scale
        self._settings = settings
    
    def plot(self, data, transformation, parameters, std):
        inputs = np.linspace(self._settings.xlim[0] * self._scale, self._settings.xlim[1] * self._scale, 128)
        self._ax.scatter(data.T[0], data.T[1], c=self._settings.scatter_color, alpha=self._settings.scatter_alpha, s=self._settings.scatter_size)
        self._ax.set_xlim(self._settings.xlim * self._scale)
        self._ax.set_ylim(self._settings.ylim * self._scale)
        self._ax.set_aspect(self._settings.aspect)
        for p in parameters:
            mean = transformation(inputs=np.expand_dims(inputs, axis=-1), parameters=p).squeeze()
            self._ax.plot(inputs, mean, c=self._settings.plot_color, alpha=self._settings.plot_alpha)
            if self._settings.aleatoric:
                self._ax.fill_between(inputs, mean - 1.96 * std, mean + 1.96 * std, color="red", alpha=self._settings.plot_alpha)

class RegressionFigure:
    def __init__(self, ax_width: float = 12.0, ax_height: float = 4.0, settings: RegressionSettings = RegressionSettings()):
        self._settings = settings
        self._ax_width = ax_width
        self._ax_height = ax_height
        self._figure = None
    
    def __del__(self):
        plt.close(self._figure)
    
    def plot(self, data, transformation, parameters, std):
        if self._figure is not None:
            self._figure.clf()
        self._figure = plt.figure(figsize=(self._ax_width, self._ax_height))
        scale = np.std(data) * 3
        ax = self._figure.add_subplot(1, 1, 1)
        plot = RegressionPlot(ax, scale=scale, settings=self._settings.plot_settings)
        plot.plot(data, transformation, parameters, std)
        return self._figure

