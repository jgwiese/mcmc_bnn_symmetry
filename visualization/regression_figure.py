from dataclasses import dataclass
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import jax
from tqdm import tqdm


@dataclass
class TransformationPlotSettings:
    xlim: Any = np.array([-1.0, 1.0])
    ylim: Any = np.array([-1.0, 1.0])
    aspect: str = "equal"
    linewidth: float = 1.0
    color: str = "black"
    alpha: float = 1.0
    alpha_std: float = 0.2
    aleatoric: bool = False


class TransformationPlot:
    def __init__(self, ax, scale, settings: TransformationPlotSettings = TransformationPlotSettings()):
        self._ax = ax
        self._scale = scale
        self._settings = settings
    
    def plot(self, transformation, parameters, std):
        self._ax.set_xlim(self._settings.xlim * self._scale)
        self._ax.set_ylim(self._settings.ylim * self._scale)
        self._ax.set_aspect(self._settings.aspect)
        
        inputs = np.linspace(self._settings.xlim[0] * self._scale, self._settings.xlim[1] * self._scale, 128)
        means = jax.vmap(transformation, in_axes=(None, 0))(np.expand_dims(inputs, axis=-1), parameters).squeeze(-1)
        for mean in tqdm(means):
            self._ax.plot(inputs, mean, c=self._settings.color, alpha=self._settings.alpha)
            if self._settings.aleatoric:
                self._ax.fill_between(inputs, mean - 1.96 * std, mean + 1.96 * std, color="red", alpha=self._settings.alpha_std)


@dataclass
class DataPlotSettings:
    xlim: Any = np.array([-1.0, 1.0])
    ylim: Any = np.array([-1.0, 1.0])
    aspect: str = "equal"
    color: str = "black"
    alpha: float = 1.0
    size: float = 4.0


class DataPlot:
    def __init__(self, ax, scale, settings: DataPlotSettings = DataPlotSettings()):
        self._ax = ax
        self._scale = scale
        self._settings = settings
    
    def plot(self, data):
        self._ax.scatter(data.T[0], data.T[1], c=self._settings.color, alpha=self._settings.alpha, s=self._settings.size)
        self._ax.set_xlim(self._settings.xlim * self._scale)
        self._ax.set_ylim(self._settings.ylim * self._scale)
        self._ax.set_aspect(self._settings.aspect)


@dataclass
class RegressionSettings:
    data_plot_settings: DataPlotSettings = DataPlotSettings()
    transformation_plot_settings: TransformationPlotSettings = TransformationPlotSettings()


class RegressionFigure:
    def __init__(self, ax_width: float = 12.0, ax_height: float = 4.0, settings: RegressionSettings = RegressionSettings()):
        self._settings = settings
        self._ax_width = ax_width
        self._ax_height = ax_height
        self._figure = None
    
    def __del__(self):
        plt.close(self._figure)
    
    def plot(self, data=None, transformation=None, parameters=None, std=None, scale=1.0):
        if self._figure is not None:
            self._figure.clf()
        self._figure = plt.figure(figsize=(self._ax_width, self._ax_height))
        ax = self._figure.add_subplot(1, 1, 1)
        
        if data is not None:
            scale = np.std(data) * 3.0
            data_plot = DataPlot(ax, scale=scale, settings=self._settings.data_plot_settings)
            data_plot.plot(data=data)
        if not (transformation is None or parameters is None or std is None):
            transformation_plot = TransformationPlot(ax, scale=scale, settings=self._settings.transformation_plot_settings)
            transformation_plot.plot(transformation, parameters, std)
        return self._figure
    
    def save(self, path):
        self._figure.savefig(path, bbox_inches="tight")

