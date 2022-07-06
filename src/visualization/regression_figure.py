from typing import Any
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import jax
from data.datasets import ConditionalDataset
from utils.settings import SettingsRegressionFigure
from visualization import Scatter, Plot


class RegressionFigure:
    def __init__(self, settings: SettingsRegressionFigure = SettingsRegressionFigure()):
        self._settings = settings
        self._figure = None
    
    def __del__(self):
        plt.close(self._figure)
    
    def plot(self, dataset=None, transformation=None, parameters_list=None, std=None, scale=1.0):
        if self._figure is not None:
            self._figure.clf()
        self._figure = plt.figure(
            figsize=(
                self._settings.ax_width,
                self._settings.ax_height
            )
        )
        if len(dataset.conditional_indices) == 1:
            ax = self._figure.add_subplot(1, 1, 1)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        elif len(dataset.conditional_indices) == 2:
            ax = self._figure.add_subplot(1, 1, 1, projection="3d")
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            ax.set_zlabel("y")
        else:
            return 1
            
        if dataset is not None:
            scale = np.std(dataset.data) * 3.0
            scatter_plot = Scatter(ax, scale=scale, settings=self._settings.settings_scatter)
            scatter_plot.plot(dataset=dataset)
        if not (transformation is None or parameters_list is None or std is None):
            transformation_plot = Plot(ax, scale=scale, settings=self._settings.settings_plot)
            for j, parameters in enumerate(parameters_list):
                if j > 0:
                    transformation_plot._settings.aleatoric=False
                color = self._settings.settings_plot.cmap(1.0 * j /len(parameters_list))
                transformation_plot.plot(transformation, parameters, std, color=color, dataset=dataset)
            if len(dataset.conditional_indices) == 1:
                for j, parameters in enumerate(parameters_list):
                    color = self._settings.settings_plot.cmap(1.0 * j /len(parameters_list))
                    transformation_plot.plot_means(transformation, parameters, color=color, dataset=dataset)
        return self._figure
    
    def save(self, path):
        self._figure.savefig(path, bbox_inches="tight")
