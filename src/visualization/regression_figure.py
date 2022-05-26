from dataclasses import dataclass
from typing import Any
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import jax
from tqdm import tqdm
from data.datasets import ConditionalDataset


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
    epistemic: bool = False
    cmap: matplotlib.colors.Colormap = matplotlib.cm.get_cmap("gist_rainbow") # "jet"


class TransformationPlot:
    def __init__(self, ax, scale, settings: TransformationPlotSettings = TransformationPlotSettings()):
        self._ax = ax
        self._scale = scale
        self._settings = settings
    
    def plot(self, transformation, parameters, std, color, dataset):
        if len(dataset.conditional_indices) == 1:
            self._ax.set_xlim(self._settings.xlim * self._scale)
            self._ax.set_ylim(self._settings.ylim * self._scale)
            self._ax.set_aspect(self._settings.aspect)
            
            inputs = np.linspace(self._settings.xlim[0] * self._scale, self._settings.xlim[1] * self._scale, 128)[:, np.newaxis]
            means = jax.vmap(transformation, in_axes=(None, 0))(inputs, parameters).squeeze(-1)
            for mean in tqdm(means):
                self._ax.plot(inputs[:, 0], mean, c=color, alpha=self._settings.alpha, linewidth=self._settings.linewidth)
                if self._settings.aleatoric and not self._settings.epistemic:
                    self._ax.fill_between(inputs[:, 0], mean - 1.96 * std, mean + 1.96 * std, color=color, alpha=self._settings.alpha_std, label=r"$1.96 \cdot \sigma_a$")
            if self._settings.epistemic:
                mean = np.mean(means, axis=0)
                epistemic_std = np.std(means, axis=0)
                if self._settings.aleatoric:
                    std = np.sqrt(np.power(epistemic_std, 2) + np.power(std, 2))
                    self._ax.fill_between(inputs[:, 0], mean - 1.96 * std, mean + 1.96 * std, color=color, alpha=self._settings.alpha_std, label=r"$1.96 \cdot \sqrt{\sigma_e^2 + \sigma_a^2}$")
                else:
                    std = epistemic_std
                    self._ax.fill_between(inputs[:, 0], mean - 1.96 * std, mean + 1.96 * std, color=color, alpha=self._settings.alpha_std, label=r"$1.96 \cdot \sigma_e}$")
        elif len(dataset.conditional_indices) == 2:
            x = np.linspace(self._settings.xlim[0] * self._scale, self._settings.xlim[1] * self._scale, 128)[:, np.newaxis]
            xx, yy = np.meshgrid(x, x)
            outputs = jax.vmap(transformation, in_axes=(None, 0))(np.stack([xx, yy], -1).reshape((-1, 2)), parameters).squeeze(-1).reshape((-1, 128, 128))
            for output in outputs:
                self._ax.plot_surface(xx, yy, output, alpha=self._settings.alpha, color=color)
        else:
            return 1

    def plot_means(self, transformation, parameters, color, dataset):
        inputs = np.linspace(self._settings.xlim[0] * self._scale, self._settings.xlim[1] * self._scale, 128)[:, np.newaxis]
        means = jax.vmap(transformation, in_axes=(None, 0))(inputs, parameters).squeeze(-1)
        self._ax.plot(inputs[:, 0], means.mean(0), c="gray", linestyle="solid", linewidth=self._settings.linewidth, alpha=1.0)
        self._ax.plot(inputs[:, 0], means.mean(0), c=color, linestyle="dashed", linewidth=self._settings.linewidth, alpha=1.0, label=r"$\mathbb{E}[f]$")


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
    
    def plot(self, dataset: ConditionalDataset):
        conditional = dataset.data[:, dataset.conditional_indices]
        dependent = dataset.data[:, dataset.dependent_indices]
        if conditional.shape[-1] == 1:
            self._ax.scatter(conditional, dependent, c=self._settings.color, alpha=self._settings.alpha, s=self._settings.size)
            self._ax.set_xlim(self._settings.xlim * self._scale)
            self._ax.set_ylim(self._settings.ylim * self._scale)
            self._ax.set_aspect(self._settings.aspect)
        elif conditional.shape[-1] == 2:
            self._ax.scatter(conditional.T[0], conditional.T[1], dependent.T[0], c=self._settings.color, alpha=self._settings.alpha, s=self._settings.size)


@dataclass
class RegressionSettings:
    data_plot_settings: DataPlotSettings = DataPlotSettings()
    transformation_plot_settings: TransformationPlotSettings = TransformationPlotSettings()
    ax_width: float = 12.0
    ax_height: float = 4.0


class RegressionFigure:
    def __init__(self, settings: RegressionSettings = RegressionSettings()):
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
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        else:
            return 1
            
        if dataset is not None:
            scale = np.std(dataset.data) * 3.0
            data_plot = DataPlot(ax, scale=scale, settings=self._settings.data_plot_settings)
            data_plot.plot(dataset=dataset)
        if not (transformation is None or parameters_list is None or std is None):
            transformation_plot = TransformationPlot(ax, scale=scale, settings=self._settings.transformation_plot_settings)
            for j, parameters in enumerate(parameters_list):
                if j > 0:
                    transformation_plot._settings.aleatoric=False
                color = self._settings.transformation_plot_settings.cmap(1.0 * j /len(parameters_list))
                transformation_plot.plot(transformation, parameters, std, color=color, dataset=dataset)
            for j, parameters in enumerate(parameters_list):
                color = self._settings.transformation_plot_settings.cmap(1.0 * j /len(parameters_list))
                transformation_plot.plot_means(transformation, parameters, color=color, dataset=dataset)
        return self._figure
    
    def save(self, path):
        self._figure.savefig(path, bbox_inches="tight")

