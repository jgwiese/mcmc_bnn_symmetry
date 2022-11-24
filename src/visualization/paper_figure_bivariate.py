import copy
from typing import Any
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils import settings
from visualization import Scatter, UnivariateKDEPlot
from data import datasets


class PaperFigureBivariate:
    def __init__(self, settings: settings.SettingsPaperFigureBivariate = settings.SettingsPaperFigureBivariate()):
        self._settings = settings
        self._figure = None
    
    def __del__(self):
        plt.close(self._figure)
    
    def plot(self, data_list, scatter_index_0, scatter_index_1, univariate_index, labels, scale=None, rasterized=False, univariate=False, textsize=None, y_axis=True):
        rows = 1
        cols = 1
        if self._figure is not None:
            self._figure.clf()
        if univariate:
            self._figure = plt.figure(figsize=(self._settings.ax_width * cols * 2, self._settings.ax_height * rows * 0.5), tight_layout=True)
        else:
            self._figure = plt.figure(figsize=(self._settings.ax_width * cols, self._settings.ax_height * rows), tight_layout=True)
        if scale is None:
            scale = np.std(np.concatenate(data_list, axis=0)) * 3.0
        if self._settings.shift:
            shift = np.mean(np.concatenate(data_list, axis=0), axis=0)
        else:
            shift = np.zeros(2)
        
        ax = self._figure.add_subplot(1, cols, 1)
        if univariate:
            # univariate plot
            plot = UnivariateKDEPlot(ax, scale, shift[0], self._settings.settings_plot)
            for j, data in enumerate(data_list):
                color = self._settings.cmap(1.0 * j / len(data_list))
                plot.plot(data.T[univariate_index], color=color, rasterized=False)
            if labels[univariate_index] != "":
                ax.set_xlabel(labels[univariate_index])

        else:
            # scatter plot
            plot = Scatter(ax, scale, shift, self._settings.settings_scatter)
            for j, data in enumerate(data_list):
                dataset = datasets.ConditionalDataset(data[:, np.array([scatter_index_0, scatter_index_1])], "none", [0], [1])
                color = self._settings.cmap(1.0 * j / len(data_list))
                plot.plot(dataset, color=color, size=None, rasterized=rasterized)
            ax.set_xlabel(labels[1])
            ax.set_ylabel(labels[0])
        
        if textsize is not None:
            ax.yaxis.label.set_size(textsize)
            ax.xaxis.label.set_size(textsize)
            ax.tick_params(axis='both', which='major', labelsize=textsize - 2)
            if not y_axis:
                ax.set_yticklabels([])
                ax.set_ylabel("")
            
        else:
            ax.yaxis.label.set_size(self._settings.label_size)
            ax.xaxis.label.set_size(self._settings.label_size)
        return self._figure

    def save(self, path):
        self.figure.savefig(path, bbox_inches="tight")

