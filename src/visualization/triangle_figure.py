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


class TriangleFigure:
    def __init__(self, settings: settings.SettingsTriangleFigure = settings.SettingsTriangleFigure()):
        self._settings = settings
        self._figure = None
    
    def __del__(self):
        plt.close(self._figure)
    
    def plot(self, data_list, scale=None, sizes=None, adjacency_matrix=None, triangle="lower", textsize=None, rasterized=False):
        if sizes is not None:
            assert len(sizes) == len(data_list), "number of sizes needs to match the length of data_list"
        rows = cols = data_list[0].shape[-1]
        if self._figure is not None:
            self._figure.clf()
        self._figure = plt.figure(figsize=(self._settings.ax_width * rows, self._settings.ax_height * cols), tight_layout=True)
        if scale is None:
            scale = np.std(np.concatenate(data_list, axis=0)) * 3
        if self._settings.shift:
            shift = np.mean(np.concatenate(data_list, axis=0), axis=0)
        else:
            shift = np.zeros(2)
        
        for row in tqdm(range(rows)):
            for col in range(cols):
                if triangle == "lower":
                    if col > row:
                        continue
                    elif col == row and not self._settings.univariate:
                        continue
                elif triangle == "upper":
                    if col < row:
                        continue
                    elif col == row and not self._settings.univariate:
                        continue
                else:
                    pass

                i = row * cols + col
                ax = self._figure.add_subplot(cols, rows, i + 1)

                plot = None
                if col == row:
                    plot = UnivariateKDEPlot(ax, scale, shift[0], self._settings.settings_plot)
                    for j, data in enumerate(data_list):
                        color = self._settings.cmap(1.0 * j / len(data_list))
                        plot.plot(data.T[row], color=color)
                if col != row:
                    plot = Scatter(ax, scale, shift, self._settings.settings_scatter)
                    for j, data in enumerate(data_list):
                        dataset = datasets.ConditionalDataset(data[:, np.array([col, row])], "none", [0], [1])
                        color = self._settings.cmap(1.0 * j / len(data_list))
                        if adjacency_matrix is not None and j > 0:
                            color = self._settings.cmap(1.0 * (j - 1) / (len(data_list) - 1))
                        current_size = None
                        if sizes is not None:
                            current_size = sizes[j]
                        if adjacency_matrix is not None and j == 0:
                            if len(data_list) > 1:
                                current_size = 0.0
                            plot.plot(dataset, color=color, size=current_size, adjacency_matrix=adjacency_matrix, rasterized=rasterized)
                        else:
                            plot.plot(dataset, color=color, size=current_size, adjacency_matrix=None, rasterized=rasterized)
                
                # labels
                if triangle == "lower":
                    if row == rows - 1:
                        ax.set_xlabel(f"${self._settings.prefix}_{{{col}}}$")
                    else:
                        ax.set_xticklabels([])
                    if col == 0:
                        ax.set_ylabel(f"${self._settings.prefix}_{{{row}}}$")
                    else:
                        ax.set_yticklabels([])
                elif triangle == "upper":
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.tick_right()
                    ax.xaxis.set_label_position("top")
                    ax.xaxis.tick_top()
                    if row == 0:
                        ax.set_xlabel(f"${self._settings.prefix}_{{{col}}}$")
                    else:
                        ax.set_xticklabels([])
                    if col == cols - 1:
                        ax.set_ylabel(f"${self._settings.prefix}_{{{row}}}$")
                    else:
                        ax.set_yticklabels([])
                
                if textsize is not None:
                    ax.yaxis.label.set_size(textsize)
                    ax.xaxis.label.set_size(textsize)
                    ax.tick_params(axis='both', which='major', labelsize=textsize - 2)
                    #ax.set_xticklabels(, rotation=45, ha='right')
                    plt.xticks(rotation=-45)
                else:
                    ax.yaxis.label.set_size(self._settings.label_size)
                    ax.xaxis.label.set_size(self._settings.label_size)
        
        return self._figure

    def save(self, path):
        self.figure.savefig(path, bbox_inches="tight")

