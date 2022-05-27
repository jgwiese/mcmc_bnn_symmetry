import copy
from typing import Any
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class TriangleFigure:
    def __init__(self, settings: TriangleSettings = TriangleSettings()):
        self._settings = settings
        self.figure = None
    
    def __del__(self):
        plt.close(self.figure)
    
    def plot(self, data_list, scale=None, sizes=None, adjacency_matrix=None):
        if sizes is not None:
            assert len(sizes) == len(data_list), "number of sizes needs to match the length of data_list"
        rows = cols = data_list[0].shape[-1]
        if self.figure is not None:
            self.figure.clf()
        self.figure = plt.figure(figsize=(self._settings.ax_width * rows, self._settings.ax_height * cols))
        if scale is None:
            scale = np.std(np.concatenate(data_list, axis=0)) * 3
        if self._settings.shift:
            shift = np.mean(np.concatenate(data_list, axis=0), axis=0)
        else:
            shift = np.zeros(2)
        
        for row in tqdm(range(rows)):
            for col in range(cols):
                if col > row:
                    continue
                elif col == row and not self._settings.univariate:
                    continue

                i = row * cols + col
                ax = self.figure.add_subplot(cols, rows, i + 1)

                plot = None
                if col == row:
                    plot = UnivariatePlot(ax, scale, shift[0], self._settings.plot_settings)
                    for j, data in enumerate(data_list):
                        color = self._settings.cmap(1.0 * j / len(data_list))
                        plot.plot(data.T[row], color=color)
                if col != row:
                    plot = BivariatePlot(ax, scale, shift, self._settings.scatter_settings)
                    for j, data in enumerate(data_list):
                        color = self._settings.cmap(1.0 * j / len(data_list))
                        if adjacency_matrix is not None and j > 0:
                            color = self._settings.cmap(1.0 * (j - 1) / (len(data_list) - 1))
                        current_size = None
                        if sizes is not None:
                            current_size = sizes[j]
                        if adjacency_matrix is not None and j == 0:
                            if len(data_list) > 1:
                                current_size = 0.0
                            plot.plot(data.T[np.array([col, row])], color=color, size=current_size, adjacency_matrix=adjacency_matrix)
                        else:
                            plot.plot(data.T[np.array([col, row])], color=color, size=current_size, adjacency_matrix=None)
                
                # labels
                if row == rows - 1:
                    ax.set_xlabel(f"${self._settings.prefix}_{{{col}}}$")
                else:
                    ax.set_xticklabels([])
                if col == 0:
                    ax.set_ylabel(f"${self._settings.prefix}_{{{row}}}$")
                else:
                    ax.set_yticklabels([])
        
        return self.figure

    def save(self, path):
        self.figure.savefig(path, bbox_inches="tight")

