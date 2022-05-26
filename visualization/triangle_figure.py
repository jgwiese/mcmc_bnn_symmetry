from dataclasses import dataclass
import copy
from typing import Any
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


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
    alpha: float = 1.0
    size: float = 1.0
    marker: str = "."


@dataclass
class TriangleSettings:
    plot_settings: PlotSettings = PlotSettings()
    scatter_settings: ScatterSettings = ScatterSettings()
    ax_width: float = np.sqrt(2.0)
    ax_height: float = np.sqrt(2.0)
    prefix: str = r"\theta"
    univariate: bool = True
    shift: bool = False
    cmap: matplotlib.colors.Colormap = matplotlib.cm.get_cmap("gist_rainbow") # "jet"


class UnivariatePlot:
    def __init__(self, ax, scale, shift, settings: PlotSettings):
        self._ax = ax
        self._scale = scale
        self._shift = shift
        self._settings = settings
    
    def plot(self, data, color):
        sns.kdeplot(x=data, ax=self._ax, fill=False, color=color, alpha=self._settings.alpha, linewidth=1.0, warn_singular=False)
        self._ax.set_xlim(self._settings.xlim * self._scale + self._shift)
        self._ax.set_ylabel("")
        self._ax.tick_params(direction="in")


class BivariatePlot:
    def __init__(self, ax, scale, shift, settings: ScatterSettings):
        self._ax = ax
        self._scale = scale
        self._shift = shift
        self._settings = settings
    
    def plot(self, data, color, size=None, adjacency_matrix=None):
        if size is None:
            size = self._settings.size
        
        self._ax.grid(visible=True)
        self._ax.scatter(data[0], data[1], s=size, alpha=self._settings.alpha, color=color, marker=self._settings.marker)
        self._ax.set_xlim(self._settings.xlim * self._scale + self._shift[0])
        self._ax.set_ylim(self._settings.ylim * self._scale + self._shift[1])
        self._ax.set_aspect(self._settings.aspect)
        self._ax.tick_params(direction="in")
        
        if adjacency_matrix is not None:
            for i, row in enumerate(range(adjacency_matrix.shape[0])):
                for j, col in enumerate(range(adjacency_matrix.shape[1])):
                    if j > i:
                        if adjacency_matrix[i, j] > 0.0:
                            line = np.stack([data[0, [i, j]], data[1, [i, j]]])
                            self._ax.plot(
                                line[0], line[1], color="blue", alpha=0.9
                            )
                        if adjacency_matrix[i, j] < 0.0:
                            line = np.stack([data[0, [i, j]], data[1, [i, j]]])
                            self._ax.plot(
                                line[0], line[1], color="red", alpha=0.1
                            )



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

