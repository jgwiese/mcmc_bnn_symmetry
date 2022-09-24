from utils import settings
from data import datasets
import numpy as np


class Scatter:
    def __init__(self, ax, scale: float = 1.0, shift: np.array = np.zeros(2), settings: settings.SettingsScatter = settings.SettingsScatter()):
        self._ax = ax
        self._scale = scale
        self._shift = shift
        self._settings = settings
    
    def plot(self, dataset: datasets.ConditionalDataset, color=None, size=None, adjacency_matrix=None, feature=0):
        if color is None:
            color = self._settings.color
        if size is None:
            size = self._settings.size
        
        conditional = dataset.data[:, dataset.conditional_indices]
        if conditional.shape[-1] > 2:
            conditional = conditional[:, feature:feature+1]
        dependent = dataset.data[:, dataset.dependent_indices]
        if conditional.shape[-1] == 1:
            self._ax.scatter(conditional, dependent, color=color, alpha=self._settings.alpha, s=size)
            self._ax.set_xlim(self._settings.xlim * self._scale + self._shift[0])
            self._ax.set_ylim(self._settings.ylim * self._scale + self._shift[1])
            self._ax.set_aspect(self._settings.aspect)
            self._ax.tick_params(direction="in")
            self._ax.grid(visible=True)

            # TODO: not tested after refactoring.
            if adjacency_matrix is not None:
                for i, row in enumerate(range(adjacency_matrix.shape[0])):
                    for j, col in enumerate(range(adjacency_matrix.shape[1])):
                        if j > i:
                            if adjacency_matrix[i, j] > 0.0:
                                line = np.stack([dataset.data[[i, j], 0], dataset.data[[i, j], 1]])
                                self._ax.plot(
                                    line[0], line[1], color="blue", alpha=0.9
                                )
                            if adjacency_matrix[i, j] < 0.0:
                                line = np.stack([dataset.data[[i, j], 0], dataset.data[[i, j], 1]])
                                self._ax.plot(
                                    line[0], line[1], color="red", alpha=0.1
                                )
        
        elif conditional.shape[-1] == 2:
            self._ax.scatter(conditional.T[0], conditional.T[1], dependent.T[0], color=color, alpha=self._settings.alpha, s=size)
        else:
            pass
