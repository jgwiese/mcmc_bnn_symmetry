class ScatterPlot:
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

"""
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
"""