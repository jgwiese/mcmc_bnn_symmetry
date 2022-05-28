from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import settings
import numpy as np
from visualization import Scatter, Plot


class SequenceFigure:
    def __init__(self, settings: settings.SettingsSequenceFigure = settings.SettingsSequenceFigure()):
        self._settings = settings
        self._figure = None
    
    def __del__(self):
        plt.close(self._figure)
    
    def plot(self, dataset, transformation, parameters_paths):
        cols = self._settings.cols
        rows = int(len(parameters_paths) / cols)

        if self._figure is not None:
            self._figure.clf()
        self._figure = plt.figure(figsize=(self._settings.ax_width * cols, self._settings.ax_height * rows))

        for row in range(rows):
            for col in range(cols):
                i = row * cols + col
                if i > len(parameters_paths):
                    break
                ax = self._figure.add_subplot(rows, cols, i + 1)
                if len(dataset.conditional_indices) == 1:
                    if col == 0:
                        ax.set_ylabel("y")
                    if row == rows - 1:
                        ax.set_xlabel("x")
                
                label, parameters_list = parameters_paths[i]
                parameters_list = [parameters_list]
                ax.set_title("iteration {}".format(label))

                if dataset is not None:
                    scale = np.std(dataset.data) * 3.0
                    data_plot = Scatter(ax, scale=scale, settings=self._settings.settings_scatter)
                    data_plot.plot(dataset=dataset)
                transformation_plot = Plot(ax, scale=scale, settings=self._settings.settings_plot)
                for j, parameters in enumerate(parameters_list):
                    color = self._settings.settings_plot.cmap(1.0 * j / len(parameters_list))
                    transformation_plot.plot(transformation, parameters, std=None, color=color, dataset=dataset)
                for j, parameters in enumerate(parameters_list):
                    color = self._settings.settings_plot.cmap(1.0 * j /len(parameters_list))
                    transformation_plot.plot_means(transformation, parameters, color=color, dataset=dataset)
        return self._figure
