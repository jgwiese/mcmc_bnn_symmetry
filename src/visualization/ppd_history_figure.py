import matplotlib.pyplot as plt
from visualization import settings
import jax.numpy as jnp


class PPDHistoryFigure():
    def __init__(self, settings: settings.SettingsPPDHistoryFigure = settings.SettingsPPDHistoryFigure):
        self._settings = settings
        self._figure = None

    def __del__(self):
        plt.close(self._figure)

    def plot(self, experiment, ppd_history, indices):
        if self._figure is not None:
            self._figure.clf()
       
        resolution_x = ppd_history[0].shape[0]
        resolution_y = ppd_history[0].shape[1]
        x = jnp.linspace(self._settings.xlim[0], self._settings.xlim[1], resolution_x)
        y = jnp.linspace(self._settings.ylim[0], self._settings.ylim[1], resolution_y)
        xx, yy = jnp.meshgrid(x.squeeze(), y.squeeze())

        cols = 4
        rows = int(1.0 * len(indices) / cols)
        if (rows * cols) < len(indices):
            rows += 1

        self._figure = plt.figure(figsize=(4 * cols, 4 * rows))
        for row in range(rows):
            for col in range(cols):
                i = row * cols + col
                if i >= len(indices):
                    continue
                posterior = indices[i]
                ax = self._figure.add_subplot(rows, cols, i + 1)
                ax.set_xlim(self._settings.xlim[0], self._settings.xlim[1])
                ax.set_ylim(self._settings.ylim[0], self._settings.ylim[1])
                ax.set_title(f"{posterior + 1} samples", **self._settings.csfont)
                
                if col == 0:
                    ax.set_ylabel("y")
                if row == rows - 1:
                    ax.set_xlabel("x")

                maximum, minimum = ppd_history[posterior].max(), ppd_history[posterior].min()
                values = 1.0 - (ppd_history[posterior] - minimum) / (maximum - minimum)
                values = ppd_history[posterior]
                ax.pcolormesh(xx, yy, 0.2**(values.T + 1e-6), cmap="Blues_r", shading="gouraud", rasterized=True)
                ax.scatter(experiment._dataset.data[:, 0], experiment._dataset.data[:, 1], c="black", s=1, rasterized=True)
                ax.yaxis.label.set_size(self._settings.textsize)
                ax.xaxis.label.set_size(self._settings.textsize)
                ax.tick_params(axis='both', which='major', labelsize=self._settings.textsize - 2)
                ax.tick_params(axis='both', which='minor', labelsize=self._settings.textsize - 2)
                ax.title.set_size(self._settings.textsize)
                ax.set_aspect("equal") 

        return self._figure


