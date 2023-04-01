import matplotlib.pyplot as plt
from visualization import settings
import jax.numpy as jnp


class CardinalityFigure():
    def __init__(self, settings: settings.SettingsCardinalityFigure = settings.SettingsCardinalityFigure):
        self._settings = settings
        self._figure = None

    def __del__(self):
        plt.close(self._figure)

    def plot(self, cardinalities: jnp.array):
        if self._figure is not None:
            self._figure.clf()
        self._figure = plt.figure(figsize=(self._settings.ax_width * 8, self._settings.ax_height * 4), tight_layout=True)

        ax = self._figure.add_subplot(1, 1, 1)
        ax.set_xlabel("neurons")
        ax.set_ylabel("cardinalities (log)")
        ax.set_yscale("log")
        ax.yaxis.label.set_size(self._settings.textsize)
        ax.xaxis.label.set_size(self._settings.textsize)
        plt.xticks(fontsize=self._settings.textsize - 2)
        plt.yticks(fontsize=self._settings.textsize - 2)
        
        ax.plot(jnp.arange(len(cardinalities)), cardinalities[:, 1], linestyle="dashed", label=r"$\mathcal{T}$")
        ax.plot(jnp.arange(len(cardinalities)), cardinalities[:, 0], linestyle="dashed", label=r"$\mathcal{P}$")
        ax.plot(jnp.arange(len(cardinalities)), cardinalities[:, 2], linestyle="solid", label=r"$\mathcal{E}$")
        ax.legend(prop={"size": self._settings.textsize})

        return self._figure
        
