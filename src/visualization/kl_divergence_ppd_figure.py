import matplotlib.pyplot as plt
from visualization import settings
import jax.numpy as jnp


class KLDivergencePPDFigure():
    def __init__(self, settings: settings.SettingsKLDivergencePPDFigure = settings.SettingsKLDivergencePPDFigure):
        self._settings = settings
        self._figure = None

    def __del__(self):
        plt.close(self._figure)

    def plot(self, kl_divergences):
        if self._figure is not None:
            self._figure.clf()

        self._figure = plt.figure(figsize=(24, 4))
        ax1 = self._figure.add_subplot(1, 1, 1)
        ax1.plot(jnp.arange(len(kl_divergences)), jnp.array(kl_divergences), color="black")
        ax1.set_xlabel("samples")
        ax1.set_ylabel(r"avg. KL-Div.", color="black")
        ax1.yaxis.label.set_size(self._settings.textsize)
        ax1.xaxis.label.set_size(self._settings.textsize)
        ax1.tick_params(axis='both', which='major', labelsize=self._settings.textsize - 2)
        ax1.tick_params(axis='both', which='minor', labelsize=self._settings.textsize - 2)

        ax2 = ax1.twinx()
        ax2.plot(jnp.arange(len(kl_divergences)), jnp.log(jnp.array(kl_divergences)), color="tab:blue")
        ax2.set_ylabel(r"log", color="tab:blue")
        ax2.yaxis.label.set_size(self._settings.textsize)
        ax2.xaxis.label.set_size(self._settings.textsize)
        ax2.tick_params(axis='both', which='major', labelsize=self._settings.textsize - 2)
        ax2.tick_params(axis='both', which='minor', labelsize=self._settings.textsize - 2)

        return self._figure

