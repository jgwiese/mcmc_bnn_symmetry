from visualization import settings
import seaborn as sns


class UnivariateKDEPlot:
    def __init__(self, ax, scale, shift, settings: settings.SettingsPlot):
        self._ax = ax
        self._scale = scale
        self._shift = shift
        self._settings = settings
    
    def plot(self, data, color, rasterized=False):
        sns.kdeplot(x=data, ax=self._ax, fill=False, color=color, alpha=self._settings.alpha, linewidth=1.0, warn_singular=False, rasterized=rasterized)
        self._ax.set_xlim(self._settings.xlim * self._scale + self._shift)
        self._ax.set_ylabel("")
        self._ax.tick_params(direction="in")
