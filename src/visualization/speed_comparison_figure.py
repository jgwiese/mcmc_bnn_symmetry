import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import gridspec
from utils import settings


class SpeedComparisonFigure:
    def __init__(self, settings: settings.SettingsSpeedComparisonFigure = settings.SettingsSpeedComparisonFigure()):
        self._settings = settings
        
        # setup sample transformation
        if self._settings.sample_transformation == "log2":
            self._sample_transformation = jnp.log2
        elif self._settings.sample_transformation == "log":
            self._sample_transformation = jnp.log
        elif self._settings.sample_transformation == "none":
            self._sample_transformation = lambda x: x
        else:
            print("sample transformation not supported, switching to identity")
            self._sample_transformation = lambda x: x
        
        # setup sample transformation
        if self._settings.time_transformation == "log2":
            self._time_transformation = jnp.log2
        elif self._settings.time_transformation == "log":
            self._time_transformation = jnp.log
        elif self._settings.time_transformation == "none":
            self._time_transformation = lambda x: x
        else:
            print("time transformation not supported, switching to identity")
            self._time_transformation = lambda x: x
        
        # time scale
        if self._settings.time_unit == "ms":
            self._time_scale = 1000.0
        else:
            print("time unit not supported, switching to seconds")
            self._time_scale = 1.0
        
        self._create_figure()
    
    def _create_figure(self):
        self._figure = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])
        self._ax = self._figure.add_subplot(gs[1])
        self._ax.set_xlabel("sample count [{}(#)]".format(self._settings.sample_transformation))
        self._ax.set_ylabel("time per sample [{}({})]".format(self._settings.time_transformation if self._settings.time_transformation != "none" else "", self._settings.time_unit))
        self._ax_2 = self._figure.add_subplot(gs[0])
        self._ax_2.set_ylabel("speedup")
        #self._ax_2.set_ylim([-100, 100])
        self._ax_2.set_xticklabels([])
        self._ax_2.set_xticks([])
        plt.subplots_adjust(wspace=0.0, hspace=0.02)
    
    def __del__(self):
        self.clear()
    
    def plot_acc(self, data: jnp.array, label: str, speedup: bool = False):
        if self._figure is None:
            self._create_figure()
        
        if speedup:
            # speedup plot
            total_samples = self._sample_transformation(data[:, 0])
            maximum = jnp.max(total_samples)
            self._ax_2.axhline(y=1.0, color="gray", linestyle="--")
            self._ax_2.plot(total_samples, data[:, 1], c="black")
            self._ax_2.scatter(total_samples, data[:, 1], marker='x', c="black")
            return self._figure
        
        # data plot
        total_samples = self._sample_transformation(data[:, 0])
        sample_times_means = self._time_transformation(self._time_scale * data[:, 1])
        sample_times_stds = self._time_transformation(self._time_scale * data[:, 2])
        self._ax.plot(total_samples, sample_times_means, label=label)
        for i, element in enumerate(total_samples):
            x = jnp.stack([element] * 20)
            y = self._time_transformation(self._time_scale * data[i, 3:-1])
            self._ax.scatter(x, y, c="black", alpha=0.2)
            self._ax.scatter(total_samples[i], sample_times_means[i], c="black")
            self._ax.annotate(text="{:.2f}".format(sample_times_means[i]), xy=(total_samples[i] + 0.05, sample_times_means[i] + 0.1), xycoords="data")
            self._ax.errorbar(total_samples[i], sample_times_means[i], sample_times_stds[i], c="black")

        self._ax.legend()
        return self._figure
    
    def clear(self):
        plt.close(self._figure)
        del self._figure
        self._figure = None
