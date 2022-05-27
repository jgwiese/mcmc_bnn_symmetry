class TransformationPlot:
    def __init__(self, ax, scale, settings: TransformationPlotSettings = TransformationPlotSettings()):
        self._ax = ax
        self._scale = scale
        self._settings = settings
    
    def plot(self, transformation, parameters, std, color, dataset):
        if len(dataset.conditional_indices) == 1:
            self._ax.set_xlim(self._settings.xlim * self._scale)
            self._ax.set_ylim(self._settings.ylim * self._scale)
            self._ax.set_aspect(self._settings.aspect)
            
            inputs = np.linspace(self._settings.xlim[0] * self._scale, self._settings.xlim[1] * self._scale, 128)[:, np.newaxis]
            means = jax.vmap(transformation, in_axes=(None, 0))(inputs, parameters).squeeze(-1)
            for mean in tqdm(means):
                self._ax.plot(inputs[:, 0], mean, c=color, alpha=self._settings.alpha, linewidth=self._settings.linewidth)
                if self._settings.aleatoric and not self._settings.epistemic:
                    self._ax.fill_between(inputs[:, 0], mean - 1.96 * std, mean + 1.96 * std, color=color, alpha=self._settings.alpha_std, label=r"$1.96 \cdot \sigma_a$")
            if self._settings.epistemic:
                mean = np.mean(means, axis=0)
                epistemic_std = np.std(means, axis=0)
                if self._settings.aleatoric:
                    std = np.sqrt(np.power(epistemic_std, 2) + np.power(std, 2))
                    self._ax.fill_between(inputs[:, 0], mean - 1.96 * std, mean + 1.96 * std, color=color, alpha=self._settings.alpha_std, label=r"$1.96 \cdot \sqrt{\sigma_e^2 + \sigma_a^2}$")
                else:
                    std = epistemic_std
                    self._ax.fill_between(inputs[:, 0], mean - 1.96 * std, mean + 1.96 * std, color=color, alpha=self._settings.alpha_std, label=r"$1.96 \cdot \sigma_e}$")
        elif len(dataset.conditional_indices) == 2:
            x = np.linspace(self._settings.xlim[0] * self._scale, self._settings.xlim[1] * self._scale, 128)[:, np.newaxis]
            xx, yy = np.meshgrid(x, x)
            outputs = jax.vmap(transformation, in_axes=(None, 0))(np.stack([xx, yy], -1).reshape((-1, 2)), parameters).squeeze(-1).reshape((-1, 128, 128))
            for output in outputs:
                self._ax.plot_surface(xx, yy, output, alpha=self._settings.alpha, color=color)
        else:
            return 1

    def plot_means(self, transformation, parameters, color, dataset):
        inputs = np.linspace(self._settings.xlim[0] * self._scale, self._settings.xlim[1] * self._scale, 128)[:, np.newaxis]
        means = jax.vmap(transformation, in_axes=(None, 0))(inputs, parameters).squeeze(-1)
        self._ax.plot(inputs[:, 0], means.mean(0), c="gray", linestyle="solid", linewidth=self._settings.linewidth, alpha=1.0)
        self._ax.plot(inputs[:, 0], means.mean(0), c=color, linestyle="dashed", linewidth=self._settings.linewidth, alpha=1.0, label=r"$\mathbb{E}[f]$")
