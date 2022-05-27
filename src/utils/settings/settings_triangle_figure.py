from dataclasses import dataclass


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