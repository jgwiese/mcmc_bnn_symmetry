from dataclasses import dataclass


@dataclass
class RegressionSettings:
    data_plot_settings: DataPlotSettings = DataPlotSettings()
    transformation_plot_settings: TransformationPlotSettings = TransformationPlotSettings()
    ax_width: float = 12.0
    ax_height: float = 4.0
