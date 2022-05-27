from dataclasses import dataclass


@dataclass
class SettingsTraceFigure:
    prefix: str = r"\theta"
    ax_height: float = jnp.sqrt(2.0)
    t_width: float = 0.00125
    ylim: Any = jnp.array([-1.0, 1.0])
