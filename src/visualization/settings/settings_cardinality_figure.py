from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class SettingsCardinalityFigure:
    ax_width: float = jnp.sqrt(2.0)
    ax_height: float = jnp.sqrt(2.0)
    textsize: int = 10 # 22

