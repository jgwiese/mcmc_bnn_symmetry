from dataclasses import dataclass
import numpy as np
from typing import Any


@dataclass
class SettingsTraceFigure:
    prefix: str = r"\theta"
    ax_height: float = np.sqrt(2.0)
    t_width: float = 0.00125
    ylim: Any = np.array([-1.0, 1.0])
