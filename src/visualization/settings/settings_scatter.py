from dataclasses import dataclass
import numpy as np
from typing import Any


@dataclass
class SettingsScatter:
    xlim: Any = np.array([-1.0, 1.0])
    ylim: Any = np.array([-1.0, 1.0])
    aspect: str = "equal"
    color: str = "black"
    alpha: float = 1.0
    size: float = 4.0
    grid: bool = True

