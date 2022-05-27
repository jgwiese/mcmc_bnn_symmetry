from dataclasses import dataclass


@dataclass
class SettingsScatter:
    xlim: Any = np.array([-1.0, 1.0])
    ylim: Any = np.array([-1.0, 1.0])
    aspect: str = "equal"
    color: str = "black"
    alpha: float = 1.0
    size: float = 4.0

"""
@dataclass
class ScatterSettings:
    xlim: Any = np.array([-1.0, 1.0])
    ylim: Any = np.array([-1.0, 1.0])
    aspect: str = "equal"
    alpha: float = 1.0
    size: float = 1.0
    marker: str = "."
"""