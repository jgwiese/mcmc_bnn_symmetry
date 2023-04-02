from dataclasses import dataclass
import numpy as np
from typing import Any, Dict


@dataclass
class SettingsPPDHistoryFigure:
    xlim: Any = np.array([-3.0, 3.0])
    ylim: Any = np.array([-3.0, 3.0])
    textsize: int = 10 # 20
    csfont = {'fontname':'Times New Roman'}

