from dataclasses import dataclass
from utils.experiments import settings
from typing import List, Any, Dict


@dataclass
class Result:
    identifier: str
    date: str
    experiment_type: str
    settings: settings.SettingsExperiment
    dataset: Any
    indices_train: List
    indices_validate: List
    point_estimates: Dict
