from dataclasses import dataclass
import utils.experiments.settings as settings


@dataclass
class SettingsExperimentSample(settings.SettingsExperiment):
    num_warmup: int
    statistic: str
    statistic_p: float
    identifiable_modes: int
    samples_per_chain: int
    seed: int
    overwrite_chains: int = None

