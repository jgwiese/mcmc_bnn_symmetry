from dataclasses import dataclass


@dataclass
class SettingsExperimentSample:
    output_path: str
    dataset: str
    dataset_normalization: str
    hidden_layers: int
    hidden_neurons: int
    activation: str
    activation_last_layer: str
    num_warmup: int
    statistic: str
    statistic_p: float
    samples_per_mode: int
    identifiable_modes: int
    pool_size: int
    seed: int
