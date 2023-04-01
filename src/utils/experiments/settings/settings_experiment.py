from dataclasses import dataclass


@dataclass
class SettingsExperiment:
    output_path: str
    dataset: str
    dataset_normalization: str
    hidden_layers: int
    hidden_neurons: int
    activation: str
    activation_last_layer: str
    pool_size: int

