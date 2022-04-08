from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import List, Any
from tqdm import tqdm
import jax.numpy as jnp


@dataclass
class TraceSettings:
    prefix: str = r"\theta"
    ax_height: float = jnp.sqrt(2.0)
    t_width: float = 0.00125
    ylim: Any = jnp.array([-1.0, 1.0])

class TraceFigure:
    def __init__(self, settings: TraceSettings = TraceSettings()):
        self._settings = settings
        self._figure = None

    def __del__(self):
        plt.close(self._figure)
    
    def plot(self, chains: List):
        rows = chains[0].shape[-1]
        if self._figure is not None:
            self._figure.clf()
        self._figure = plt.figure(figsize=(self._settings.t_width * len(chains[0]), rows * self._settings.ax_height))

        # data scale for x/y limits
        scale = jnp.array(chains).std() * 1.96

        for row in tqdm(range(rows)):
            ax = self._figure.add_subplot(rows, 1, row + 1)
            ax.grid(visible=True)
            ax.tick_params(direction="in")
            ax.set_xlim([0, len(chains[0])])
            ax.set_ylim(self._settings.ylim * scale)
            ax.set_ylabel(f"${self._settings.prefix}_{{{row}}}$")
            
            if row != rows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("t")
                
            for chain in chains:
                ax.plot(jnp.arange(len(chain)), chain[:, row], linewidth=0.1, color="black")
        return self._figure
        
