"""Visualizations."""

import os
from typing import Dict, List

import pandas as pd
import seaborn as sns

from src.utils import load_json


class Plotter:
    """Class to create plots."""

    def __init__(self):
        """Instantiate plotter."""
        self.plots = {}

    def make_plots(self, result_file: str, plot_type: str) -> None:
        """Create plots from results."""
        results = load_json(result_file)
        datasets = list(results.keys())
        learners = list(results[datasets[0]].keys())
        self._make_barplot(results, datasets, learners, plot_type)

    def save_plots(self, save_dir: str, file_ext: str = 'png') -> None:
        """Save created plots."""
        for name, plot in self.plots.items():
            plot.figure.savefig(os.path.join(save_dir, f'{name}.{file_ext}'))

    def _make_barplot(
        self,
        results: Dict,
        datasets: List[str],
        learners: List[str],
        metric: str,
    ) -> None:
        """Plot metric for different learners and datasets."""
        if metric in ['rmse', 'nll']:
            rows = []
            for ds in datasets:
                for bl in learners:
                    rows.append([ds, bl, results[ds][bl][metric]])
            df = pd.DataFrame(rows, columns=['dataset', 'bl', metric])
            sns.set_theme()
            plot = sns.barplot(
                data=df,
                x='dataset',
                y=metric,
                hue='bl',
            )
            plot.tick_params(axis='x', rotation=90)
            plot.figure.set_figheight(10)
            plot.figure.set_figwidth(12)
            self.plots.update({metric: plot})

        else:
            raise NotImplementedError(f'Metric {metric} not supported.')
