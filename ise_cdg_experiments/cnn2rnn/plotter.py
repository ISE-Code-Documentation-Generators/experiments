import json
from typing import TYPE_CHECKING
from ise_cdg_experiments.experiment import BaseExperiment
from ise_cdg_experiments.interfaces import (
    ExperimentVisitorInterface,
)
from ise_cdg_utility.metrics import CodeMetric


if TYPE_CHECKING:
    from .evaluator import CNN2RNNBaseEvaluator


class CNN2RNNPlotter(ExperimentVisitorInterface):
    def __init__(self) -> None:
        self.to_plot = {
            "loss": {
                "baseline": [],
                "proposed": [],
            },
            "bleu_on_eval": {
                "bleu_1": {
                    "baseline": [],
                    "proposed": [],
                },
                "bleu_2": {
                    "baseline": [],
                    "proposed": [],
                },
                "bleu_3": {
                    "baseline": [],
                    "proposed": [],
                },
                "bleu_4": {
                    "baseline": [],
                    "proposed": [],
                },
            },
        }

    def add_to_plot(self, item, *keys):
        val = None
        for key in keys:
            if val is None:
                val = self.to_plot[key]
            else:
                val = val[key]
        val.append(item)

    def get_plot(self):
        return self.to_plot

    def visit_evaluator(self, evaluator: "CNN2RNNBaseEvaluator"):
        from ise_cdg_experiments.cnn2rnn import CNN2RNNEvaluator

        model = "baseline" if isinstance(evaluator, CNN2RNNEvaluator) else "proposed"
        for k, v in evaluator._metrics[CodeMetric.BLEU].items():
            self.plotter.add_to_plot(float(v), "bleu_on_eval", k, model)

    def visit_experiment(self, experiment: BaseExperiment):
        self.add_to_plot(experiment._last_losses[0], "loss", "baseline")
        self.add_to_plot(experiment._last_losses[1], "loss", "proposed")
