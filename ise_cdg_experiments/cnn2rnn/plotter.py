import json
from typing import TYPE_CHECKING
from ise_cdg_experiments.interfaces import (
    ExperimentVisitorInterface,
)
from ise_cdg_utility.metrics import CodeMetric


if TYPE_CHECKING:
    from .evaluator import CNN2RNNBaseEvaluator
    from ise_cdg_experiments.experiment import BaseExperiment


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
            "rouge_on_eval": {
                "rouge1": {
                    "baseline": [],
                    "proposed": [],
                },
                "rouge2": {
                    "baseline": [],
                    "proposed": [],
                },
                "rougeL": {
                    "baseline": [],
                    "proposed": [],
                },
                "rougeLsum": {
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

    def __extract_rouge_metrics(
        self, evaluator: "CNN2RNNBaseEvaluator", rouge_type: str
    ):
        rouge_specifics = ["fmeasure", "precision", "recall"]
        result = []
        for specific in rouge_specifics:
            result.append(
                float(evaluator._metrics[CodeMetric.ROUGE][f"{rouge_type}_{specific}"])
            )
        return result

    def visit_evaluator(self, evaluator: "CNN2RNNBaseEvaluator"):
        if evaluator._metrics is None:
            return

        from ise_cdg_experiments.cnn2rnn import CNN2RNNEvaluator

        model = "baseline" if isinstance(evaluator, CNN2RNNEvaluator) else "proposed"
        for k, v in evaluator._metrics[CodeMetric.BLEU].items():
            self.add_to_plot(float(v), "bleu_on_eval", k, model)

        for k in self.to_plot["rouge_on_eval"].keys():
            self.add_to_plot(
                self.__extract_rouge_metrics(evaluator, k), "rouge_on_eval", k, model
            )

    def visit_experiment(self, experiment: "BaseExperiment"):
        if experiment._last_losses is None:
            return

        self.add_to_plot(experiment._last_losses[0], "loss", "baseline")
        self.add_to_plot(experiment._last_losses[1], "loss", "proposed")

    def save_result(self, filename):
        with open(filename, "w") as outfile:
            json.dump(self.to_plot, outfile)
