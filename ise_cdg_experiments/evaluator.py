from abc import ABC
from typing import Any, List

import torch
from ise_cdg_experiments.interfaces import (
    ExperimentVisitorInterface,
    ModelTrainEvaluatorInterface,
)


class BaseModelTrainEvaluator(ModelTrainEvaluatorInterface, ABC):

    def __init__(self, model: Any, device: torch.device) -> None:
        super().__init__()

    def accept_visitors(self, visitors: List["ExperimentVisitorInterface"]):
        for v in visitors:
            v.visit_evaluator(self)
