from abc import ABC, abstractmethod
import typing
from typing import Any, List

import torch
from torch import optim, nn

from ise_cdg_experiments.experiment import BaseExperiment


class ExperimentalModelInterface(ABC):

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def __call__(self, *forward_inputs) -> None:
        pass

    @abstractmethod
    def save_checkpoint(self):
        pass

    @abstractmethod
    def load_checkpoint(self, remain_silent: bool = True):
        pass

    @abstractmethod
    def get_batch_holder(self, batch: typing.Any):
        pass


class ExperimentalBatchInterface(ABC):
    criterion_target: torch.Tensor
    forward_inputs: typing.Tuple

    @abstractmethod
    def __call__(self, batch: typing.Any) -> Any:
        pass


class BatchTrainerInterface(ABC):
    optimizer: optim.Optimizer
    criterion: nn.Module

    @abstractmethod
    def train_one_batch(
        self, model: typing.Any, batch_holder: ExperimentalBatchInterface
    ):
        pass


class ModelTrainEvaluatorInterface(ABC):
    model: typing.Any
    device: torch.device

    @abstractmethod
    def valid_for_save(self) -> bool:
        pass

    @abstractmethod
    def accept_visitors(self, visitors: List["ExperimentVisitorInterface"]):
        pass


class ExperimentVisitorInterface(ABC):

    @abstractmethod
    def visit_experiment(self, experiment: BaseExperiment):
        pass

    @abstractmethod
    def visit_evaluator(self, evaluator: ModelTrainEvaluatorInterface):
        pass
