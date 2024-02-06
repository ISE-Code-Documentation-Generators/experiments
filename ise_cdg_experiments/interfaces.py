from abc import ABC, abstractmethod
import typing
from typing import Any, List

import torch
from torch import optim, nn

if typing.TYPE_CHECKING:
    from ise_cdg_experiments.experiment import BaseExperiment


class ExperimentalOptimizer:

    def __init__(self, experimental_model: "ExperimentalModelInterface") -> None:
        self.experimental_model = experimental_model

    def __enter__(self):
        return self.experimental_model.optimizer.zero_grad()
        # You can return an object that will be assigned to the variable after 'as'
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        self.experimental_model.optimizer.step()
        # Return True to suppress the exception, or False to propagate it
        return False


class ExperimentalModelInterface(ABC):
    optimizer: optim.Optimizer

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

    @abstractmethod
    def optimize(self) -> "ExperimentalOptimizer":
        pass


class ExperimentalBatchInterface(ABC):
    criterion_target: torch.Tensor
    forward_inputs: typing.Tuple

    @abstractmethod
    def __call__(self, batch: typing.Any) -> Any:
        pass


class BatchTrainerInterface(ABC):
    criterion: nn.Module

    @abstractmethod
    def train_one_batch(
        self,
        model: "ExperimentalModelInterface",
        batch_holder: "ExperimentalBatchInterface",
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
    def visit_experiment(self, experiment: "BaseExperiment"):
        pass

    @abstractmethod
    def visit_evaluator(self, evaluator: "ModelTrainEvaluatorInterface"):
        pass
