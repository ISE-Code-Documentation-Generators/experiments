from abc import ABC, abstractmethod
import typing
from typing import Any

import torch


class ExperimentalModelInterface(ABC):

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def __call__(self, *forward_inputs) -> None:
        pass

    # TODO
    # @abstractmethod
    # def generate_one_markdown(self, *args: Any, **kwargs: Any) -> Any:
    #     return self.model.generate_one_markdown(*args, **kwargs)

    @abstractmethod
    def save_checkpoint(self):
        pass

    @abstractmethod
    def load_checkpoint(self, remain_silent: bool = True):
        pass


class ExperimentalBatchInterface(ABC):
    criterion_target: torch.Tensor
    forward_inputs: typing.Tuple

    @abstractmethod
    def __call__(self, batch: typing.Any) -> Any:
        pass


class BatchTrainerInterface(ABC):
    
    @abstractmethod
    def train_one_batch(self, model: typing.Any, criterion_target, forward_inputs):
        pass

