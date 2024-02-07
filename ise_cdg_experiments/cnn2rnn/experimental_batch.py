from abc import ABC, abstractmethod
import typing
from typing import Any
import torch
from ise_cdg_experiments.interfaces import ExperimentalBatchInterface
from ise_cdg_utility import to_device


class CNN2RNNBaseExperimentalBatch(ExperimentalBatchInterface, ABC):

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self._criterion_target: typing.Optional[torch.Tensor] = None
        self._forward_inputs: typing.Optional[typing.Tuple] = None

    @property
    def criterion_target(self) -> torch.Tensor:
        assert (
            self._criterion_target is not None
        ), "you should initially deploy a batch then use it."
        return self._criterion_target

    @property
    def forward_inputs(self) -> typing.Tuple:
        assert (
            self._forward_inputs is not None
        ), "you should initially deploy a batch then use it."
        return self._forward_inputs

    @abstractmethod
    def set_forward_inputs(self, src, features, md):
        pass

    def __call__(self, batch: Any) -> Any:
        src, features, md = batch
        src, features, md = to_device(self.device, src, features, md)
        self._criterion_target = md[1:].reshape(-1)
        self.set_forward_inputs(src, features, md)


class CNN2RNNExperimentalBatch(CNN2RNNBaseExperimentalBatch):

    def set_forward_inputs(self, src, features, md):
        self._forward_inputs = (src, md, self.device)


class CNN2RNNFeaturesExperimentalBatch(CNN2RNNBaseExperimentalBatch):

    def set_forward_inputs(self, src, features, md):
        self._forward_inputs = (src, features, md, self.device)
