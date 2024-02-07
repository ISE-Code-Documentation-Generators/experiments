import typing
from typing import Any, Optional

from torch import optim

from ise_cdg_utility.checkpoint import CheckpointInterface
import torch

from ise_cdg_experiments.interfaces import (
    ExperimentalBatchInterface,
    ExperimentalModelInterface,
    ExperimentalOptimizer,
    ModelTrainEvaluatorInterface,
)


class ModelExperimentAdaptation(ExperimentalModelInterface):

    def __init__(
        self,
        model: typing.Any,
        optimizer: optim.Optimizer,
        checkpoint: CheckpointInterface,
        batch_holder: ExperimentalBatchInterface,
        evaluator: Optional[ModelTrainEvaluatorInterface] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.checkpoint = checkpoint
        self.evaluator = evaluator
        self.batch_holder = batch_holder
        self.optimizer = optimizer

    def train(self, *args: Any, **kwargs: Any):
        return self.model.train(*args, **kwargs)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.model(*args, **kwds)

    def generate_one_markdown(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.generate_one_markdown(*args, **kwargs)

    def save_checkpoint(self):
        if True or (self.evaluator and self.evaluator.valid_for_save()):
            self.checkpoint.save_checkpoint()

    def load_checkpoint(self, remain_silent: bool = True):
        try:
            self.checkpoint.load_checkpoint()
        except FileNotFoundError as e:
            if not remain_silent:
                raise e
            print("No file for checkpoint found on the directory!")

    def get_batch_holder(self, batch: typing.Any):
        self.batch_holder(batch)
        return self.batch_holder
    
    def optimize(self) -> ExperimentalOptimizer:
        return ExperimentalOptimizer(self)
    
    def parameters(self) -> torch.Tensor:
        return self.model.parameters()
