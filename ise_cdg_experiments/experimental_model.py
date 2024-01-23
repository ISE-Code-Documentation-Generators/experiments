import typing
from typing import Any


from ise_cdg_utility.checkpoint import CheckpointInterface

from ise_cdg_experiments.interfaces import ExperimentalModelInterface


class ModelExperimentAdaptation(ExperimentalModelInterface):

    def __init__(
        self, 
        model: typing.Any,
        checkpoint: CheckpointInterface, 
        evaluator: typing.Any = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.checkpoint = checkpoint
        self.evaluator = evaluator

    def train(self, *args: Any, **kwargs: Any):
        return self.model.train(*args, **kwargs)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.model(*args, **kwds)

    def generate_one_markdown(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.generate_one_markdown(*args, **kwargs)
    
    def save_checkpoint(self):
        if self.evaluator and self.evaluator.valid_for_save():
            self.checkpoint.save_checkpoint()

    def load_checkpoint(self, remain_silent: bool = True):
        try:
            self.checkpoint.load_checkpoint()
        except FileNotFoundError as e:
            if not remain_silent:
                raise e
            print("No file for checkpoint found on the directory!")
