from typing import Any, Optional
from ise_cdg_utility.checkpoint import CheckpointInterface
from ise_cdg_experiments.experimental_model import ModelExperimentAdaptation
from ise_cdg_experiments.interfaces import (
    ExperimentalBatchInterface,
    ModelTrainEvaluatorInterface,
)


class CNN2RNNExperimentalModel(ModelExperimentAdaptation):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._save_checkpoint_call = 0

    def save_checkpoint(self):
        return True
        self._save_checkpoint_call += 1
        if not self._save_checkpoint_call % 2:
            return super().save_checkpoint()
        return False
