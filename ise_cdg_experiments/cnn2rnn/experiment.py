from typing import List
from torch import device
from torch.utils.data import DataLoader
from ise_cdg_experiments.cnn2rnn import CNN2RNNBatchTrainer
from ise_cdg_experiments.experiment import BaseExperiment
from ise_cdg_experiments.interfaces import (
    ExperimentalModelInterface,
)


class CNN2RNNExperiment(BaseExperiment):

    def __init__(
        self,
        models: List[ExperimentalModelInterface],
        device: device,
        num_epochs: int,
        loader: DataLoader,
        trainer: CNN2RNNBatchTrainer,
    ) -> None:
        super().__init__(models, device, num_epochs, loader, trainer)

    def should_save(self, epoch: int) -> bool:
        return not ((epoch + 1) % 10)
