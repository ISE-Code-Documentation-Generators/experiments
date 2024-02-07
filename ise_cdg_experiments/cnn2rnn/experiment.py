from typing import List
from torch import device
from torch.utils.data import DataLoader
from ise_cdg_experiments.cnn2rnn import CNN2RNNBatchTrainer
from ise_cdg_experiments.experiment import BaseExperiment
from ise_cdg_experiments.interfaces import (
    ExperimentalModelInterface,
)


class CNN2RNNExperiment(BaseExperiment):

    def should_save(self, epoch: int) -> bool:
        return True or not ((epoch + 1) % 10)
