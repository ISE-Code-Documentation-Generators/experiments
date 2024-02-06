import typing
from ise_cdg_experiments.interfaces import (
    BatchTrainerInterface,
    ExperimentalBatchInterface,
)
from torch import optim
from torch import nn


class CNN2RNNBatchTrainer(BatchTrainerInterface):

    def __init__(
        self,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.criterion = criterion

    def train_one_batch(
        self, model: typing.Any, batch_holder: ExperimentalBatchInterface
    ):
        output = model(*batch_holder.forward_inputs)
        output = output[1:].reshape(-1, output.shape[2])

        self.optimizer.zero_grad()
        loss = self.criterion(output, batch_holder.criterion_target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        self.optimizer.step()
        return loss
