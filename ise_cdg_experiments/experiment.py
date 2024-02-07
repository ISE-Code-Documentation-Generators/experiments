from abc import ABC, abstractmethod
import typing
from ise_cdg_utility.py import each
from torch import device as Device
from torch.utils.data import DataLoader

from tqdm import tqdm
from ise_cdg_experiments.interfaces import (
    BatchTrainerInterface,
    ExperimentVisitorInterface,
    ExperimentalModelInterface,
)


class BaseExperiment(ABC):

    def __init__(
        self,
        models: typing.List[ExperimentalModelInterface],
        device: Device,
        num_epochs: int,
        loader: DataLoader,
        trainer: BatchTrainerInterface,
        visitors: typing.Optional[typing.List[ExperimentVisitorInterface]] = None,
    ) -> None:
        super().__init__()
        self.models = models
        self.device = device
        self.num_epochs = num_epochs
        self.loader = loader
        self.trainer = trainer
        self._last_losses = None
        self.visitors = visitors or []

    def get_models(self) -> typing.List[ExperimentalModelInterface]:
        return self.models

    def should_save(self, epoch: int) -> bool:
        return epoch + 1 == self.num_epochs

    def train(self):
        models = self.get_models()
        for epoch in range(self.num_epochs):
            print(f"[Epoch {epoch + 1} / {self.num_epochs}]")

            each(lambda model: model.train(), models)

            for batch in tqdm(self.loader, desc="Training..."):

                self._last_losses = each(
                    lambda model: self.trainer.train_one_batch(
                        model,
                        model.get_batch_holder(batch),
                    ),
                    models,
                )

            # After each epoch
            self.accept_visitor(self.visitors)
            if self.should_save(epoch):
                each(
                    lambda model: model.save_checkpoint(),
                    models,
                )

    def accept_visitor(self, visitor: typing.List[ExperimentVisitorInterface]):
        for v in visitor:
            v.visit_experiment(self)
