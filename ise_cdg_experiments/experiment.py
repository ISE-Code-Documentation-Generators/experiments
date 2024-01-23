from abc import ABC, abstractmethod
import typing
from ise_cdg_utility.py import each
from torch import device as Device
from torch.utils.data import DataLoader

from tqdm import tqdm
from ise_cdg_experiments.interfaces import BatchTrainerInterface, ExperimentalBatchInterface, ExperimentalModelInterface



class BaseExperiment(ABC):

    def __init__(
        self,
        models: typing.List[ExperimentalModelInterface],
        device: Device,
        num_epochs: int,
        loader: DataLoader,
        batch_holder: ExperimentalBatchInterface,
        trainer: BatchTrainerInterface,
    ) -> None:
        super().__init__()
        self.models = models
        self.device = device
        self.num_epochs = num_epochs
        self.loader = loader
        self.batch_holder = batch_holder
        self.trainer = trainer

    def get_models(self) -> typing.List[ExperimentalModelInterface]:
        return self.models

    @abstractmethod
    def should_save(self, epoch: int) -> bool:
        pass

    def train(self):
        models = self.get_models()
        for epoch in range(self.num_epochs):
            print(f"[Epoch {epoch + 1} / {self.num_epochs}]")

            each(lambda model: model.train(), models)

            for batch in tqdm(self.loader, desc="Training..."):
                self.batch_holder(batch)

                losses = each(
                    lambda model: self.trainer.train_one_batch(
                        model,
                        self.batch_holder.criterion_target,
                        self.batch_holder.forward_inputs,
                    ),
                    models,
                )

            if self.should_save(epoch):
                each(
                    lambda model: model.save_checkpoint(),
                    models,
                )
