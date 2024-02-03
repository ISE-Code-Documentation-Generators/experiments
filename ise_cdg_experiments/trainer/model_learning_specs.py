import torch
from torch import optim
from torch import nn

from ise_cdg_utility.checkpoint import get_model_checkpoint


class ModelLearningSpecs:
    def __init__(self, model, learning_rate, md_vocab) -> None:
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        pad_idx = md_vocab.get_stoi()["<pad>"]
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def train_one_batch(self, inputs, expected_md) -> "torch.Tensor":
        md_for_criterion = expected_md[1:].reshape(-1)
        output = self.model(*inputs)
        output = output[1:].reshape(-1, output.shape[2])

        self.optimizer.zero_grad()
        loss = self.criterion(output, md_for_criterion)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
        self.optimizer.step()
        return loss

    def get_checkpoint(self, workspace: "str", *args, **kwargs):
        return get_model_checkpoint(
            workspace, self.model, self.optimizer, *args, **kwargs
        )
