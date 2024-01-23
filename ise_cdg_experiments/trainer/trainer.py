from typing import TYPE_CHECKING
import torch
from torch import nn
from tqdm import tqdm
from .early_stop_detector import EarlyStopDetector
from ise_cdg_utility import to_device
from ise_cdg_utility.metrics import CodeMetric

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

class Trainer:
    def __init__(
        self,
        checkpoint_rate: "int",
        num_epochs: "int",
        proposed_model: "torch.nn.Module",
        proposed_optimizer, # TODO: type?
        proposed_criterion, # TODO: type?
        proposed_checkpoint, # TODO: type?
        proposed_evaluation_tester, # TODO: type?
        baseline_model: "torch.nn.Module",
        baseline_optimizer, # TODO: type?
        baseline_criterion, # TODO: type?
        baseline_checkpoint, # TODO: type?
        baseline_evaluation_tester, # TODO: type?
        data_loader: "DataLoader",
        device: "torch.device",
        save_model: "bool",
    ) -> None:
        self.num_epochs = num_epochs
        self.checkpoint_rate = checkpoint_rate

        self.baseline_model = baseline_model
        self.baseline_optimizer = baseline_optimizer
        self.baseline_criterion = baseline_criterion
        self.baseline_checkpoint = baseline_checkpoint
        self.baseline_evaluation_tester = baseline_evaluation_tester

        self.proposed_model = proposed_model
        self.proposed_optimizer = proposed_optimizer
        self.proposed_criterion = proposed_criterion
        self.proposed_checkpoint = proposed_checkpoint
        self.proposed_evaluation_tester = proposed_evaluation_tester
        
        self.data_loader = data_loader
        self.device = device
        self.save_model = save_model

        self.plotter = Plotter()

    def train(self):
        baseline_esd = EarlyStopDetector(self.baseline_model)
        proposed_esd = EarlyStopDetector(self.proposed_model)
        baseline_loss, proposed_lost = None, None
        baseline_stop = False
        proposed_stop = False

        for epoch in range(self.num_epochs):
            if baseline_stop and proposed_stop:
                print("Finished since all the models has stopped.")
                break

            print(f"[Epoch {epoch} / {self.num_epochs}]")

            self.baseline_model.train()
            self.proposed_model.train()

            for data1 in tqdm(self.data_loader, desc="Training..."):
                src, features, md = data1
                src, features, md = to_device(self.device, src, features, md)
                md_for_criterion = md[1:].reshape(-1)
                if not baseline_stop:
                    baseline_loss = self.train_one_batch(
                        self.baseline_model,
                        self.baseline_optimizer,
                        self.baseline_criterion,
                        md_for_criterion,
                        src,
                        md,
                        self.device,
                    )
                    self.plotter.add_to_plot(float(baseline_loss), "loss", "baseline")
                if not proposed_stop:
                    proposed_loss = self.train_one_batch(
                        self.proposed_model,
                        self.proposed_optimizer,
                        self.proposed_criterion,
                        md_for_criterion,
                        src,
                        features,
                        md,
                        self.device,
                    )
                    self.plotter.add_to_plot(float(proposed_loss), "loss", "proposed")

            if self.save_model and not ((epoch + 1) % self.checkpoint_rate):
                if not baseline_stop:
                    self.baseline_checkpoint.save_checkpoint("baseline.ptr")
                if not proposed_stop:
                    self.proposed_checkpoint.save_checkpoint("proposed.ptr")
            if not ((epoch + 1) % self.checkpoint_rate):
                if not baseline_stop:
                    baseline_metrics, _, _ = self.baseline_evaluation_tester.start_testing()
                    #                 print(baseline_metrics)
                    baseline_stop = baseline_esd.should_stop(
                        baseline_metrics[CodeMetric.BLEU]
                    )
                    for k, v in baseline_metrics[CodeMetric.BLEU].items():
                        self.plotter.add_to_plot(float(v), "bleu_on_eval", k, "baseline")
                    if baseline_stop:
                        print("Baseline Early Stopped!")
                if not proposed_stop:
                    proposed_metrics, _, _ = self.proposed_evaluation_tester.start_testing()
                    proposed_stop = proposed_esd.should_stop(
                        proposed_metrics[CodeMetric.BLEU]
                    )
                    for k, v in proposed_metrics[CodeMetric.BLEU].items():
                        self.plotter.add_to_plot(float(v), "bleu_on_eval", k, "proposed")
                    if proposed_stop:
                        print("Proposed Model Early Stopped!")

        return self.plotter.get_plot()

    def train_one_batch(self, model, optimizer, criterion, md_for_criterion, *inputs):
        output = model(
            *inputs,
        )
        output = output[1:].reshape(-1, output.shape[2])

        optimizer.zero_grad()
        loss = criterion(output, md_for_criterion)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        return loss


class Plotter:
    def __init__(self) -> None:
        self.to_plot = {
            'loss':{
                'baseline': [],
                'proposed': [],
            },
            'bleu_on_eval': {
                'bleu_1': {
                    'baseline': [],
                    'proposed': [],
                },
                'bleu_2': {
                    'baseline': [],
                    'proposed': [],
                },
                'bleu_3': {
                    'baseline': [],
                    'proposed': [],
                },
                'bleu_4': {
                    'baseline': [],
                    'proposed': [],
                },
            }
        }

    def add_to_plot(self, item, *keys):
        val = None
        for key in keys:
            if val is None:
                val = self.to_plot[key]
            else:
                val = val[key]
        val.append(item)

    def get_plot(self):
        return self.to_plot

