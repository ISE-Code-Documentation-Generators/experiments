import abc
import random
import typing
from typing import List, Dict, Callable
from sympy import im
from torch.nn import functional as F

from ise_cdg_models.required_interfaces import MetricInterface
from ise_cdg_utility.metrics import get_vectorized_metrics
from ise_cdg_data.dataset import Md4DefDatasetInterface

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


if typing.TYPE_CHECKING:
    from ise_cdg_models.cnn2rnn import CNN2RNN, CNN2RNNAttention, CNN2RNNFeatures




class CNN2RNNTesterOnDatasetBase:

    def __init__(
            self, 
            name: str,
            model: torch.Module,
            dataset: Md4DefDatasetInterface,
            md_vocab,
            source_max_length,
            sos_ind: int, eos_ind: int,
            device: torch.device,
            example_ratio : typing.Union[float, None] = None,
            printer = print,
    ) -> None:
        self.name = name
        self.model = model
        self.dataset = dataset
        self.md_vocab = md_vocab
        self.printer = printer
        self.source_max_length = source_max_length
        self.metrics_with_name = get_vectorized_metrics(self.md_vocab)

        self.sos_ind = sos_ind
        self.eos_ind = eos_ind
        self.example_ratio = example_ratio
        self.device = device

    def __pad_to_length(self, raw_input):
        # raw_input.shape: (..., raw_input_length)
        raw_input_length = raw_input.shape[0]
        pad_length = self.source_max_length - raw_input_length
        # shape: (..., expected_length)
        return F.pad(raw_input, (0, 0, 0, pad_length), value=0)

    
    def should_print_example(self) -> bool:
        return self.example_ratio is not None and random.random() < self.example_ratio

    @abc.abstractmethod
    def generate_one_markdown(self, src, features):
        pass
    
    def start_testing(self):
        self.printer(f'x---------- {self.name} Started Testing ----------x')
        self.model.eval()
        examples_shown = 0
        with torch.no_grad():
            candidates = []
            mds = []
            for i in tqdm(range(len(self.dataset))):
                src, features, md = self.dataset[i]
                src = self.__pad_to_length(src.unsqueeze(1).to(self.device))
                features = features.unsqueeze(0).to(self.device)
                output = self.generate_one_markdown(src, features)
                candidate = [int(ind) for ind in output.tolist()]
                target = [int(ind) for ind in md.tolist()]
                candidates.append(candidate)
                mds.append([target])
                if self.should_print_example():
                    examples_shown += 1
                    self.printer(f'x- example {examples_shown} -x')
                    self.printer('\tReal: ' + ' '.join([self.md_vocab.get_itos()[tok_ind] for tok_ind in target]))
                    self.printer('\tPredicted: ' + ' '.join([self.md_vocab.get_itos()[tok_ind] for tok_ind in candidate]))
        
        metric_results = {}
        for metric_name, metric in self.metrics_with_name.items():
            self.printer(f'x--- {metric_name} ---x')
            metric.set_references(mds)
            result = metric(candidates)
            self.printer(str(result))
            metric_results[metric_name] = result
        self.printer('')
        return metric_results, candidates, mds

class CNN2RNNTesterOnDataset(CNN2RNNTesterOnDatasetBase):
    model: "CNN2RNN"

    def generate_one_markdown(self, src, features):
        return self.model.generate_one_markdown(
                src,
                self.sos_ind, self.eos_ind,
                sequence_max_length=25,
                device=self.device,
            )

class CNN2RNNFeaturesTesterOnDataset(CNN2RNNTesterOnDatasetBase):
    model: "CNN2RNNFeatures"

    def generate_one_markdown(self, src, features):
        return self.model.generate_one_markdown(
                src, features,
                self.sos_ind, self.eos_ind,
                sequence_max_length=25,
                device=self.device,
            )

    def start_testing(
            self,
            metrics_with_name: Dict[str, MetricInterface],
            dataset_id_generator: Callable,
            sos_ind: int, eos_ind: int,
            device: torch.device,
            example_ratio : typing.Union[float, None] = None,
    ):
        self.printer(f'x---------- {self.name} Started Testing ----------x')
        self.model.eval()
        examples_shown = 0
        with torch.no_grad():
            candidates = []
            mds = []
            for i in dataset_id_generator():
                src, features, md = self.dataset[i]
                src = self.__pad_to_length(src.unsqueeze(1).to(device))
                features = features.unsqueeze(0).to(device)
                if self.send_features:
                    model: 'CNN2RNNFeatures' = self.model
                    output = model.generate_one_markdown(
                        src, features,
                        sos_ind, eos_ind,
                        sequence_max_length=25,
                        device=device,
                    )
                else:
                    model: typing.Union['CNN2RNN', 'CNN2RNNAttention'] = self.model
                    output = model.generate_one_markdown(
                        src,
                        sos_ind, eos_ind,
                        sequence_max_length=25,
                        device=device,
                    )
                candidate = [int(ind) for ind in output.tolist()]
                target = [int(ind) for ind in md.tolist()]
                candidates.append(candidate)
                mds.append([target])
                if example_ratio is not None and random.random() < example_ratio:
                    examples_shown += 1
                    self.printer(f'x- example {examples_shown} -x')
                    self.printer('\tReal: ' + ' '.join([self.md_vocab.get_itos()[tok_ind] for tok_ind in target]))
                    self.printer('\tPredicted: ' + ' '.join([self.md_vocab.get_itos()[tok_ind] for tok_ind in candidate]))
        
        metric_results = {}
        for metric_name, metric in metrics_with_name.items():
            self.printer(f'x--- {metric_name} ---x')
            metric.set_references(mds)
            result = metric(candidates)
            self.printer(str(result))
            metric_results[metric_name] = result
        self.printer('')
        return metric_results, candidates, mds