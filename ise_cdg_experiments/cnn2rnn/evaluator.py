import abc
import random
import typing
from typing import Any, List, Optional
from torch._tensor import Tensor
from torch.nn import functional as F

from ise_cdg_utility.metrics import get_vectorized_metrics
from ise_cdg_data.dataset import Md4DefDatasetInterface
from ise_cdg_utility.metrics import CodeMetric

import torch
from tqdm import tqdm
from ise_cdg_experiments.evaluator import BaseModelTrainEvaluator

from ise_cdg_experiments.interfaces import (
    ExperimentVisitorInterface,
)


if typing.TYPE_CHECKING:
    from ise_cdg_models.cnn2rnn import CNN2RNN, CNN2RNNFeatures


class EarlyStopDetector:
    def __init__(self, threshold=0.02, allowed_chances=1):
        self.threshold = threshold
        self.allowed_chances, self.chances = allowed_chances, 0
        self.last_bleu_metrics = None

    def majority(self, l):
        l.sort()
        return l[len(l) // 2]

    def should_stop(self, bleu_metrics: dict):
        if self.last_bleu_metrics is None:
            self.last_bleu_metrics = bleu_metrics
            return False
        else:
            should_stop_dict = {}
            for key, last_result in self.last_bleu_metrics.items():
                should_stop_dict[key] = last_result - bleu_metrics[key] > self.threshold
            should_stop = self.majority(list(should_stop_dict.values()))
            if should_stop:
                self.chances += 1
                return self.chances >= self.allowed_chances
            else:
                self.chances = 0
        return False


class CNN2RNNBaseEvaluator(BaseModelTrainEvaluator, abc.ABC):
    def __init__(
        self,
        model: Any,
        device: torch.device,
        dataset: "Md4DefDatasetInterface",
        md_vocab,
        source_max_length,
        sos_ind: "int",
        eos_ind: "int",
        example_ratio: typing.Union[float, None] = None,
        printer_name: str = "",
        printer=print,
        visitors: Optional[List[ExperimentVisitorInterface]] = None,
    ) -> None:
        super().__init__(model, device)
        self.dataset = dataset
        self.md_vocab = md_vocab
        self.printer_name = printer_name or self.__class__.__name__
        self.printer = printer
        self.source_max_length = source_max_length
        self.metrics_with_name = get_vectorized_metrics(self.md_vocab)

        self.sos_ind = sos_ind
        self.eos_ind = eos_ind
        self.example_ratio = example_ratio
        self.early_stop_detector = EarlyStopDetector()
        self._metrics = None
        self.visitors = visitors or []

    def __pad_to_length(self, raw_input):
        # raw_input.shape: (..., raw_input_length)
        raw_input_length = raw_input.shape[0]
        pad_length = self.source_max_length - raw_input_length
        # shape: (..., expected_length)
        return F.pad(raw_input, (0, 0, 0, pad_length), value=0)

    @abc.abstractmethod
    def generate_one_markdown(self, src, features) -> torch.Tensor:
        pass

    def should_print_example(self) -> bool:
        return self.example_ratio is not None and random.random() < self.example_ratio

    def evaluate(self):
        self.printer(f"x---------- {self.printer_name} Started Testing ----------x")
        self.model.eval()
        examples_shown = 0
        with torch.no_grad():
            candidates = []
            mds = []
            for i in tqdm(range(len(self.dataset))):
                src, features, md = self.dataset[i]
                src = self.__pad_to_length(src.unsqueeze(1).to(self.device))
                features = features.unsqueeze(0).to(self.device)
                features = torch.einsum(
                    "be->eb", features
                )  # shape: (features_length, batch)
                output = self.generate_one_markdown(src, features)
                candidate = [int(ind) for ind in output.tolist()]
                target = [int(ind) for ind in md.tolist()]
                candidates.append(candidate)
                mds.append([target])
                if self.should_print_example():
                    examples_shown += 1
                    self.printer(f"x- example {examples_shown} -x")
                    self.printer(
                        "\tReal: "
                        + " ".join(
                            [self.md_vocab.get_itos()[tok_ind] for tok_ind in target]
                        )
                    )
                    self.printer(
                        "\tPredicted: "
                        + " ".join(
                            [self.md_vocab.get_itos()[tok_ind] for tok_ind in candidate]
                        )
                    )

        metric_results = {}
        for metric_name, metric in self.metrics_with_name.items():
            self.printer(f"x--- {metric_name} ---x")
            metric.set_references(mds)
            result = metric(candidates)
            self.printer(str(result))
            metric_results[metric_name] = result
        self.printer("")
        return metric_results, candidates, mds

    def valid_for_save(self) -> bool:
        self._metrics, _, _ = self.evaluate()
        self.accept_visitors(self.visitors)
        return not self.early_stop_detector.should_stop(self._metrics[CodeMetric.BLEU])

    def accept_visitors(self, visitors: List[ExperimentVisitorInterface]):
        for v in visitors:
            v.visit_evaluator(self)


class CNN2RNNEvaluator(CNN2RNNBaseEvaluator):
    model: "CNN2RNN"

    def generate_one_markdown(self, src, features) -> Tensor:
        return self.model.generate_one_markdown(
            src,
            self.sos_ind,
            self.eos_ind,
            self.source_max_length,
            self.device,
        )


class CNN2RNNFeaturesEvaluator(CNN2RNNBaseEvaluator):
    model: "CNN2RNNFeatures"

    def generate_one_markdown(self, src, features) -> Tensor:
        return self.model.generate_one_markdown(
            src,
            features,
            self.sos_ind,
            self.eos_ind,
            self.source_max_length,
            self.device,
        )
