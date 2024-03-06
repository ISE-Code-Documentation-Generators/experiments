import abc
from ise_cdg_data.dataset import Facade, Md4DefDatasetInterface
from ise_cdg_models import CNN2RNNFeatures

import torch
import pandas as pd

class _MockDatasetContainer:
    def __init__(self) -> None:
        source_max_length = 320
        compute_features = True
        dataset_path = "final_dataset.csv"
        use_header = not compute_features
        mode = Facade.DatasetMode.WITH_PREPROCESS

        cache_dataset_path = "cache_final_dataset.gzip"
        cache_filter_length = 20

        cache_df = pd.read_parquet(cache_dataset_path)
        cache_df = cache_df[:cache_filter_length]
        self.dataset = Facade(mode).get_cnn2rnn_features(
            dataset_path,
            source_max_length,
            compute_features=compute_features,
            use_header=use_header,
            cache_df=cache_df,
        )

    # TODO: use cached-property
    def get_dataset(self) -> "Md4DefDatasetInterface":
        return self.dataset

    def get_source_max_length(self) -> int:
        # I get the result from "dataset" on purpose, despite being able to have it from "self.source_max_length"
        # This reflects the idea that objects can hold the parameters
        return self.dataset.src_max_length


class CNN2RNNFeaturesInterface(metaclass=abc.ABCMeta):
    def generate_one_markdown(
        self,
        source: torch.Tensor,
        features: torch.Tensor,
        sos_ind: int,
        eos_ind: int,
        sequence_max_length: int,
        device: torch.device,
    ):
        pass

    def forward(self, source, features, markdown, device, teacher_force_ratio=0.9):
        pass


class _MockCNN2RNNFeatures(torch.nn.Module, CNN2RNNFeaturesInterface):
    def __init__(self, md_vocab):
        super().__init__()
        from ise_cdg_models.cnn2rnn.embeddings import VocabEmbeddingHelper

        md_vocab_helper = VocabEmbeddingHelper(md_vocab)
        self.md_vocab_size = md_vocab_helper.vocab_size

    def forward(self, source, features, markdown, device, teacher_force_ratio=0.9):
        batch_size = source.shape[1]
        target_sequence_len = markdown.shape[0]
        target_vocab_size = self.md_vocab_size
        return torch.ones(target_sequence_len, batch_size, target_vocab_size).to(device)

    def generate_one_markdown(
        self,
        source: torch.Tensor,
        features: torch.Tensor,
        sos_ind: int,
        eos_ind: int,
        sequence_max_length: int,
        device: torch.device,
    ):
        return torch.ones(
            sequence_max_length,
        ).to(device)


class MockExperimentFacade:
    def __init__(self) -> None:
        from tqdm import tqdm
        tqdm.pandas()

        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__dataset_container = _MockDatasetContainer()

        self.__proposed_model = _MockCNN2RNNFeatures(
            md_vocab=self.get_md_vocab(),
        ).to(self.get_device())

    def get_device(self):
        return self.__device

    def get_dataset(self):
        return self.__dataset_container.get_dataset()

    def get_md_vocab(self):
        return self.get_dataset().md_vocab
    
    def get_source_max_length(self):
        return self.__dataset_container.get_source_max_length()

    def get_proposed_model(self):
        return self.__proposed_model