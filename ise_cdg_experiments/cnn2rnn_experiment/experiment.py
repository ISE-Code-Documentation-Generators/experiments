

from ise_cdg_experiments.experiment import BaseExperiment


class CNN2RNNExperiment(BaseExperiment):
    def should_save(self, epoch: int) -> bool:
        return not ((epoch + 1) % 10)