from ise_cdg_experiments.experiment import BaseExperiment


class CNN2RNNExperiment(BaseExperiment):

    def should_save(self, epoch: int) -> bool:
        return bool(not epoch % 2) or super().should_save(epoch)
