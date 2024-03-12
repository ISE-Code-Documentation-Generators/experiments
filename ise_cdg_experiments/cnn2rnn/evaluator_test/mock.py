from functools import cached_property
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ise_cdg_experiments.cnn2rnn.test.mock import MockExperimentFacade
    from ise_cdg_experiments.cnn2rnn.plotter import CNN2RNNPlotter
    from ise_cdg_experiments.interfaces import ExperimentVisitorInterface

class SampleCNN2RNNFeaturesEvaluatorContainer:
    def __init__(self, mock_experiment_facade: "MockExperimentFacade") -> None:
        from ise_cdg_experiments.cnn2rnn.evaluator import CNN2RNNFeaturesEvaluator

        test_visitors: List["ExperimentVisitorInterface"] = [self.__plotter]
        test_example_ratio: float = 0.001
        proposed_tester = CNN2RNNFeaturesEvaluator(
            device=mock_experiment_facade.get_device(),
            dataset=mock_experiment_facade.get_dataset(),
            md_vocab=mock_experiment_facade.get_md_vocab(),
            sos_ind=mock_experiment_facade.get_md_vocab().get_stoi()["<sos>"],
            eos_ind=mock_experiment_facade.get_md_vocab().get_stoi()["<eos>"],
            source_max_length=mock_experiment_facade.get_source_max_length(),
            model=mock_experiment_facade.get_proposed_model(),
            visitors=test_visitors,
            example_ratio=test_example_ratio,
        )

        self.__proposed_tester = proposed_tester

    @cached_property
    def __plotter(self) -> "CNN2RNNPlotter":
        from ise_cdg_experiments.cnn2rnn.plotter import CNN2RNNPlotter
        return CNN2RNNPlotter()

    def __get_evaluator(self):
        return self.__proposed_tester
    
    def get_metrics(self):
        self.__get_evaluator().valid_for_save()
        return self.__plotter.to_plot