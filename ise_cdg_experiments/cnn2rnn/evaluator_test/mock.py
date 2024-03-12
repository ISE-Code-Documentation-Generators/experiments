from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ise_cdg_experiments.cnn2rnn.test.mock import MockExperimentFacade

class SampleCNN2RNNFeaturesEvaluatorContainer:
    def __init__(self, mock_experiment_facade: "MockExperimentFacade") -> None:
        from ise_cdg_experiments.cnn2rnn.evaluator import CNN2RNNFeaturesEvaluator
        from ise_cdg_experiments.cnn2rnn.plotter import CNN2RNNPlotter

        test_plotter = CNN2RNNPlotter()
        test_visitors = [test_plotter]
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

        self.__test_plotter = test_plotter
        self.__proposed_tester = proposed_tester

    def __get_evaluator(self):
        return self.__proposed_tester

    def __get_plotter(self):
        return self.__test_plotter
    
    def get_metrics(self):
        self.__get_evaluator().valid_for_save()
        return self.__get_plotter().to_plot