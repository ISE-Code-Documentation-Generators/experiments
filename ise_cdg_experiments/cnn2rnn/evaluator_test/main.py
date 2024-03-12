from functools import cached_property
from typing import TYPE_CHECKING, Dict
import unittest


if TYPE_CHECKING:
    from ise_cdg_experiments.cnn2rnn.evaluator_test.mock import (
        SampleCNN2RNNFeaturesEvaluatorContainer,
    )
    from ise_cdg_experiments.cnn2rnn.test.custom_io import Custom_IO


class EvaluatorTester(unittest.TestCase):
    @cached_property
    def __evaluator_container(self) -> "SampleCNN2RNNFeaturesEvaluatorContainer":
        from ise_cdg_experiments.cnn2rnn.test.mock import MockExperimentFacade
        from ise_cdg_experiments.cnn2rnn.evaluator_test.mock import (
            SampleCNN2RNNFeaturesEvaluatorContainer,
        )

        return SampleCNN2RNNFeaturesEvaluatorContainer(
            mock_experiment_facade=MockExperimentFacade()
        )

    @cached_property
    def __io(self) -> "Custom_IO[Dict]":
        from ise_cdg_experiments.cnn2rnn.test.custom_io import JSON_IO

        return JSON_IO(results_dir="ise_cdg_experiments/cnn2rnn/evaluator_test")

    def test_metrics(self):
        metrics = self.__evaluator_container.get_metrics()
        # self.__io.write(metrics, 'metrics.json')
        expected_metrics = self.__io.read("metrics.json")
        self.assertEqual(metrics, expected_metrics)

    # TODO: This is the logic to also assert the expected (candidate, md) results of the evaluation.
    #       # However, it doesn't work due to computational complexities issues.
    def test_numbers(self):
        # _, candidates, mds = self.__proposed_tester.evaluate()
        # numbers = [candidates, mds]
        # self.__io.write(numbers, 'numbers.json')
        # expected_numbers = self.__io.read(f'numbers.json')
        # self.assertEqual(numbers, expected_numbers)
        pass


if __name__ == "__main__":
    unittest.main()
