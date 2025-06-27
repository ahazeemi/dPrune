from abc import ABC, abstractmethod
from datasets import Dataset


class Scorer(ABC):
    """
    Abstract base class for all scoring methods.
    A Scorer assigns a numerical score to each example in a dataset.
    """

    @abstractmethod
    def score(self, dataset: Dataset, **kwargs) -> Dataset:
        """
        Takes a Hugging Face Dataset and returns a new Dataset with a 'score'
        column appended.

        Args:
            dataset (Dataset): The dataset to score.
            **kwargs: Additional scorer-specific arguments.

        Returns:
            Dataset: The dataset with an added 'score' column.
        """
        pass


class Pruner(ABC):
    """
    Abstract base class for all pruning methods.
    A Pruner selects a subset of a scored dataset based on a specific strategy.
    """

    @abstractmethod
    def prune(self, scored_dataset: Dataset, **kwargs) -> Dataset:
        """
        Takes a scored dataset and returns a pruned subset of it.

        Args:
            scored_dataset (Dataset): A dataset that includes a 'score' column.
            **kwargs: Additional pruner-specific arguments.

        Returns:
            Dataset: The pruned dataset.
        """
        pass
