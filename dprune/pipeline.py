from datasets import Dataset
from .base import Scorer, Pruner


class PruningPipeline:
    """
    A pipeline that orchestrates the data pruning process by chaining a
    scorer and a pruner.
    """

    def __init__(self, scorer: Scorer, pruner: Pruner):
        """
        Initializes the PruningPipeline.

        Args:
            scorer (Scorer): An instance of a Scorer to assign scores to the data.
            pruner (Pruner): An instance of a Pruner to select a subset of the data.
        """
        if not isinstance(scorer, Scorer):
            raise TypeError("scorer must be an instance of a Scorer subclass")
        if not isinstance(pruner, Pruner):
            raise TypeError("pruner must be an instance of a Pruner subclass")

        self.scorer = scorer
        self.pruner = pruner

    def run(self, dataset: Dataset, **kwargs) -> Dataset:
        """
        Executes the full pruning pipeline: scoring followed by pruning.

        Args:
            dataset (Dataset): The initial dataset to be pruned.
            **kwargs: Arguments to be passed to the scorer's score method.

        Returns:
            Dataset: The final, pruned dataset.
        """
        scored_dataset = self.scorer.score(dataset, **kwargs)
        pruned_dataset = self.pruner.prune(scored_dataset)
        return pruned_dataset
