import math
from typing import Union
from datasets import Dataset
from ..base import Pruner
import pandas as pd


class TopKPruner(Pruner):
    """
    Pruner that selects the top-k percent or top-k number of examples
    from a scored dataset.
    """

    def __init__(self, k: Union[float, int]):
        self.k = k

    def prune(self, scored_dataset: Dataset, **kwargs) -> Dataset:
        
        if isinstance(self.k, float) and (self.k < 0 or self.k > 1):
            raise ValueError(f"k must be a float between 0 and 1, but got {self.k}")

        num_examples = len(scored_dataset)
        
        if isinstance(self.k, float):
            k_examples = math.ceil(num_examples * self.k)
        else:
            k_examples = self.k
        
        if k_examples >= num_examples:
            return scored_dataset

        sorted_dataset = scored_dataset.sort("score", reverse=True)
        return sorted_dataset.select(range(k_examples))


class BottomKPruner(Pruner):
    """
    Pruner that selects the bottom-k percent or bottom-k number of examples
    from a scored dataset.
    """

    def __init__(self, k: Union[float, int]):
        self.k = k

    def prune(self, scored_dataset: Dataset, **kwargs) -> Dataset:
        
        if isinstance(self.k, float) and (self.k < 0 or self.k > 1):
            raise ValueError(f"k must be a float between 0 and 1, but got {self.k}")

        num_examples = len(scored_dataset)
        
        if isinstance(self.k, float):
            k_examples = math.ceil(num_examples * self.k)
        else:
            k_examples = self.k

        if k_examples >= num_examples:
            return scored_dataset

        sorted_dataset = scored_dataset.sort("score", reverse=False)
        return sorted_dataset.select(range(k_examples))


class StratifiedPruner(Pruner):
    """
    Pruner that performs stratified sampling based on the scores.
    It divides the dataset into a specified number of strata based on score
    quantiles and then samples a proportional number of examples from each.
    """

    def __init__(self, k: Union[float, int], num_strata: int = 10):
        if not (isinstance(num_strata, int) and num_strata > 0):
            raise ValueError("num_strata must be a positive integer.")
        self.k = k
        self.num_strata = num_strata

    def prune(self, scored_dataset: Dataset, **kwargs) -> Dataset:
        if isinstance(self.k, float) and (self.k < 0 or self.k > 1):
            raise ValueError(f"k must be a float between 0 and 1, but got {self.k}")

        num_examples = len(scored_dataset)
        
        if isinstance(self.k, float):
            k_examples = math.ceil(num_examples * self.k)
        else:
            k_examples = self.k

        if k_examples >= num_examples:
            return scored_dataset

        # Use pandas for easy quantile-based binning
        df = scored_dataset.to_pandas()
        
        try:
            df['stratum'] = pd.qcut(df['score'], self.num_strata, labels=False, duplicates='drop')
        except ValueError:
            # Fallback for when scores are not unique enough for qcut
            df['stratum'] = pd.cut(df['score'], self.num_strata, labels=False)

        
        num_per_stratum = math.ceil(k_examples / self.num_strata)
        
        pruned_indices = df.groupby('stratum').apply(
            lambda x: x.sample(n=min(len(x), num_per_stratum), random_state=42)
        ).index.get_level_values(1) # Get original indices

        return scored_dataset.select(pruned_indices)


class RandomPruner(Pruner):
    """
    Pruner that randomly selects a k percent or k number of examples
    from a dataset. This pruner ignores the 'score' column and is useful for
    creating a baseline.
    """

    def __init__(self, k: Union[float, int]):
        self.k = k

    def prune(self, scored_dataset: Dataset, **kwargs) -> Dataset:
        if isinstance(self.k, float) and (self.k < 0 or self.k > 1):
            raise ValueError(f"k must be a float between 0 and 1, but got {self.k}")

        num_examples = len(scored_dataset)
        
        if isinstance(self.k, float):
            k_examples = math.ceil(num_examples * self.k)
        else:
            k_examples = self.k

        if k_examples >= num_examples:
            return scored_dataset

        # Use the dataset's built-in shuffle and select for efficiency
        return scored_dataset.shuffle(seed=42).select(range(k_examples))
