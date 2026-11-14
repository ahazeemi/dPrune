import math
from typing import Union
import numpy as np
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


class ThresholdPruner(Pruner):
    """
    Pruner that keeps examples whose score is below (or above) a threshold.
    Useful for score-based filtering such as SemDeDup's epsilon deduplication
    (`ThresholdPruner(threshold=1 - eps, keep="below")`).
    """

    def __init__(self, threshold: float, keep: str = "below", inclusive: bool = False):
        if keep not in ("below", "above"):
            raise ValueError("keep must be 'below' or 'above'.")
        self.threshold = threshold
        self.keep = keep
        self.inclusive = inclusive

    def prune(self, scored_dataset: Dataset, **kwargs) -> Dataset:
        scores = scored_dataset["score"]
        if self.keep == "below":
            keep = [i for i, s in enumerate(scores) if (s <= self.threshold if self.inclusive else s < self.threshold)]
        else:
            keep = [i for i, s in enumerate(scores) if (s >= self.threshold if self.inclusive else s > self.threshold)]
        return scored_dataset.select(keep)


class CoveragePruner(Pruner):
    """
    Coverage-centric Coreset Selection (CCS) from Zheng et al., 2023, "Coverage-centric
    Coreset Selection for High Pruning Rates" (https://arxiv.org/abs/2210.15809).

    Assumes a higher score means a harder example (e.g. `EL2NScorer`,
    `CrossEntropyScorer`, or `DataMapsScorer` with an inverted metric). The hardest
    `hard_cutoff` fraction of examples is dropped first (they tend to be noisy or
    mislabeled), then the remaining score range is split into `num_strata` equal-width
    bins and the budget is spread as evenly as possible across bins, sampling randomly
    inside each. Sparse bins are kept whole and unused budget is redistributed to
    denser bins, so the selection covers easy, medium and hard examples. This is
    especially helpful at high pruning rates, where top-k selection of hard examples fails.
    """

    def __init__(
        self,
        k: Union[float, int],
        num_strata: int = 50,
        hard_cutoff: float = 0.1,
        seed: int = 42,
    ):
        if not (isinstance(num_strata, int) and num_strata > 0):
            raise ValueError("num_strata must be a positive integer.")
        if not (0 <= hard_cutoff < 1):
            raise ValueError("hard_cutoff must be in [0, 1).")
        self.k = k
        self.num_strata = num_strata
        self.hard_cutoff = hard_cutoff
        self.seed = seed

    def prune(self, scored_dataset: Dataset, **kwargs) -> Dataset:
        if isinstance(self.k, float) and (self.k < 0 or self.k > 1):
            raise ValueError(f"k must be a float between 0 and 1, but got {self.k}")

        n = len(scored_dataset)
        k_examples = math.ceil(n * self.k) if isinstance(self.k, float) else self.k
        if k_examples >= n:
            return scored_dataset

        scores = np.asarray(scored_dataset["score"], dtype=np.float64)
        rng = np.random.RandomState(self.seed)

        # Drop the hardest fraction, but never so many that the budget cannot be met.
        n_drop = min(int(math.floor(n * self.hard_cutoff)), n - k_examples)
        order = np.argsort(scores, kind="stable")
        candidates = order[: n - n_drop]

        cand_scores = scores[candidates]
        lo, hi = cand_scores.min(), cand_scores.max()
        if hi == lo:
            bins = np.zeros(len(candidates), dtype=int)
        else:
            bins = np.minimum(
                ((cand_scores - lo) / (hi - lo) * self.num_strata).astype(int),
                self.num_strata - 1,
            )
        strata = [candidates[bins == b] for b in range(self.num_strata)]
        strata = [s for s in strata if len(s) > 0]
        strata.sort(key=len)

        selected = []
        remaining = k_examples
        for i, stratum in enumerate(strata):
            quota = min(len(stratum), math.ceil(remaining / (len(strata) - i)))
            selected.extend(rng.choice(stratum, size=quota, replace=False).tolist())
            remaining -= quota

        return scored_dataset.select(sorted(selected))


class ImportanceResamplingPruner(Pruner):
    """
    Samples `k` examples without replacement with probability proportional to
    `exp(score / temperature)`, using the Gumbel top-k trick: add i.i.d. Gumbel
    noise to the scores and keep the top-k. This is the resampling step of DSIR
    (Xie et al., 2023, https://arxiv.org/abs/2302.03169) and expects scores that
    are log importance weights (e.g. from `DSIRScorer`).

    Compared to `TopKPruner`, sampling keeps some diversity instead of
    concentrating on the few most target-like examples. As `temperature -> 0`
    it reduces to `TopKPruner`; as it grows it approaches `RandomPruner`.
    """

    def __init__(self, k: Union[float, int], temperature: float = 1.0, seed: int = 42):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.k = k
        self.temperature = temperature
        self.seed = seed

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

        scores = np.asarray(scored_dataset["score"], dtype=np.float64) / self.temperature
        rng = np.random.default_rng(self.seed)
        perturbed = scores + rng.gumbel(size=num_examples)
        indices = np.argsort(-perturbed, kind="stable")[:k_examples]
        return scored_dataset.select(sorted(indices.tolist()))
