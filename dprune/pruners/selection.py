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


class D2Pruner(Pruner):
    """
    D2 Pruning from Maharana, Yadav & Bansal, ICLR 2024, "D2 Pruning: Message Passing
    for Balancing Diversity and Difficulty in Data Pruning"
    (https://arxiv.org/abs/2310.07931).

    The dataset is a k-nearest-neighbour graph over example embeddings, with each
    node starting from its difficulty score (the `score` column, higher = harder,
    e.g. from `EL2NScorer`, `CrossEntropyScorer` or `ForgettingScorer`).

    1. **Forward message passing**: each node gathers difficulty from its neighbours,
       ``s_i <- s_i + sum_j exp(-forward_gamma * d_ij^2) * s_j``, so examples in
       dense, hard regions score highest.
    2. **Greedy selection with reverse message passing**: the node with the highest
       updated score is selected, then each of its neighbours is down-weighted by
       ``exp(-reverse_gamma * d_ij^2) * s_i``, so near-duplicates of selected
       examples become less likely to be picked.

    The result covers the embedding space while still favouring informative
    examples. Embeddings are read from `embedding_column` of the scored dataset,
    passed as ``prune(dataset, embeddings=...)``, or computed with a Hugging Face
    `model` / `tokenizer` (CLS embedding of `text_column`).

    Args:
        k: Number (int) or fraction (float in [0, 1]) of examples to keep.
        n_neighbors: Neighbours per node in the kNN graph.
        forward_gamma: Decay of forward messages with distance.
        reverse_gamma: Decay of reverse (down-weighting) messages with distance.
        normalize_embeddings: L2-normalise embeddings before building the graph.
        normalize_scores: Min-max scale difficulty scores to [0, 1] first.
        hard_cutoff: Fraction of hardest examples to drop before selection
            (optional, as in CCS; useful with noisy labels).
        embedding_column: Dataset column holding embedding vectors.
        model, tokenizer, text_column, batch_size: Used to compute embeddings
            when none are supplied.
    """

    def __init__(
        self,
        k: Union[float, int],
        n_neighbors: int = 10,
        forward_gamma: float = 1.0,
        reverse_gamma: float = 1.0,
        normalize_embeddings: bool = True,
        normalize_scores: bool = True,
        hard_cutoff: float = 0.0,
        embedding_column: str = "embedding",
        model=None,
        tokenizer=None,
        text_column: str = "text",
        batch_size: int = 8,
    ):
        if not (isinstance(n_neighbors, int) and n_neighbors > 0):
            raise ValueError("n_neighbors must be a positive integer.")
        if forward_gamma < 0 or reverse_gamma < 0:
            raise ValueError("forward_gamma and reverse_gamma must be non-negative.")
        if not (0 <= hard_cutoff < 1):
            raise ValueError("hard_cutoff must be in [0, 1).")
        self.k = k
        self.n_neighbors = n_neighbors
        self.forward_gamma = forward_gamma
        self.reverse_gamma = reverse_gamma
        self.normalize_embeddings = normalize_embeddings
        self.normalize_scores = normalize_scores
        self.hard_cutoff = hard_cutoff
        self.embedding_column = embedding_column
        self.model = model
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.batch_size = batch_size

    def _get_embeddings(self, scored_dataset: Dataset, embeddings) -> np.ndarray:
        if embeddings is not None:
            emb = np.asarray(embeddings, dtype=np.float64)
        elif self.embedding_column in scored_dataset.column_names:
            emb = np.asarray(scored_dataset[self.embedding_column], dtype=np.float64)
        elif self.model is not None and self.tokenizer is not None:
            from ..scorers.unsupervised import _get_embeddings

            emb = _get_embeddings(
                scored_dataset, self.model, self.tokenizer, self.text_column, self.batch_size
            ).astype(np.float64)
        else:
            raise ValueError(
                "D2Pruner needs embeddings: pass `embeddings=` to prune(), add an "
                f"'{self.embedding_column}' column, or provide `model` and `tokenizer`."
            )
        if emb.ndim != 2 or emb.shape[0] != len(scored_dataset):
            raise ValueError(
                f"embeddings must have shape (num_examples, dim); got {emb.shape} "
                f"for {len(scored_dataset)} examples."
            )
        return emb

    def prune(self, scored_dataset: Dataset, embeddings=None, **kwargs) -> Dataset:
        import heapq
        from sklearn.neighbors import NearestNeighbors

        if isinstance(self.k, float) and (self.k < 0 or self.k > 1):
            raise ValueError(f"k must be a float between 0 and 1, but got {self.k}")

        n = len(scored_dataset)
        k_examples = math.ceil(n * self.k) if isinstance(self.k, float) else self.k
        if k_examples >= n:
            return scored_dataset
        if k_examples <= 0:
            return scored_dataset.select([])

        emb = self._get_embeddings(scored_dataset, embeddings)
        scores = np.asarray(scored_dataset["score"], dtype=np.float64)

        # Optionally drop the hardest fraction (never below the budget).
        n_drop = min(int(math.floor(n * self.hard_cutoff)), n - k_examples)
        candidates = np.argsort(scores, kind="stable")[: n - n_drop]
        emb, scores = emb[candidates], scores[candidates]
        m = len(candidates)

        if self.normalize_embeddings:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.where(norms == 0, 1.0, norms)
        if self.normalize_scores:
            lo, hi = scores.min(), scores.max()
            scores = (scores - lo) / (hi - lo) if hi > lo else np.ones_like(scores)

        # kNN graph (first neighbour returned is the node itself).
        n_nb = min(self.n_neighbors + 1, m)
        dists, nbrs = NearestNeighbors(n_neighbors=n_nb).fit(emb).kneighbors(emb)
        dists, nbrs = dists[:, 1:], nbrs[:, 1:]
        sq = dists ** 2

        # Forward message passing.
        node = scores + (np.exp(-self.forward_gamma * sq) * scores[nbrs]).sum(axis=1)
        reverse_w = np.exp(-self.reverse_gamma * sq)

        # Greedy selection with reverse message passing (max-heap with stale entries).
        heap = [(-v, i) for i, v in enumerate(node)]
        heapq.heapify(heap)
        selected = np.zeros(m, dtype=bool)
        picked = []
        while len(picked) < k_examples and heap:
            neg_v, i = heapq.heappop(heap)
            if selected[i] or -neg_v != node[i]:
                continue
            selected[i] = True
            picked.append(i)
            s_i = node[i]
            for j, w in zip(nbrs[i], reverse_w[i]):
                if not selected[j]:
                    node[j] -= w * s_i
                    heapq.heappush(heap, (-node[j], j))

        return scored_dataset.select(sorted(candidates[picked].tolist()))
