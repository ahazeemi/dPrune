import re
import zlib

import numpy as np
import torch
from datasets import Dataset
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Optional, Callable, Union

from ..base import Scorer

# Imports for PerplexityScorer
try:
    import kenlm
    KENLM_AVAILABLE = True
except ImportError:
    KENLM_AVAILABLE = False


def _get_embeddings(
    dataset: Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text_column: str,
    batch_size: int,
) -> np.ndarray:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def collate_fn(batch):
        texts = [item[text_column] for item in batch]
        return tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    embeddings = []
    for batch in tqdm(data_loader, desc="Extracting embeddings"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
            # Use the CLS token embedding from the last hidden state
            last_hidden_state = outputs.hidden_states[-1]
            cls_embeddings = last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)

    return np.concatenate(embeddings)


class KMeansCentroidDistanceScorer(Scorer):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        text_column: str,
        num_clusters: int,
        batch_size: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.num_clusters = num_clusters
        self.batch_size = batch_size

    def score(self, dataset: Dataset, **kwargs) -> Dataset:
        
        embeddings = _get_embeddings(
            dataset, self.model, self.tokenizer, self.text_column, self.batch_size
        )

        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        distances = np.linalg.norm(embeddings - centroids[cluster_labels], axis=1)

        return dataset.add_column("score", distances.tolist())


def _semdedup_scores(
    embeddings: np.ndarray, cluster_labels: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """
    For each example, the maximum cosine similarity to any other example in its
    cluster that has higher priority. Within a cluster, examples are ordered by
    decreasing distance to the centroid (most "unusual" first, which SemDeDup
    keeps), so only later examples can be flagged as duplicates of earlier
    ones. The highest-priority example of each cluster gets a score of -1.
    """
    norm_emb = embeddings / np.clip(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None
    )
    scores = np.full(len(embeddings), -1.0)
    for c in np.unique(cluster_labels):
        idx = np.where(cluster_labels == c)[0]
        if len(idx) < 2:
            continue
        dist = np.linalg.norm(embeddings[idx] - centroids[c], axis=1)
        idx = idx[np.argsort(-dist, kind="stable")]
        sim = norm_emb[idx] @ norm_emb[idx].T
        # Keep only similarities to higher-priority (earlier) examples.
        sim = np.tril(sim, k=-1) + np.triu(np.full_like(sim, -1.0))
        scores[idx] = sim.max(axis=1)
    return scores


class SemDeDupScorer(Scorer):
    """
    Semantic deduplication scorer from Abbas et al., 2023, "SemDeDup: Data-efficient
    learning at web-scale through semantic deduplication"
    (https://arxiv.org/abs/2303.09540).

    Embeddings are clustered with k-means; inside each cluster, every example is
    scored by its maximum cosine similarity to a higher-priority example (examples
    farthest from the centroid have priority). A score close to 1 means the example
    is a semantic near-duplicate of another one. Use `BottomKPruner` to keep the
    least redundant examples, or `ThresholdPruner(threshold=1 - eps, keep="below")`
    to reproduce the paper's epsilon-threshold deduplication.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        text_column: Optional[str] = None,
        num_clusters: int = 10,
        batch_size: int = 8,
        embeddings: Optional[np.ndarray] = None,
    ):
        """
        Args:
            model, tokenizer, text_column: Used to embed the text (CLS embedding).
                Not needed if `embeddings` is given.
            num_clusters (int): Number of k-means clusters.
            batch_size (int): Batch size for embedding extraction.
            embeddings (Optional[np.ndarray]): Precomputed embeddings of shape
                (len(dataset), dim), e.g. from a sentence-embedding model.
        """
        if embeddings is None and (model is None or tokenizer is None or text_column is None):
            raise ValueError(
                "Provide either precomputed `embeddings` or `model`, `tokenizer` and `text_column`."
            )
        if not isinstance(num_clusters, int) or num_clusters <= 0:
            raise ValueError("num_clusters must be a positive integer")
        self.model = model
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        self.embeddings = None if embeddings is None else np.asarray(embeddings, dtype=np.float64)

    def score(self, dataset: Dataset, **kwargs) -> Dataset:
        if self.embeddings is not None:
            embeddings = self.embeddings
            if len(embeddings) != len(dataset):
                raise ValueError(
                    f"Number of embeddings ({len(embeddings)}) does not match "
                    f"dataset size ({len(dataset)})."
                )
        else:
            embeddings = _get_embeddings(
                dataset, self.model, self.tokenizer, self.text_column, self.batch_size
            ).astype(np.float64)

        num_clusters = min(self.num_clusters, len(dataset))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        scores = _semdedup_scores(embeddings, cluster_labels, kmeans.cluster_centers_)

        return dataset.add_column("score", scores.tolist())


class PerplexityScorer(Scorer):
    """
    A Scorer that calculates perplexity scores using a KenLM language model.
    Lower perplexity indicates easier and more prototypical instances. 
    Higher perplexity indicates harder (and potentially more informative) instances.
    """

    def __init__(
        self,
        model_path: str,
        text_column: str,
        text_normalizer: Optional[Callable[[str], str]] = None,
        batch_size: int = 100,
    ):
        """
        Initializes the PerplexityScorer.

        Args:
            model_path (str): Path to the KenLM language model file.
            text_column (str): Name of the column containing text to score.
            text_normalizer (Optional[Callable[[str], str]]): Optional function to normalize text.
                If None, uses basic whitespace normalization.
            batch_size (int): Number of examples to process at once for progress tracking.

        Raises:
            ImportError: If kenlm is not installed.
            ValueError: If model_path is empty or text_column is empty.
            FileNotFoundError: If the model file doesn't exist.
        """
        if not KENLM_AVAILABLE:
            raise ImportError(
                "kenlm is required for PerplexityScorer. Install it with: pip install kenlm"
            )

        if not model_path:
            raise ValueError("model_path cannot be empty")
        
        if not text_column:
            raise ValueError("text_column cannot be empty")

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        try:
            import kenlm
            self.model = kenlm.Model(model_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not load KenLM model from {model_path}: {e}")

        self.text_column = text_column
        self.text_normalizer = text_normalizer or self._default_normalizer
        self.batch_size = batch_size

    def _default_normalizer(self, text: str) -> str:
        """
        Default text normalization: strip whitespace and normalize spaces.
        
        Args:
            text (str): Input text to normalize.
            
        Returns:
            str: Normalized text.
        """
        if not isinstance(text, str):
            text = str(text)
        return ' '.join(text.strip().split())

    def _calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity for a single text.
        
        Args:
            text (str): Input text.
            
        Returns:
            float: Perplexity score.
        """
        normalized_text = self.text_normalizer(text)
        
        if not normalized_text:
            return float('inf')  # Return high perplexity for empty text
        
        score = self.model.score(normalized_text)
        word_count = len(normalized_text.split())
        
        if word_count == 0:
            return float('inf')
        
        # Calculate perplexity: 10^(-score / word_count)
        perplexity = 10 ** (-score / word_count)
        return perplexity

    def score(self, dataset: Dataset, **kwargs) -> Dataset:
        """
        Calculate perplexity scores for all texts in the dataset.
        
        Args:
            dataset (Dataset): The dataset to score.
            **kwargs: Additional arguments (unused).
            
        Returns:
            Dataset: The dataset with an added 'score' column containing perplexity scores.
            
        Raises:
            ValueError: If the specified text_column doesn't exist in the dataset.
        """
        if self.text_column not in dataset.column_names:
            raise ValueError(
                f"Column '{self.text_column}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        scores = []
        
        for i in tqdm(range(0, len(dataset), self.batch_size), desc="Calculating perplexity"):
            batch_end = min(i + self.batch_size, len(dataset))
            batch = dataset[i:batch_end]
            
            if isinstance(batch[self.text_column], str):
                text = batch[self.text_column]
                score = self._calculate_perplexity(text)
                scores.append(score)
            else:
                for text in batch[self.text_column]:
                    score = self._calculate_perplexity(text)
                    scores.append(score)

        return dataset.add_column("score", scores)


_DSIR_TOKEN_RE = re.compile(r"\w+|[^\w\s]+")


def _dsir_tokenize(text: str):
    """Lowercased word/punctuation tokenization (like NLTK's wordpunct_tokenize)."""
    return _DSIR_TOKEN_RE.findall(str(text).lower())


def _dsir_hashed_ngrams(text: str, num_buckets: int, ngram_order: int):
    """Sparse counts of hashed 1..ngram_order-grams as (bucket_ids, counts).
    Uses a stable hash (crc32), so features are reproducible across processes
    (unlike Python's built-in `hash`)."""
    tokens = _dsir_tokenize(text)
    buckets = [
        zlib.crc32(" ".join(tokens[i:i + n]).encode("utf-8")) % num_buckets
        for n in range(1, ngram_order + 1)
        for i in range(len(tokens) - n + 1)
    ]
    return np.unique(np.asarray(buckets, dtype=np.int64), return_counts=True)


class DSIRScorer(Scorer):
    """
    Data Selection with Importance Resampling (DSIR) from Xie et al., 2023,
    "Data Selection for Language Models via Importance Resampling" (NeurIPS 2023,
    https://arxiv.org/abs/2302.03169).

    Selects raw data that looks like a (small) *target* dataset. Every text is
    mapped to hashed unigram + bigram counts; two bag-of-ngrams models are fit,
    one on the target texts and one on the raw dataset being scored. Each raw
    example is scored by its log importance weight

        score(x) = sum_j count_j(x) * (log p_target(j) - log p_raw(j)),

    so higher means "more like the target distribution". No model, GPU or labels
    are needed, which makes it practical for very large corpora.

    Use `ImportanceResamplingPruner` to reproduce DSIR's Gumbel top-k resampling
    (sampling proportional to the importance weights, without replacement), or
    `TopKPruner` for a deterministic top-k selection.
    """

    def __init__(
        self,
        target_texts,
        text_column: str,
        raw_texts=None,
        num_buckets: int = 10000,
        ngram_order: int = 2,
        smoothing: float = 1.0,
        length_normalize: bool = False,
    ):
        """
        Args:
            target_texts: Iterable of strings representing the target distribution
                (e.g. a few thousand in-domain examples).
            text_column (str): Name of the text column in the dataset to score.
            raw_texts: Optional iterable of strings to fit the raw distribution on.
                Defaults to the texts of the dataset being scored.
            num_buckets (int): Number of hash buckets for n-gram features.
            ngram_order (int): Highest n-gram order (2 = unigrams and bigrams, as in
                the paper).
            smoothing (float): Add-`smoothing` (Laplace) pseudocount per bucket.
            length_normalize (bool): Divide each score by the example's number of
                n-gram features, which removes the bias towards long documents.
        """
        if not text_column:
            raise ValueError("text_column cannot be empty")
        if not isinstance(num_buckets, int) or num_buckets <= 0:
            raise ValueError("num_buckets must be a positive integer")
        if not isinstance(ngram_order, int) or ngram_order <= 0:
            raise ValueError("ngram_order must be a positive integer")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        target_texts = list(target_texts)
        if len(target_texts) == 0:
            raise ValueError("target_texts cannot be empty")

        self.text_column = text_column
        self.num_buckets = num_buckets
        self.ngram_order = ngram_order
        self.smoothing = smoothing
        self.length_normalize = length_normalize
        self.target_log_probs = self._fit(target_texts)
        self.raw_texts = None if raw_texts is None else list(raw_texts)

    def _accumulate(self, features) -> np.ndarray:
        counts = np.zeros(self.num_buckets, dtype=np.float64)
        for ids, cnt in features:
            np.add.at(counts, ids, cnt)
        return counts

    def _log_probs(self, counts: np.ndarray) -> np.ndarray:
        counts = counts + self.smoothing
        return np.log(counts / counts.sum())

    def _fit(self, texts) -> np.ndarray:
        return self._log_probs(self._accumulate(
            _dsir_hashed_ngrams(t, self.num_buckets, self.ngram_order) for t in texts
        ))

    def score(self, dataset: Dataset, **kwargs) -> Dataset:
        if self.text_column not in dataset.column_names:
            raise ValueError(
                f"Column '{self.text_column}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )
        features = [
            _dsir_hashed_ngrams(t, self.num_buckets, self.ngram_order)
            for t in tqdm(dataset[self.text_column], desc="Hashing n-grams")
        ]

        if self.raw_texts is None:
            raw_log_probs = self._log_probs(self._accumulate(features))
        else:
            raw_log_probs = self._fit(self.raw_texts)

        log_ratio = self.target_log_probs - raw_log_probs
        scores = np.array([float(cnt @ log_ratio[ids]) for ids, cnt in features])
        if self.length_normalize:
            lengths = np.array([cnt.sum() for _, cnt in features], dtype=np.float64)
            scores = scores / np.maximum(lengths, 1.0)

        return dataset.add_column("score", scores.tolist())
