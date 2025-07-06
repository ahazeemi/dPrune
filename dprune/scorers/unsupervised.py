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
