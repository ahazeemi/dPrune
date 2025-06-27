import numpy as np
import torch
from datasets import Dataset
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..base import Scorer


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
