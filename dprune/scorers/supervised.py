import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Sequence, Union

from ..base import Scorer
from ..callbacks import ForgettingCallback


class CrossEntropyScorer(Scorer):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        text_column: str,
        label_column: str,
        batch_size: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.batch_size = batch_size

    def score(self, dataset: Dataset, **kwargs) -> Dataset:
        
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        def collate_fn(batch):
            texts = [item[self.text_column] for item in batch]
            labels = [item[self.label_column] for item in batch]
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs["labels"] = torch.tensor(labels, dtype=torch.long)
            return inputs

        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=collate_fn
        )
        
        scores = []
        for batch in tqdm(data_loader, desc="Scoring"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = torch.nn.functional.cross_entropy(
                    outputs.logits, batch["labels"], reduction="none"
                )
                scores.extend(loss.cpu().numpy())

        return dataset.add_column("score", scores)


class EL2NScorer(Scorer):
    """
    Error L2-Norm (EL2N) scorer from Paul et al., 2021, "Deep Learning on a
    Data Diet: Finding Important Examples Early in Training"
    (https://arxiv.org/abs/2107.07075).

    The score of an example is the L2 norm of the difference between the
    model's predicted class probabilities and the one-hot encoded label.
    Higher scores indicate harder examples, so the paper keeps the highest
    scoring examples (e.g. with `TopKPruner`). Best results are obtained with a
    model trained for only a few epochs; scores can optionally be averaged
    over several models by passing a list of models as `model`.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, Sequence[PreTrainedModel]],
        tokenizer: PreTrainedTokenizer,
        text_column: str,
        label_column: str,
        batch_size: int = 8,
    ):
        self.models = list(model) if isinstance(model, (list, tuple)) else [model]
        if not self.models:
            raise ValueError("At least one model is required.")
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.batch_size = batch_size

    def score(self, dataset: Dataset, **kwargs) -> Dataset:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def collate_fn(batch):
            texts = [item[self.text_column] for item in batch]
            labels = [item[self.label_column] for item in batch]
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs["labels"] = torch.tensor(labels, dtype=torch.long)
            return inputs

        total = None
        for model in self.models:
            model.eval()
            model.to(device)
            data_loader = DataLoader(
                dataset, batch_size=self.batch_size, collate_fn=collate_fn
            )
            scores = []
            for batch in tqdm(data_loader, desc="Scoring (EL2N)"):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                with torch.no_grad():
                    logits = model(**batch).logits
                    probs = torch.softmax(logits, dim=-1)
                    one_hot = torch.nn.functional.one_hot(
                        labels, num_classes=probs.shape[-1]
                    ).to(probs.dtype)
                    scores.extend(torch.linalg.vector_norm(probs - one_hot, dim=-1).cpu().tolist())
            scores = np.asarray(scores, dtype=np.float64)
            total = scores if total is None else total + scores

        return dataset.add_column("score", (total / len(self.models)).tolist())


class ForgettingScorer(Scorer):
    """
    A Scorer that uses a ForgettingCallback to assign a "forgetting score"
    to each example. The score is the number of times an example was
    "forgotten" during training (i.e., transitioned from being classified
    correctly to incorrectly).
    """

    def __init__(self, forgetting_callback: ForgettingCallback):
        """
        Initializes the ForgettingScorer.

        Args:
            forgetting_callback (ForgettingCallback): A ForgettingCallback instance
                that has been used during a Trainer's training run.
        """
        self.callback = forgetting_callback

    def score(self, dataset: Dataset, **kwargs) -> Dataset:
        """
        Calculates and adds the forgetting scores to the dataset.
        The dataset passed here should be the same one used for training.
        """
        scores = self.callback.calculate_forgetting_scores()

        if len(scores) != len(dataset):
            raise ValueError(
                f"The number of scores from the callback ({len(scores)}) does not match "
                f"the dataset size ({len(dataset)}). Ensure the same dataset was used "
                "for training and scoring."
            )

        return dataset.add_column("score", scores)
