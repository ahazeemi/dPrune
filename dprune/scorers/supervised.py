import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


def _get_device(model: PreTrainedModel) -> torch.device:
    try:
        first_param = next(model.parameters())
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if first_param.is_cuda:
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        device = _get_device(self.model)
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


class GraNdScorer(Scorer):
    """Implements the GraNd score from "Deep Learning on a Data Diet" (Paul et al., 2021).

    The GraNd score measures the gradient norm of the loss with respect to the
    model parameters for each training example. Examples with a higher gradient
    norm are deemed more important as they are expected to contribute more to
    the optimisation trajectory. This scorer computes the per-example gradient
    norm using the current model weights.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        text_column: str,
        label_column: str,
        batch_size: int = 4,
        max_length: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.batch_size = batch_size
        self.max_length = max_length

    def score(self, dataset: Dataset, **kwargs) -> Dataset:
        self.model.eval()
        device = _get_device(self.model)
        self.model.to(device)

        def collate_fn(batch):
            texts = [item[self.text_column] for item in batch]
            labels = [item[self.label_column] for item in batch]
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs["labels"] = torch.tensor(labels, dtype=torch.long)
            return inputs

        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=collate_fn
        )

        scores = []
        for batch in tqdm(data_loader, desc="Scoring with GraNd"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            outputs = self.model(**batch)
            logits = outputs.logits
            loss_per_example = torch.nn.functional.cross_entropy(
                logits, labels, reduction="none"
            )

            for loss in loss_per_example:
                grads = torch.autograd.grad(
                    loss,
                    self.model.parameters(),
                    retain_graph=True,
                    allow_unused=True,
                )
                grad_norm_sq = 0.0
                for g in grads:
                    if g is None:
                        continue
                    grad_norm_sq += g.pow(2).sum().item()
                scores.append(grad_norm_sq**0.5)

            self.model.zero_grad(set_to_none=True)

        return dataset.add_column("score", scores)


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
