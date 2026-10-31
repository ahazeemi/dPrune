import numpy as np
from typing import List, Optional
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, Trainer


class ForgettingCallback(TrainerCallback):
    """
    A TrainerCallback to record learning events during training to later
    calculate forgetting scores.

    An example is "forgotten" if it is misclassified at an epoch after having
    been correctly classified in a previous epoch.

    Usage:
        callback = ForgettingCallback()
        trainer = Trainer(..., callbacks=[callback])
        callback.trainer = trainer
        trainer.train()
        scores = callback.calculate_forgetting_scores()
    """

    def __init__(self):
        self.learning_events = {}
        self.trainer: Optional[Trainer] = None

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        At the end of each epoch, get predictions for the entire training set
        and record whether each example was classified correctly.
        """
        if self.trainer is None or self.trainer.train_dataset is None:
            return

        # This can be computationally expensive on large datasets.
        predictions = self.trainer.predict(self.trainer.train_dataset)
  
        if predictions.predictions is None or predictions.label_ids is None:
            return

        predicted_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids

        for i, (pred, true) in enumerate(zip(predicted_labels, true_labels)):
            if i not in self.learning_events:
                self.learning_events[i] = []
            self.learning_events[i].append(1 if pred == true else 0)

    def calculate_forgetting_scores(self) -> List[int]:
        """
        Calculates the forgetting score for each example based on the
        recorded learning events.
        """
        if not self.learning_events:
            return []

        max_index = max(self.learning_events.keys())
        forgetting_scores = [0] * (max_index + 1)

        for i, events in self.learning_events.items():
            if len(events) < 2:
                continue

            # A transition from correct (1) to incorrect (0) is a forget event
            transitions = zip(events, events[1:])
            forget_count = sum(1 for prev, curr in transitions if prev == 1 and curr == 0)
            forgetting_scores[i] = forget_count

        return forgetting_scores 

class DataMapsCallback(TrainerCallback):
    """
    A TrainerCallback that records, at the end of every epoch, the probability
    the model assigns to the gold label of each training example. Used with
    `DataMapsScorer` (Swayamdipta et al., 2020, "Dataset Cartography",
    https://arxiv.org/abs/2009.10795).

    Usage:
        callback = DataMapsCallback()
        trainer = Trainer(..., callbacks=[callback])
        callback.trainer = trainer
        trainer.train()
    """

    def __init__(self):
        # One list of gold-label probabilities (one entry per epoch) per example index.
        self.gold_probs: dict = {}
        self.trainer: Optional[Trainer] = None

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.trainer is None or self.trainer.train_dataset is None:
            return

        predictions = self.trainer.predict(self.trainer.train_dataset)
        if predictions.predictions is None or predictions.label_ids is None:
            return

        logits = np.asarray(predictions.predictions, dtype=np.float64)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        labels = np.asarray(predictions.label_ids).astype(int)
        gold = probs[np.arange(len(labels)), labels]

        for i, p in enumerate(gold):
            self.gold_probs.setdefault(i, []).append(float(p))

    def calculate_statistics(self) -> dict:
        """
        Returns a dict with per-example lists `confidence` (mean gold-label
        probability across epochs), `variability` (std of that probability) and
        `correctness` (fraction of epochs with gold probability > 0.5).
        """
        if not self.gold_probs:
            return {"confidence": [], "variability": [], "correctness": []}
        n = max(self.gold_probs.keys()) + 1
        conf, var, corr = [0.0] * n, [0.0] * n, [0.0] * n
        for i, p in self.gold_probs.items():
            arr = np.asarray(p, dtype=np.float64)
            conf[i] = float(arr.mean())
            var[i] = float(arr.std())
            corr[i] = float((arr > 0.5).mean())
        return {"confidence": conf, "variability": var, "correctness": corr}
