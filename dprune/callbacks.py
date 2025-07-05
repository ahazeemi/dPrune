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