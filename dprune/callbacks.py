import numpy as np
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class ForgettingCallback(TrainerCallback):
    """
    A TrainerCallback to record learning events during training to later
    calculate forgetting scores.

    An example is "forgotten" if it is misclassified at an epoch after having
    been correctly classified in a previous epoch.

    Usage:
        callback = ForgettingCallback()
        trainer = Trainer(..., callbacks=[callback])
        trainer.train()
        scores = callback.calculate_forgetting_scores()
    """

    def __init__(self):
        self.learning_events = {}  # Using a dict: {example_index: [events]}

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        At the end of each epoch, get predictions for the entire training set
        and record whether each example was classified correctly.
        """
        trainer = kwargs.get("trainer")
        if trainer is None or trainer.train_dataset is None:
            return

        # This can be computationally expensive on large datasets.
        # It's a known trade-off for this kind of analysis.
        predictions = trainer.predict(trainer.train_dataset)

        # Ensure predictions and labels are available
        if predictions.predictions is None or predictions.label_ids is None:
            return

        predicted_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids

        for i, (pred, true) in enumerate(zip(predicted_labels, true_labels)):
            if i not in self.learning_events:
                self.learning_events[i] = []
            self.learning_events[i].append(1 if pred == true else 0)

    def calculate_forgetting_scores(self) -> list[int]:
        """
        Calculates the forgetting score for each example based on the
        recorded learning events.
        """
        if not self.learning_events:
            return []

        max_index = max(self.learning_events.keys())
        # Ensure we have a score for every example up to the max index
        forgetting_scores = [0] * (max_index + 1)

        for i, events in self.learning_events.items():
            if len(events) < 2:
                continue  # Need at least two epochs to have a transition

            # A transition from correct (1) to incorrect (0) is a forget event
            transitions = zip(events, events[1:])
            forget_count = sum(1 for prev, curr in transitions if prev == 1 and curr == 0)
            forgetting_scores[i] = forget_count

        return forgetting_scores 