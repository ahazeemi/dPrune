import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from datasets import Dataset
from dprune.callbacks import ForgettingCallback

@pytest.fixture
def mock_trainer():
    """Mocks the Trainer class."""
    trainer = MagicMock()
    # Mock dataset
    trainer.train_dataset = Dataset.from_dict({
        'text': ['a', 'b', 'c', 'd'],
        'label': [0, 1, 0, 1]
    })
    return trainer

def test_forgetting_callback_on_epoch_end(mock_trainer):
    callback = ForgettingCallback()
    # Set the trainer on the callback (as done in real usage)
    callback.trainer = mock_trainer
    
    # --- Epoch 1: two correct, two incorrect ---
    mock_trainer.predict.return_value = Mock(
        predictions=np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.7, 0.3]]), # preds: 0, 1, 0, 0
        label_ids=np.array([0, 1, 0, 1]) # labels: 0, 1, 0, 1
    )
    # Expected learning events: 1, 1, 1, 0 (correct, correct, correct, incorrect)
    callback.on_epoch_end(args=None, state=None, control=None)
    
    assert callback.learning_events[0] == [1]
    assert callback.learning_events[1] == [1]
    assert callback.learning_events[2] == [1]
    assert callback.learning_events[3] == [0]

    # --- Epoch 2: one is forgotten ---
    mock_trainer.predict.return_value = Mock(
        predictions=np.array([[0.1, 0.9], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]), # preds: 1, 1, 0, 1
        label_ids=np.array([0, 1, 0, 1]) # labels: 0, 1, 0, 1
    )
    # Expected learning events for example 0: [1, 0] (forgotten)
    callback.on_epoch_end(args=None, state=None, control=None)

    assert callback.learning_events[0] == [1, 0]
    assert callback.learning_events[1] == [1, 1]
    assert callback.learning_events[2] == [1, 1]
    assert callback.learning_events[3] == [0, 1]


def test_calculate_forgetting_scores():
    callback = ForgettingCallback()
    # Mock learning events manually
    callback.learning_events = {
        0: [1, 0, 1], # Forgotten once
        1: [0, 1, 1], # Never forgotten
        2: [1, 1, 1], # Never forgotten
        3: [1, 0, 0], # Forgotten once
        4: [0, 0, 0], # Never learned
        5: [1, 0, 1, 0, 1] # Forgotten twice
    }
    
    scores = callback.calculate_forgetting_scores()
    
    assert scores == [1, 0, 0, 1, 0, 2] 