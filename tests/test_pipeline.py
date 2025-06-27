import pytest
from unittest.mock import Mock
from datasets import Dataset

from dprune.base import Scorer, Pruner
from dprune.pipeline import PruningPipeline


@pytest.fixture
def mock_scorer():
    """Mocks the Scorer class."""
    scorer = Mock(spec=Scorer)
    # The score method should return a dataset with a 'score' column
    scorer.score.return_value = Dataset.from_dict({'score': [1, 2, 3]})
    return scorer


@pytest.fixture
def mock_pruner():
    """Mocks the Pruner class."""
    pruner = Mock(spec=Pruner)
    # The prune method should return the final pruned dataset
    pruner.prune.return_value = Dataset.from_dict({'score': [3]})
    return pruner


@pytest.fixture
def initial_dataset():
    """Provides a dummy initial dataset."""
    return Dataset.from_dict({'text': ['a', 'b', 'c']})


def test_pruning_pipeline(mock_scorer, mock_pruner, initial_dataset):
    """
    Tests that the PruningPipeline correctly calls the scorer and pruner in order.
    """
    # 1. Setup
    pipeline = PruningPipeline(scorer=mock_scorer, pruner=mock_pruner)

    # 2. Run
    final_dataset = pipeline.run(initial_dataset)

    # 3. Assert
    # Check that scorer.score was called once with the initial dataset
    mock_scorer.score.assert_called_once_with(initial_dataset)
    
    # Check that pruner.prune was called once with the result of the scorer
    mock_pruner.prune.assert_called_once_with(mock_scorer.score.return_value)

    # Check that the final result is the output of the pruner
    assert final_dataset == mock_pruner.prune.return_value


def test_pipeline_type_checking():
    """
    Tests that the PruningPipeline raises TypeError for invalid components.
    """
    with pytest.raises(TypeError):
        # Pass a non-scorer object
        PruningPipeline(scorer=object(), pruner=Mock(spec=Pruner))
    
    with pytest.raises(TypeError):
        # Pass a non-pruner object
        PruningPipeline(scorer=Mock(spec=Scorer), pruner=object()) 