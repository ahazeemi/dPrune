import pytest
from datasets import Dataset
from dprune.pruners.selection import TopKPruner, BottomKPruner, StratifiedPruner, RandomPruner
import numpy as np


@pytest.fixture
def scored_dataset():
    """A pytest fixture to create a dummy scored dataset."""
    data = {
        'text': [f'text_{i}' for i in range(10)],
        'score': [float(i) for i in range(10)]  # scores from 0.0 to 9.0
    }
    return Dataset.from_dict(data)


@pytest.fixture
def large_scored_dataset():
    """A pytest fixture for a larger, more varied dataset."""
    num_samples = 100
    data = {
        'text': [f'text_{i}' for i in range(num_samples)],
        # Scores with some randomness to ensure uniqueness for quantiles
        'score': np.linspace(0, 99, num_samples) + np.random.rand(num_samples) * 0.1
    }
    return Dataset.from_dict(data)


def test_top_k_pruner_int(scored_dataset):
    pruner = TopKPruner(k=3)
    pruned_dataset = pruner.prune(scored_dataset)
    assert len(pruned_dataset) == 3
    # Top 3 scores are 9.0, 8.0, 7.0
    assert pruned_dataset['score'] == [9.0, 8.0, 7.0]


def test_top_k_pruner_float(scored_dataset):
    pruner = TopKPruner(k=0.5)  # keep 50%
    pruned_dataset = pruner.prune(scored_dataset)
    assert len(pruned_dataset) == 5
    # Top 5 scores
    assert pruned_dataset['score'] == [9.0, 8.0, 7.0, 6.0, 5.0]


def test_bottom_k_pruner_int(scored_dataset):
    pruner = BottomKPruner(k=3)
    pruned_dataset = pruner.prune(scored_dataset)
    assert len(pruned_dataset) == 3
    # Bottom 3 scores are 0.0, 1.0, 2.0
    assert pruned_dataset['score'] == [0.0, 1.0, 2.0]


def test_bottom_k_pruner_float(scored_dataset):
    pruner = BottomKPruner(k=0.5)  # keep 50%
    pruned_dataset = pruner.prune(scored_dataset)
    assert len(pruned_dataset) == 5
    # Bottom 5 scores
    assert pruned_dataset['score'] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_pruner_invalid_float_k(scored_dataset):
    with pytest.raises(ValueError):
        pruner = TopKPruner(k=1.1)
        pruner.prune(scored_dataset)

    with pytest.raises(ValueError):
        pruner = BottomKPruner(k=-0.1)
        pruner.prune(scored_dataset)


def test_stratified_pruner(large_scored_dataset):
    """Tests the StratifiedPruner."""
    num_strata = 5
    k = 0.5  # Keep 50%
    pruner = StratifiedPruner(k=k, num_strata=num_strata)

    pruned_dataset = pruner.prune(large_scored_dataset)

    # Check if the number of examples is roughly correct
    expected_size = int(len(large_scored_dataset) * k)
    # The size can be slightly larger due to `ceil`
    assert abs(len(pruned_dataset) - expected_size) <= num_strata

    # Check that scores are from different parts of the distribution
    original_scores = np.array(large_scored_dataset['score'])
    pruned_scores = np.array(pruned_dataset['score'])

    # Check that we have both low and high scores in the pruned set
    assert np.min(pruned_scores) < np.median(original_scores)
    assert np.max(pruned_scores) > np.median(original_scores)


def test_random_pruner(large_scored_dataset):
    """Tests the RandomPruner."""
    k = 0.3 # Keep 30%
    pruner = RandomPruner(k=k)
    
    pruned_dataset = pruner.prune(large_scored_dataset)
    
    expected_size = int(len(large_scored_dataset) * k)

    assert len(pruned_dataset) == expected_size
    # Verify that the selection is not sorted by score
    original_scores = large_scored_dataset.sort("score")['score']
    pruned_scores = pruned_dataset.sort("score")['score']
    assert original_scores[:expected_size] != pruned_scores 