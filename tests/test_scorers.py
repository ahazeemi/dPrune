import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from unittest.mock import Mock, patch
from typing import Callable

from dprune.scorers.supervised import CrossEntropyScorer, ForgettingScorer
from dprune.scorers.unsupervised import KMeansCentroidDistanceScorer, PerplexityScorer
from dprune.callbacks import ForgettingCallback


@pytest.fixture(scope="module")
def setup_for_scoring():
    """
    Fixture to set up a dummy dataset, model, and tokenizer for testing scorers.
    The model is fine-tuned for one step to be realistic.
    """
    # 1. Create a dummy dataset
    data = {
        'text': [
            'A great movie!', 'The plot was predictable.',
            'Amazing acting.', 'A waste of time.'
        ],
        'label': [1, 0, 1, 0]
    }
    dataset = Dataset.from_dict(data)

    # 2. Load tokenizer and model
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 3. Fine-tune the model for a single step
    training_args = TrainingArguments(
        output_dir='./test_results',
        num_train_epochs=1,
        per_device_train_batch_size=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    trainer.train()
    
    return {
        "dataset": dataset,
        "model": trainer.model,
        "tokenizer": tokenizer,
    }


def test_cross_entropy_scorer(setup_for_scoring):
    """
    Tests the CrossEntropyScorer.
    """
    scorer = CrossEntropyScorer(
        model=setup_for_scoring["model"],
        tokenizer=setup_for_scoring["tokenizer"],
        text_column='text',
        label_column='label'
    )
    
    scored_dataset = scorer.score(setup_for_scoring["dataset"])

    # Check that a 'score' column was added
    assert 'score' in scored_dataset.column_names
    # Check that the number of scores matches the number of examples
    assert len(scored_dataset['score']) == len(setup_for_scoring["dataset"])
    # Check that scores are floats
    assert isinstance(scored_dataset['score'][0], float)


def test_kmeans_centroid_distance_scorer(setup_for_scoring):
    """
    Tests the KMeansCentroidDistanceScorer.
    """
    scorer = KMeansCentroidDistanceScorer(
        model=setup_for_scoring["model"],
        tokenizer=setup_for_scoring["tokenizer"],
        text_column='text',
        num_clusters=2  # Using 2 clusters for this small dataset
    )
    
    scored_dataset = scorer.score(setup_for_scoring["dataset"])

    # Check that a 'score' column was added
    assert 'score' in scored_dataset.column_names
    # Check that the number of scores matches the number of examples
    assert len(scored_dataset['score']) == len(setup_for_scoring["dataset"])
    # Check that scores are floats
    assert isinstance(scored_dataset['score'][0], float)
    # Check that scores are non-negative (distances)
    assert all(s >= 0 for s in scored_dataset['score'])


def test_forgetting_scorer():
    """Tests the ForgettingScorer."""
    # 1. Mock a ForgettingCallback that has "run"
    mock_callback = Mock(spec=ForgettingCallback)
    mock_callback.calculate_forgetting_scores.return_value = [1, 0, 2]

    # 2. Create a dataset that matches the scores
    dataset = Dataset.from_dict({'text': ['a', 'b', 'c']})

    # 3. Setup and run scorer
    scorer = ForgettingScorer(mock_callback)
    scored_dataset = scorer.score(dataset)

    # 4. Assert
    mock_callback.calculate_forgetting_scores.assert_called_once()
    assert 'score' in scored_dataset.column_names
    assert scored_dataset['score'] == [1, 0, 2]


def test_forgetting_scorer_mismatch_len():
    """Tests that an error is raised if dataset and scores have different lengths."""
    mock_callback = Mock(spec=ForgettingCallback)
    mock_callback.calculate_forgetting_scores.return_value = [1, 0]  # Only 2 scores
    dataset = Dataset.from_dict({'text': ['a', 'b', 'c']})  # 3 examples

    scorer = ForgettingScorer(mock_callback)
    with pytest.raises(ValueError):
        scorer.score(dataset)


def test_perplexity_scorer():
    """Tests the PerplexityScorer with a mocked kenlm model."""
    # Mock the entire kenlm module
    mock_kenlm = Mock()
    mock_model = Mock()
    mock_model.score.side_effect = lambda text: -len(text.split()) * 2.0  # Simple mock scoring
    mock_kenlm.Model.return_value = mock_model

    # Create test dataset
    dataset = Dataset.from_dict({
        'text': ['This is a test.', 'Another test sentence.', 'Short text.']
    })

    # Patch sys.modules to include kenlm
    import sys
    with patch.dict(sys.modules, {'kenlm': mock_kenlm}):
        with patch('dprune.scorers.unsupervised.KENLM_AVAILABLE', True):
            scorer = PerplexityScorer(
                model_path='dummy_path.bin',
                text_column='text',
                batch_size=2
            )

            # Score the dataset
            scored_dataset = scorer.score(dataset)

            # Check that a 'score' column was added
            assert 'score' in scored_dataset.column_names
            # Check that the number of scores matches the number of examples
            assert len(scored_dataset['score']) == len(dataset)
            # Check that scores are floats
            assert all(isinstance(score, float) for score in scored_dataset['score'])
            # Check that scores are positive (perplexity values)
            assert all(score > 0 for score in scored_dataset['score'])


def test_perplexity_scorer_with_custom_normalizer():
    """Tests the PerplexityScorer with a custom text normalizer."""
    # Mock the entire kenlm module
    mock_kenlm = Mock()
    mock_model = Mock()
    mock_model.score.side_effect = lambda text: -len(text.split()) * 2.0
    mock_kenlm.Model.return_value = mock_model

    # Create test dataset
    dataset = Dataset.from_dict({
        'text': ['This is a TEST!!!', 'Another TEST sentence???']
    })

    # Custom normalizer that lowercases and removes punctuation
    def custom_normalizer(text: str) -> str:
        import re
        return re.sub(r'[^\w\s]', '', text.lower()).strip()

    # Patch sys.modules to include kenlm
    import sys
    with patch.dict(sys.modules, {'kenlm': mock_kenlm}):
        with patch('dprune.scorers.unsupervised.KENLM_AVAILABLE', True):
            scorer = PerplexityScorer(
                model_path='dummy_path.bin',
                text_column='text',
                text_normalizer=custom_normalizer
            )

            # Score the dataset
            scored_dataset = scorer.score(dataset)

            # Verify that the normalizer was used by checking the mock calls
            assert mock_model.score.called
            # Check that normalized text was passed to the model
            called_texts = [call[0][0] for call in mock_model.score.call_args_list]
            assert 'this is a test' in called_texts
            assert 'another test sentence' in called_texts


@patch('dprune.scorers.unsupervised.KENLM_AVAILABLE', False)
def test_perplexity_scorer_kenlm_not_available():
    """Tests that PerplexityScorer raises ImportError when kenlm is not available."""
    with pytest.raises(ImportError, match="kenlm is required for PerplexityScorer"):
        PerplexityScorer(
            model_path='dummy_path.bin',
            text_column='text'
        )


def test_perplexity_scorer_invalid_parameters():
    """Tests that PerplexityScorer raises appropriate errors for invalid parameters."""
    mock_kenlm = Mock()
    mock_kenlm.Model.return_value = Mock()

    import sys
    with patch.dict(sys.modules, {'kenlm': mock_kenlm}):
        with patch('dprune.scorers.unsupervised.KENLM_AVAILABLE', True):
            # Test empty model path
            with pytest.raises(ValueError, match="model_path cannot be empty"):
                PerplexityScorer(model_path='', text_column='text')

            # Test empty text column
            with pytest.raises(ValueError, match="text_column cannot be empty"):
                PerplexityScorer(model_path='dummy_path.bin', text_column='')

            # Test invalid batch size
            with pytest.raises(ValueError, match="batch_size must be a positive integer"):
                PerplexityScorer(model_path='dummy_path.bin', text_column='text', batch_size=0)

            with pytest.raises(ValueError, match="batch_size must be a positive integer"):
                PerplexityScorer(model_path='dummy_path.bin', text_column='text', batch_size=-1)


def test_perplexity_scorer_file_not_found():
    """Tests that PerplexityScorer raises FileNotFoundError for invalid model path."""
    mock_kenlm = Mock()
    mock_kenlm.Model.side_effect = Exception("Model file not found")

    import sys
    with patch.dict(sys.modules, {'kenlm': mock_kenlm}):
        with patch('dprune.scorers.unsupervised.KENLM_AVAILABLE', True):
            with pytest.raises(FileNotFoundError, match="Could not load KenLM model"):
                PerplexityScorer(
                    model_path='nonexistent_path.bin',
                    text_column='text'
                )


def test_perplexity_scorer_missing_column():
    """Tests that PerplexityScorer raises ValueError for missing text column."""
    mock_kenlm = Mock()
    mock_kenlm.Model.return_value = Mock()

    import sys
    with patch.dict(sys.modules, {'kenlm': mock_kenlm}):
        with patch('dprune.scorers.unsupervised.KENLM_AVAILABLE', True):
            # Create dataset without the expected text column
            dataset = Dataset.from_dict({
                'content': ['This is a test.', 'Another test sentence.']
            })

            scorer = PerplexityScorer(
                model_path='dummy_path.bin',
                text_column='text'  # Column doesn't exist in dataset
            )

            with pytest.raises(ValueError, match="Column 'text' not found in dataset"):
                scorer.score(dataset)


def test_perplexity_scorer_empty_text_handling():
    """Tests that PerplexityScorer handles empty text correctly."""
    mock_kenlm = Mock()
    mock_model = Mock()
    mock_model.score.side_effect = lambda text: -len(text.split()) * 2.0
    mock_kenlm.Model.return_value = mock_model

    import sys
    with patch.dict(sys.modules, {'kenlm': mock_kenlm}):
        with patch('dprune.scorers.unsupervised.KENLM_AVAILABLE', True):
            # Create dataset with empty and whitespace-only text
            dataset = Dataset.from_dict({
                'text': ['This is normal text.', '', '   ', 'More normal text.']
            })

            scorer = PerplexityScorer(
                model_path='dummy_path.bin',
                text_column='text'
            )

            scored_dataset = scorer.score(dataset)

            # Check that infinite perplexity is assigned to empty text
            scores = scored_dataset['score']
            assert scores[0] > 0 and scores[0] < float('inf')  # Normal text
            assert scores[1] == float('inf')  # Empty text
            assert scores[2] == float('inf')  # Whitespace-only text
            assert scores[3] > 0 and scores[3] < float('inf')  # Normal text 