import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from dprune.scorers.supervised import CrossEntropyScorer
from dprune.scorers.unsupervised import KMeansCentroidDistanceScorer


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