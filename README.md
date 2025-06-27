# dPrune: A Framework for Data Pruning 

[![PyPI version](https://badge.fury.io/py/dprune.svg)](https://badge.fury.io/py/dprune)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`dPrune` is a lightweight, extensible Python library designed to make data pruning simple and accessible for NLP tasks, with first-class support for Hugging Face `datasets` and `transformers`.

Data pruning is the process of selecting a smaller, more informative subset of a large training dataset. This can lead to faster training, lower computational costs, and sometimes even better model performance by removing noisy or redundant examples. `dPrune` provides a modular framework to experiment with various pruning strategies.

---

## Key Features

- **Modular Design**: Separates the scoring logic from the pruning criteria, allowing you to mix and match strategies.
- **Extensible**: Easily create your own custom scoring functions and pruning methods.
- **Hugging Face Integration**: Works seamlessly with `datasets` and `transformers`.
- **Supervised & Unsupervised Methods**: Includes a variety of common pruning techniques.
  - **Supervised**: Score data based on model outputs (e.g., cross-entropy loss).
  - **Unsupervised**: Score data based on intrinsic properties (e.g., clustering embeddings).
- **Rich Pruning Strategies**: Supports top/bottom-k selection and stratified sampling to preserve data distribution.

## Installation

You can install `dPrune` via pip:

```bash
pip install dprune
```

Alternatively, you can use [`uv`](https://github.com/astral-sh/uv), which is a fast, drop-in replacement for `pip`:
```bash
uv pip install dprune
```

To install the library with all testing dependencies, run:

```bash
# with pip
pip install "dprune[test]"

# with uv
uv pip install "dprune[test]"
```

## Quick Start

Here's a simple example of how to prune a dataset. In this case, we'll keep the 50% of examples that have the highest cross-entropy loss according to a fine-tuned model. These are often considered the "hardest" or most informative examples.

```python
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Assuming dprune is installed
from dprune.pipeline import PruningPipeline
from dprune.scorers.supervised import CrossEntropyScorer
from dprune.pruners.selection import TopKPruner

# 1. Load your data, model, and tokenizer
data = {'text': ['A great movie!', 'Waste of time.', 'Amazing.', 'So predictable.'], 'label': [1, 0, 1, 0]}
raw_dataset = Dataset.from_dict(data)
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. Fine-tune your model (required for supervised scoring)
# (For brevity, we'll assume the model is already fine-tuned)
# trainer = Trainer(...)
# trainer.train()
# fine_tuned_model = trainer.model

# 3. Define the Scorer and Pruner
scorer = CrossEntropyScorer(model=model, tokenizer=tokenizer, text_column='text', label_column='label')
pruner = TopKPruner(k=0.5)  # Keep the top 50%

# 4. Create and run the pipeline
pipeline = PruningPipeline(scorer=scorer, pruner=pruner)
pruned_dataset = pipeline.run(raw_dataset)

# 5. Get the result
print(f"Original dataset size: {len(raw_dataset)}")
print(f"Pruned dataset size: {len(pruned_dataset)}")
# Expected output:
# Original dataset size: 4
# Pruned dataset size: 2
```

## Core Concepts

`dPrune` is built around three core components:

#### `Scorer`
A `Scorer` takes a `Dataset` and adds a new `score` column to it. The score is a numerical value that represents some property of the example (e.g., how hard it is for the model to classify).

#### `Pruner`
A `Pruner` takes a scored `Dataset` and selects a subset of it based on the `score` column.

#### `PruningPipeline`
The `PruningPipeline` is a convenience wrapper that chains a `Scorer` and a `Pruner` together into a single, easy-to-use workflow.

## Available Components

### Scorers

- **`CrossEntropyScorer`**: (Supervised) Scores examples based on the cross-entropy loss from a given model.
- **`KMeansCentroidDistanceScorer`**: (Unsupervised) Embeds the data, performs k-means clustering, and scores each example by its distance to its cluster centroid.

### Pruners

- **`TopKPruner`**: Selects the `k` examples with the highest scores.
- **`BottomKPruner`**: Selects the `k` examples with the lowest scores.
- **`StratifiedPruner`**: Divides the data into strata based on score quantiles and samples proportionally from each.

## Extending dPrune

Creating your own custom components is straightforward.

### Custom Scorer

Simply inherit from the `Scorer` base class and implement the `score` method.

```python
from dprune.base import Scorer
from datasets import Dataset
import random

class RandomScorer(Scorer):
    def score(self, dataset: Dataset, **kwargs) -> Dataset:
        scores = [random.random() for _ in range(len(dataset))]
        return dataset.add_column("score", scores)
```

### Custom Pruner

Inherit from the `Pruner` base class and implement the `prune` method.

```python
from dprune.base import Pruner
from datasets import Dataset

class ThresholdPruner(Pruner):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def prune(self, scored_dataset: Dataset, **kwargs) -> Dataset:
        indices_to_keep = [i for i, score in enumerate(scored_dataset['score']) if score > self.threshold]
        return scored_dataset.select(indices_to_keep)
```

## Running Tests

To run the full test suite, clone the repository and run `pytest` from the root directory:

```bash
git clone https://github.com/your-username/dPrune.git
cd dPrune
# Install in editable mode with test dependencies
pip install -e ".[test]"
# Or, with uv
uv pip install -e ".[test]"

pytest
```

## Contributing

Contributions are welcome! If you have a feature request, bug report, or want to add a new scorer or pruner, please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
