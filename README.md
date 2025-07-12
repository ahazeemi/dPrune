# 🌿 dPrune: A Framework for Data Pruning

[![CI](https://github.com/ahazeemi/dPrune/actions/workflows/ci.yml/badge.svg)](https://github.com/ahazeemi/dPrune/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/dprune.svg)](https://badge.fury.io/py/dprune)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`dPrune` is a lightweight, extensible Python library designed to make data selection and pruning simple and accessible for NLP and speech tasks, with first-class support for Hugging Face `datasets` and `transformers`.

Data pruning is the process of selecting a smaller, more informative, and a higher quality subset of a large training dataset. This can lead to faster training, lower computational costs, and even better model performance by removing noisy or redundant examples. `dPrune` provides a modular framework to experiment with various pruning strategies.

---

## ⭐ Key Features

- **Hugging Face Integration**: Works seamlessly with huggingface `datasets` and `transformers`.
- **Modular Design**: Separates the scoring logic from the pruning criteria.
- **Extensible**: Easily create your own custom scoring functions and pruning methods.
- **Supervised & Unsupervised Scoring Methods**: Includes a variety of common pruning techniques.
  - **Supervised**: Score data based on model outputs (e.g., cross-entropy loss, forgetting scores).
  - **Unsupervised**: Score data based on intrinsic properties (e.g., clustering embeddings, perplexity scores).
- **Multiple Pruning Strategies**: Supports top/bottom-k pruning, stratified sampling, and random pruning.

## 📦 Installation

You can install `dPrune` via pip:

```bash
pip install dprune
```

Alternatively, you can use [`uv`](https://github.com/astral-sh/uv):

```bash
uv pip install dprune
```

To install the library with all testing dependencies, run:

```bash
pip install "dprune[test]"
```

## 🚀 Quick Start

Here's a simple example of how to prune a dataset using unsupervised KMeans clustering. This approach keeps the most representative examples (closest to cluster centroids) without requiring labels or fine-tuning.

```python
from datasets import Dataset
from dprune import PruningPipeline, KMeansCentroidDistanceScorer, BottomKPruner

data = {'text': ['A great movie!', 'Waste of time.', 'Amazing.', 'So predictable.']}
raw_dataset = Dataset.from_dict(data)

scorer = KMeansCentroidDistanceScorer(
    model=model,
    tokenizer=tokenizer,
    text_column='text',
    num_clusters=2
)
pruner = BottomKPruner(k=0.5)

pipeline = PruningPipeline(scorer=scorer, pruner=pruner)
pruned_dataset = pipeline.run(raw_dataset)

print(f"Original dataset size: {len(raw_dataset)}")
print(f"Pruned dataset size: {len(pruned_dataset)}")
```

## 💡 Core Concepts

`dPrune` is built around three core components:

#### `Scorer`

A `Scorer` takes a `Dataset` and adds a new `score` column to it. The score is a numerical value that represents some property of the example (e.g., how hard it is for the model to classify).

#### `Pruner`

A `Pruner` takes a scored `Dataset` and selects a subset of it based on the `score` column.

#### `PruningPipeline`

The `PruningPipeline` is a convenience wrapper that chains a `Scorer` and a `Pruner` together into a single, easy-to-use workflow.

## 🛠️ Available Components

### Scorers

- **`KMeansCentroidDistanceScorer`**: (Unsupervised) Embeds the data, performs k-means clustering, and scores each example by its distance to its cluster centroid.
- **`PerplexityScorer`**: (Unsupervised) Calculates perplexity score for each example using the KenLM n-gram language model.
- **`CrossEntropyScorer`**: (Supervised) Scores examples based on the cross-entropy loss from a given model.
- **`ForgettingScorer`**: (Supervised) Works with a `ForgettingCallback` to score examples based on how many times they are "forgotten" during training.
- ...many more coming soon!


### Pruners

- **`TopKPruner`**: Selects the `k` examples with the highest scores.
- **`BottomKPruner`**: Selects the `k` examples with the lowest scores.
- **`StratifiedPruner`**: Divides the data into strata based on score quantiles and samples proportionally from each.
- **`RandomPruner`**: Randomly selects `k` examples, ignoring scores. Useful for establishing a baseline.

### Callbacks

- **`ForgettingCallback`**: A `TrainerCallback` that records learning events during training to be used with the `ForgettingScorer`.

## 🎨 Extending dPrune

Creating your own custom components is straightforward.

### Custom Scorer

Simply inherit from the `Scorer` base class and implement the `score` method.

```python
from dprune import Scorer
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
from dprune import Pruner
from datasets import Dataset

class ThresholdPruner(Pruner):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def prune(self, scored_dataset: Dataset, **kwargs) -> Dataset:
        indices_to_keep = [i for i, score in enumerate(scored_dataset['score']) if score > self.threshold]
        return scored_dataset.select(indices_to_keep)
```

## 📓 Example Notebooks

### 1. Supervised Pruning with Forgetting Score
`examples/supervised_pruning_with_forgetting_score.ipynb`

Shows how to use forgetting scores to prune dataset.

### 2. Unsupervised Pruning with K-Means
`examples/unsupervised_pruning_with_kmeans.ipynb` 

Demonstrates clustering-based pruning using K-means to remove outlier examples.

### 3. Unsupervised Pruning with Perplexity
`examples/unsupervised_pruning_with_perplexity.ipynb`

Shows how to use perplexity scoring for data pruning in text summarization.

## 🎓 Advanced Usage: Forgetting Score

Some pruning strategies require observing the model's behavior _during_ training. `dPrune` supports this via Hugging Face `TrainerCallback`. Here is how you would use the `ForgettingScorer`:

```python
from dprune import ForgettingCallback, ForgettingScorer

# 1. Initialize the callback and trainer
forgetting_callback = ForgettingCallback()
trainer = Trainer(
    model=model,
    train_dataset=raw_dataset,
    callbacks=[forgetting_callback],
)

# 2. Assign the trainer to the callback
forgetting_callback.trainer = trainer

# 3. Train the model. The callback will record events automatically.
trainer.train()

# 4. Create the scorer from the populated callback
scorer = ForgettingScorer(forgetting_callback)

# 5. Use the scorer in a pipeline as usual
pipeline = PruningPipeline(scorer=scorer, pruner=TopKPruner(k=0.8)) # Keep 80%
pruned_dataset = pipeline.run(raw_dataset)

print(f"Pruned with forgetting scores, final size: {len(pruned_dataset)}")
```

## 🧪 Running Tests

To run the full test suite, clone the repository and run `pytest` from the root directory:

```bash
git clone https://github.com/ahazeemi/dPrune.git
cd dPrune
# Install in editable mode with test dependencies
pip install -e ".[test]"
# Or, with uv
uv pip install -e ".[test]"

pytest
```

## 🤝 Contributing

Contributions are welcome! If you have a feature request, bug report, or want to add a new scorer or pruner, please open an issue or submit a pull request on GitHub.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.


## 📝 Citation

If you use `dPrune` in your research, please cite it as follows:

```bibtex
@software{dprune2025,
  author = {Azeemi, Abdul Hameed and Qazi, Ihsan Ayyub and Raza, Agha Ali},
  title = {dPrune: A Framework for Data Pruning},
  year = {2025},
  url = {https://github.com/ahazeemi/dPrune}
}
```

Alternatively, you can cite it in text as:

> Abdul Hameed Azeemi, Ihsan Ayyub Qazi, and Agha Ali Raza. (2025). dPrune: A Framework for Data Pruning. https://github.com/ahazeemi/dPrune
