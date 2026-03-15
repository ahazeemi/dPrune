"""
Data pruning configuration for auto data pruning experiments.

*** THIS IS THE MUTABLE FILE — the AI agent edits THIS file. ***

The agent experiments with different scorers, pruners, ratios, and
compositions to find the data subset that minimizes val_bpb.

The contract:
  - prune_dataset(full_dataset) -> pruned HuggingFace Dataset
  - Must complete within PRUNE_TIME_BUDGET seconds
  - The returned dataset must have a 'text' column
"""

import sys
import os
import time

# Add the repo root so we can import dprune
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset
from dprune import PruningPipeline
from dprune.scorers.unsupervised import KMeansCentroidDistanceScorer, PerplexityScorer
from dprune.scorers.supervised import CrossEntropyScorer, ForgettingScorer
from dprune.pruners.selection import (
    TopKPruner,
    BottomKPruner,
    StratifiedPruner,
    RandomPruner,
)

# ---------------------------------------------------------------------------
# Time budget for pruning (seconds). The rest of the 5-min budget is training.
# ---------------------------------------------------------------------------
PRUNE_TIME_BUDGET = 60

# ---------------------------------------------------------------------------
# Pruning strategy — AGENT: modify everything below this line
# ---------------------------------------------------------------------------

# Baseline: random pruning at 100% (no-op, keeps all data).
# This establishes the baseline val_bpb. The agent should then try
# different strategies to beat it.
SCORER_TYPE = "random"  # Options: "random", "perplexity", "kmeans", "cross_entropy"
PRUNER_TYPE = "random"  # Options: "topk", "bottomk", "stratified", "random"
PRUNE_RATIO = 1.0       # Float 0.0-1.0: fraction of data to KEEP


def prune_dataset(full_dataset: Dataset, model=None, tokenizer=None) -> Dataset:
    """
    Prune the dataset using the configured strategy.

    Args:
        full_dataset: HuggingFace Dataset with at least a 'text' column.
        model: Optional pre-trained model (for supervised scorers).
        tokenizer: Optional tokenizer (for supervised/embedding scorers).

    Returns:
        Pruned HuggingFace Dataset with 'text' column.
    """
    start_time = time.time()
    text_column = "text"
    original_size = len(full_dataset)

    # --- No-op shortcut ---
    if PRUNE_RATIO >= 1.0 and SCORER_TYPE == "random":
        print(f"[prune] No-op: keeping all {original_size:,} examples")
        return full_dataset

    # --- Build scorer ---
    if SCORER_TYPE == "random":
        # RandomPruner doesn't need a scorer; use a dummy
        scorer = None
    elif SCORER_TYPE == "perplexity":
        from dprune.utils import download_kenlm_model
        model_dir = os.path.join(os.path.dirname(__file__), "data", "kenlm")
        model_path = download_kenlm_model(model_dir, lang_id="en")
        scorer = PerplexityScorer(
            model_path=model_path,
            text_column=text_column,
        )
    elif SCORER_TYPE == "kmeans":
        if model is None or tokenizer is None:
            raise ValueError("KMeans scorer requires model and tokenizer")
        scorer = KMeansCentroidDistanceScorer(
            model=model,
            tokenizer=tokenizer,
            text_column=text_column,
            num_clusters=8,
            batch_size=32,
        )
    elif SCORER_TYPE == "cross_entropy":
        if model is None or tokenizer is None:
            raise ValueError("CrossEntropy scorer requires model and tokenizer")
        scorer = CrossEntropyScorer(
            model=model,
            tokenizer=tokenizer,
            text_column=text_column,
            label_column="label",
            batch_size=32,
        )
    else:
        raise ValueError(f"Unknown scorer type: {SCORER_TYPE}")

    # --- Build pruner ---
    if PRUNER_TYPE == "topk":
        pruner = TopKPruner(k=PRUNE_RATIO)
    elif PRUNER_TYPE == "bottomk":
        pruner = BottomKPruner(k=PRUNE_RATIO)
    elif PRUNER_TYPE == "stratified":
        pruner = StratifiedPruner(k=PRUNE_RATIO, num_strata=10)
    elif PRUNER_TYPE == "random":
        pruner = RandomPruner(k=PRUNE_RATIO)
    else:
        raise ValueError(f"Unknown pruner type: {PRUNER_TYPE}")

    # --- Execute pipeline ---
    if scorer is not None:
        pipeline = PruningPipeline(scorer=scorer, pruner=pruner)
        pruned = pipeline.run(full_dataset)
    else:
        # For random pruning, score with dummy scores then prune
        dummy_scores = [0.0] * len(full_dataset)
        scored = full_dataset.add_column("score", dummy_scores)
        pruned = pruner.prune(scored)

    # Remove the score column if present (training doesn't need it)
    if "score" in pruned.column_names:
        pruned = pruned.remove_columns(["score"])

    elapsed = time.time() - start_time
    print(
        f"[prune] {SCORER_TYPE}/{PRUNER_TYPE} k={PRUNE_RATIO}: "
        f"{original_size:,} → {len(pruned):,} examples "
        f"({len(pruned)/original_size:.1%}) in {elapsed:.1f}s"
    )

    if elapsed > PRUNE_TIME_BUDGET:
        print(f"[prune] WARNING: pruning took {elapsed:.1f}s > budget {PRUNE_TIME_BUDGET}s")

    return pruned


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from prepare import load_hf_dataset

    print("Loading dataset for pruning test...")
    ds = load_hf_dataset(split="train", max_examples=1000)
    print(f"Loaded {len(ds)} examples")

    pruned = prune_dataset(ds)
    print(f"Pruned to {len(pruned)} examples")
    print(f"Sample: {pruned[0]['text'][:100]}...")
