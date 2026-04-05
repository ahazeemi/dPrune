"""
Data pruning configuration for auto data pruning experiments.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset
from dprune import PruningPipeline
from dprune.scorers.unsupervised import KMeansCentroidDistanceScorer, PerplexityScorer
from dprune.pruners.selection import (
    TopKPruner,
    BottomKPruner,
    StratifiedPruner,
    RandomPruner,
)

PRUNE_TIME_BUDGET = 60

SCORER_TYPE = "random"  # Options: "random", "perplexity", "kmeans"
PRUNER_TYPE = "random"  # Options: "topk", "bottomk", "stratified", "random"
PRUNE_RATIO = 1.0       # Float 0.0-1.0: fraction of data to KEEP


def prune_dataset(full_dataset: Dataset, model=None, tokenizer=None) -> Dataset:
    """Prune a text dataset using the configured scorer and pruner."""
    start_time = time.time()
    text_column = "text"
    original_size = len(full_dataset)

    if text_column not in full_dataset.column_names:
        raise ValueError(
            f"Expected dataset to include a '{text_column}' column, "
            f"but found {full_dataset.column_names}"
        )

    if PRUNE_RATIO >= 1.0 and SCORER_TYPE == "random":
        print(f"[prune] No-op: keeping all {original_size:,} examples")
        return full_dataset

    if SCORER_TYPE == "random":
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
    else:
        raise ValueError(
            f"Unknown or unsupported scorer type for TinyStories: {SCORER_TYPE}. "
            "Supported scorers are: random, perplexity, kmeans."
        )

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

    if scorer is not None:
        pipeline = PruningPipeline(scorer=scorer, pruner=pruner)
        pruned = pipeline.run(full_dataset)
    else:
        dummy_scores = [0.0] * len(full_dataset)
        scored = full_dataset.add_column("score", dummy_scores)
        pruned = pruner.prune(scored)

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


if __name__ == "__main__":
    from prepare import load_hf_dataset

    print("Loading dataset for pruning test...")
    ds = load_hf_dataset(split="train", max_examples=1000)
    print(f"Loaded {len(ds)} examples")

    pruned = prune_dataset(ds)
    print(f"Pruned to {len(pruned)} examples")
    print(f"Sample: {pruned[0]['text'][:100]}...")
