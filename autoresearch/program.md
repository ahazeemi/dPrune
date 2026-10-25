# Auto Data Pruning — Agent Instructions

You are an autonomous AI research agent. Your goal is to find the **optimal data pruning strategy** that minimizes `val_bpb` (validation bits per byte) for a small GPT model.

## Setup

This project uses [dPrune](https://github.com/ahazeemi/dPrune) for data pruning and follows the [autoresearch](https://github.com/karpathy/autoresearch) pattern of autonomous experimentation.

Three files:
- `prepare.py` — **DO NOT EDIT.** Data download, tokenizer, dataloaders, evaluation.
- `train.py` — **DO NOT EDIT** (unless in dual-mutable mode). GPT model, training loop, reports val_bpb.
- `prune.py` — **EDIT THIS FILE.** Contains the dPrune pruning pipeline. This is your search space.

## Rules

1. **Only edit `prune.py`** (the mutable file).
2. The `prune_dataset(full_dataset, model=None, tokenizer=None)` function signature must not change.
3. It must return a HuggingFace `Dataset` with at least a `text` column.
4. **Pruning must complete within 60 seconds.** If it times out, the run is invalid.
5. After each edit, run `python train.py` and observe the reported `val_bpb`.
6. **If val_bpb improved → `git commit` (keep the change).**
7. **If val_bpb worsened → `git restore prune.py` (revert).**
8. Repeat. Target ~12 experiments per hour.

## Metric

**`val_bpb`** (validation bits per byte) — **lower is better**.

This metric is vocabulary-size-independent, so changes to tokenization or data composition are fairly compared. Results are appended to `results.json` after each run.

## Available Tools (dPrune API)

### Scorers
Score each example in the dataset with a numeric quality/difficulty signal:

| Scorer | Type | What it measures | Notes |
|--------|------|-----------------|-------|
| `PerplexityScorer` | Unsupervised | KenLM perplexity | Fast. High ppl = harder examples. Needs KenLM model. |
| `KMeansCentroidDistanceScorer` | Unsupervised | Distance to cluster centroid | Needs model+tokenizer for embeddings. Low dist = representative. |
| `Random` | Baseline | No scoring, random selection only | Cheapest baseline; useful for ratio sweeps. |

This TinyStories workflow is text-only. Do not use `CrossEntropyScorer` or `ForgettingScorer` here unless you first add labels / forgetting-event support to the pipeline.

### Pruners
Select a subset based on scores:

| Pruner | Strategy |
|--------|----------|
| `TopKPruner(k)` | Keep highest-scoring examples |
| `BottomKPruner(k)` | Keep lowest-scoring examples |
| `StratifiedPruner(k, num_strata)` | Proportional sampling across score quantiles |
| `RandomPruner(k)` | Random baseline (ignores scores) |

`k` can be a float (0.0–1.0 for percentage) or int (exact count).

### Pipeline
```python
from dprune import PruningPipeline
pipeline = PruningPipeline(scorer=scorer, pruner=pruner)
pruned = pipeline.run(full_dataset)
```

## Search Strategy

Follow this progression (approximately):

### Phase 1: Baselines (experiments 1–5)
Establish baselines with random pruning at different ratios:
- `RandomPruner(k=1.0)` — no pruning (baseline)
- `RandomPruner(k=0.7)` — 70% of data
- `RandomPruner(k=0.5)` — 50% of data
- `RandomPruner(k=0.3)` — 30% of data
- `RandomPruner(k=0.1)` — 10% of data

**Key insight:** Less data = more epochs in the 5-minute budget. There's a sweet spot.

### Phase 2: Scorer exploration (experiments 6–20)
Try each scorer with the best ratio from Phase 1:
- Random + different ratios as a sanity check
- Perplexity + TopK (keep hard examples)
- Perplexity + BottomK (keep easy examples)
- Perplexity + Stratified (balanced diversity)
- KMeans distance + TopK (keep outliers)
- KMeans distance + BottomK (keep representative core)

### Phase 3: Hyperparameter tuning (experiments 21–40)
Refine the best strategy:
- Fine-tune the pruning ratio (e.g., 0.35 vs 0.40 vs 0.45)
- Adjust num_strata for StratifiedPruner
- Adjust num_clusters for KMeans
- Try different combinations

### Phase 4: Creative strategies (experiments 41+)
Invent new approaches:
- **Two-stage pruning:** First remove noise (bottom 20% by perplexity), then select diverse subset (stratified)
- **Curriculum-aware:** Sort by difficulty, keep a mix of easy (for stable gradients) and hard (for learning signal)
- **Custom scoring functions** — write your own scorer logic in prune.py
- **Ensemble scoring** — combine multiple scorer signals

## Important Notes

- The TinyStories dataset has ~2.1M training examples. Even k=0.1 gives ~210K examples.
- Smaller pruned datasets mean more passes through the data in the fixed time budget.
- Quality > quantity is the hypothesis. But too little data causes overfitting.
- Watch for the **epoch count** in train output — if epochs >> 5, you may be overfitting.
- Results are saved in `results.json`. Review the history to track progress.

## Current Best

```
val_bpb: [pending first run]
strategy: [pending first run]
```

Update this section after each improvement.
