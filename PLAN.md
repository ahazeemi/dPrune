# Auto Data Pruning: Combining Karpathy's autoresearch with dPrune

## Concept

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) uses an AI agent loop to autonomously experiment with model architecture and hyperparameters: edit `train.py` → train 5 min → check val_bpb → keep or revert → repeat. We apply the **same autonomous loop pattern** but shift the experimentation target from model code to **data pruning strategies** using dPrune.

Instead of asking "what model architecture is best?", we ask: **"what data subset is best for training this model?"**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   AI Agent (Claude/Codex)            │
│                                                      │
│  Reads program.md → Edits prune.py → Observes metric │
└──────────┬──────────────────────────────┬────────────┘
           │                              │
           ▼                              ▼
┌─────────────────┐           ┌─────────────────────┐
│   prune.py      │           │   train.py          │
│   (MUTABLE)     │           │   (FIXED or MUTABLE)│
│                 │           │                     │
│  dPrune Scorer  │──────────▶│  Train on pruned    │
│  dPrune Pruner  │  pruned   │  dataset, report    │
│  dPrune Pipeline│  dataset  │  val_bpb            │
└─────────────────┘           └─────────────────────┘
           │
           ▼
┌─────────────────┐
│   prepare.py    │
│   (FIXED)       │
│                 │
│  Data download  │
│  Tokenizer      │
│  Full dataset   │
└─────────────────┘
```

---

## Step-by-Step Plan

### Phase 1: Fork & Scaffold (Day 1)

1. **Fork autoresearch** and create an `auto-data-pruning` variant.
2. **Add dPrune as a dependency** (`pip install dprune` or submodule).
3. **Create three core files** following autoresearch's minimal philosophy:

   | File | Role | Agent edits? |
   |------|------|-------------|
   | `prepare.py` | Downloads data, trains tokenizer, builds full HF dataset | No (fixed) |
   | `prune.py` | Contains the dPrune pipeline — scorer config, pruner config, pruning ratio | **Yes (mutable)** |
   | `train.py` | Loads pruned dataset, trains GPT, reports val_bpb | No (fixed initially) |
   | `program.md` | Instructions for the AI agent | Human-edited |

4. **Baseline `prune.py`** — start with a no-op (random pruner, k=1.0 = keep all data) so the first run establishes a baseline val_bpb.

### Phase 2: The Mutable `prune.py` (Day 1-2)

The agent's search space is everything in `prune.py`. This is the file the AI agent iterates on. It should expose:

```python
# prune.py — the agent edits this file
from dprune import PruningPipeline
from dprune.scorers import (
    CrossEntropyScorer,
    KMeansCentroidDistanceScorer,
    PerplexityScorer,
    ForgettingScorer,
)
from dprune.pruners import TopKPruner, BottomKPruner, StratifiedPruner, RandomPruner

def prune_dataset(full_dataset, model=None, tokenizer=None):
    """Called by train.py. Returns a pruned HF Dataset."""

    # --- SCORER (agent experiments here) ---
    scorer = PerplexityScorer(
        model_path="models/en.arpa.bin",
        text_column="text",
    )

    # --- PRUNER (agent experiments here) ---
    pruner = StratifiedPruner(k=0.5, num_strata=10)

    # --- PIPELINE ---
    pipeline = PruningPipeline(scorer=scorer, pruner=pruner)
    return pipeline.run(full_dataset)
```

**What the agent can vary:**
- Scorer type (perplexity vs. cross-entropy vs. k-means vs. forgetting)
- Scorer hyperparameters (num_clusters, batch_size, model choice)
- Pruner type (top-k vs. bottom-k vs. stratified vs. random)
- Pruning ratio k (0.1 to 1.0)
- Stratification params (num_strata)
- Composite strategies (e.g., score with perplexity, then re-score top-50% with cross-entropy)
- Custom scoring functions the agent invents

### Phase 3: Adapt `train.py` (Day 2)

Modify autoresearch's `train.py` to:

1. **Import and call `prune_dataset()`** before training begins.
2. **Time-budget the pruning step** separately — e.g., pruning gets 1 min, training gets 4 min of the 5-min budget. Or keep pruning outside the clock.
3. **Log pruning metadata** alongside val_bpb: dataset size after pruning, scorer used, pruner used, ratio.

```python
# In train.py, before training loop:
from prune import prune_dataset

full_dataset = load_full_dataset()  # from prepare.py
pruned_dataset = prune_dataset(full_dataset, model, tokenizer)
print(f"Pruned: {len(full_dataset)} → {len(pruned_dataset)} examples")
# ... proceed with training on pruned_dataset
```

### Phase 4: Write `program.md` (Day 2-3)

The agent instruction file. Key directives:

```markdown
# Auto Data Pruning Program

## Goal
Find the data pruning strategy that minimizes val_bpb on the validation set.

## Rules
1. You may ONLY edit `prune.py`.
2. The `prune_dataset()` function must accept (full_dataset, model, tokenizer)
   and return a HuggingFace Dataset.
3. Pruning must complete within 60 seconds.
4. You have access to all dPrune scorers and pruners.
5. You may write custom scoring logic inside prune.py.
6. After each run, check val_bpb. If it improved, keep. If not, revert.

## Search Strategy Suggestions
- Start with baselines: random pruning at various ratios (0.3, 0.5, 0.7)
- Try each scorer type with top-k and bottom-k
- Explore stratified sampling for diversity
- Try aggressive pruning (k=0.2) — less data can mean more epochs in 5 min
- Combine scorers: prune noisy data first, then select hard examples
- Consider that removing data means more training epochs fit in the time budget

## Current Best
val_bpb: [auto-updated after each run]
strategy: [auto-updated after each run]
```

### Phase 5: The Experiment Loop (Day 3+)

Run the autoresearch loop with the data pruning target:

```
Agent reads program.md
  → Edits prune.py (changes scorer, pruner, ratio, etc.)
  → System runs: python prune.py && python train.py
  → Agent observes val_bpb
  → If improved: git commit (keep)
  → If worse: git revert
  → Repeat (~12 experiments/hour, ~100 overnight)
```

### Phase 6: Extensions (Week 2+)

1. **Dual-mutable mode**: Let the agent edit BOTH `prune.py` and `train.py` — co-optimize data selection and model architecture together.
2. **Multi-stage pruning**: Score → prune → train briefly → re-score with trained model → prune again → train fully.
3. **Curriculum learning**: Agent experiments with training-order strategies using dPrune scores (easy-to-hard, hard-to-easy, mixed).
4. **Scale experiments**: Move from single-GPU nanochat to larger datasets/models where data pruning has even more impact.
5. **Leaderboard**: Track best pruning strategies across community forks (aligned with Karpathy's SETI@home vision).

---

## Why This Works

| autoresearch strength | How it applies to data pruning |
|---|---|
| Fixed time budget (5 min) | Aggressive pruning = more epochs in same time = potentially better model |
| Single metric (val_bpb) | Clean signal for comparing pruning strategies |
| Keep-or-revert via git | Safe exploration of pruning parameter space |
| Minimal codebase (~630 LOC) | dPrune's modular API fits naturally as a composable layer |
| Agent autonomy | Large combinatorial space of (scorer × pruner × ratio × params) is perfect for autonomous search |

## Key Insight

In autoresearch, the agent explores the space of *model architectures*. Here, the agent explores the space of *training data subsets*. The two are **complementary** — and in dual-mutable mode, you can search both simultaneously. Data pruning also has a **meta-benefit** in the autoresearch setting: smaller datasets = faster training = more experiments per hour = faster research progress.

---

## Files to Create/Modify in dPrune Repo

1. `autoresearch/prepare.py` — data prep (adapted from autoresearch)
2. `autoresearch/prune.py` — mutable dPrune pipeline (new)
3. `autoresearch/train.py` — training script (adapted from autoresearch)
4. `autoresearch/program.md` — agent instructions (new)
5. `autoresearch/README.md` — setup & usage guide
6. `dprune/scorers/` — potentially new scorers discovered during experiments
