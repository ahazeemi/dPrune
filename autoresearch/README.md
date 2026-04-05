# Auto Data Pruning with dPrune

Autonomous data pruning experiments following [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) pattern. An AI agent iteratively edits `prune.py` to find the optimal data pruning strategy that minimizes `val_bpb`.

## How It Works

```
Agent reads program.md
  → Edits prune.py (scorer, pruner, ratio)
  → Runs train.py (prune data → train GPT → report val_bpb)
  → If val_bpb improved: git commit (keep)
  → If val_bpb worsened: git revert
  → Repeat (~12 experiments/hour)
```

## Files

| File | Editable? | Purpose |
|------|-----------|---------|
| `prepare.py` | No | Data download, tokenizer, dataloaders, eval |
| `prune.py` | **Yes** | dPrune pipeline config — the agent's target |
| `train.py` | No* | Minimal GPT model + training loop |
| `program.md` | Human | Agent instructions |
| `run.sh` | No | Experiment runner with keep-or-revert logic |

*In dual-mutable mode, train.py can also be edited.

## Quick Start

```bash
# 1. Install dependencies
pip install -e ..                            # install dPrune and repo dependencies from repo root
pip install tokenizers torch transformers    # extra runtime deps for this workflow

# 2. Prepare data (one-time)
python prepare.py

# 3. Run a single experiment
./run.sh

# 4. Or let the agent run autonomously
# Point Claude/Codex at program.md and let it edit prune.py
```

## Running with an AI Agent

```bash
# Option A: Point your agent at program.md
# "Read program.md, then iteratively edit prune.py and run ./run.sh"

# Option B: Use the built-in loop (for manual iteration)
./run.sh --loop 20   # Run 20 experiments with auto keep-or-revert
```

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA for GPU training)
- dPrune (this repo)
- `tokenizers` (for BPE tokenizer)
- `transformers` (for KMeans embedding scorer)
- Optional: `kenlm` (for perplexity scoring)

## The Search Space

The agent explores combinations of:

- **Scorers**: Random, Perplexity, KMeans distance
- **Pruners**: TopK, BottomK, Stratified, Random
- **Ratios**: 0.1 to 1.0
- **Hyperparams**: num_clusters, num_strata, model choice
- **Compositions**: Multi-stage pruning, ensemble scoring

`CrossEntropy` and `Forgetting` are intentionally not part of this default TinyStories workflow because the dataset does not provide labels and the training loop does not record forgetting events.
