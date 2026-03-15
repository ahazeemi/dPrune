"""
Training script for auto data pruning experiments.
Adapted from Karpathy's autoresearch pattern:
- Minimal GPT model
- Fixed 5-minute wall-clock training budget
- Reports val_bpb as the single metric

This file is FIXED by default. In dual-mutable mode, the agent can edit
both this file and prune.py.
"""

import os
import sys
import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add repo root for dprune imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prepare import (
    download_and_tokenize,
    load_tokenizer,
    load_hf_dataset,
    TokenDataset,
    PrunedTokenDataset,
    make_dataloader,
    evaluate_bpb,
    TRAIN_SHARD,
    VAL_SHARD,
    VOCAB_SIZE,
    MAX_SEQ_LEN,
    DATA_DIR,
)
from prune import prune_dataset

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
TRAIN_TIME_BUDGET = 240       # seconds (4 min training, ~1 min for pruning + eval)
DEVICE_BATCH_SIZE = 16
TOTAL_BATCH_SIZE = 64         # gradient accumulation
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 100
EVAL_INTERVAL = 50            # evaluate every N steps

# Model architecture
DEPTH = 6                     # number of transformer layers
D_MODEL = 384                 # embedding dimension
N_HEADS = 6                   # attention heads
DROPOUT = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.json")


# ---------------------------------------------------------------------------
# Minimal GPT Model
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)  # (B, nh, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        hidden = 4 * d_model
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.silu(self.gate(x)) * self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, depth, max_seq_len, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len, dropout)
            for _ in range(depth)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        self.max_seq_len = max_seq_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train():
    print("=" * 60)
    print("AUTO DATA PRUNING EXPERIMENT")
    print("=" * 60)

    # Step 1: Prepare data (one-time)
    download_and_tokenize()
    tokenizer = load_tokenizer()

    # Step 2: Load and prune training data
    print("\n--- Data Pruning Phase ---")
    prune_start = time.time()
    full_train_ds = load_hf_dataset(split="train")
    pruned_ds = prune_dataset(full_train_ds)
    prune_elapsed = time.time() - prune_start
    print(f"Pruning took {prune_elapsed:.1f}s")

    # Step 3: Tokenize the pruned dataset
    print("\n--- Tokenizing pruned data ---")
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    all_tokens = []
    for example in pruned_ds:
        encoded = tokenizer.encode(example["text"])
        all_tokens.extend(encoded.ids)
        all_tokens.append(eot_id)
    print(f"Pruned data: {len(all_tokens):,} tokens")

    # Step 4: Create datasets and dataloaders
    train_dataset = PrunedTokenDataset(all_tokens, seq_len=MAX_SEQ_LEN)
    val_dataset = TokenDataset(VAL_SHARD, seq_len=MAX_SEQ_LEN)

    if len(train_dataset) == 0:
        print("ERROR: No training sequences after pruning. Pruning too aggressive.")
        save_results(float("inf"), 0, 0, len(pruned_ds))
        return

    grad_accum_steps = max(1, TOTAL_BATCH_SIZE // DEVICE_BATCH_SIZE)
    train_loader = make_dataloader(train_dataset, DEVICE_BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(val_dataset, DEVICE_BATCH_SIZE, shuffle=False)

    print(f"Train sequences: {len(train_dataset):,}")
    print(f"Val sequences:   {len(val_dataset):,}")
    print(f"Grad accum steps: {grad_accum_steps}")

    # Step 5: Build model
    print("\n--- Building model ---")
    model = GPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        depth=DEPTH,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Step 6: Optimizer with warmup + cosine decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # Step 7: Training loop with fixed time budget
    print(f"\n--- Training (budget: {TRAIN_TIME_BUDGET}s) ---")
    model.train()
    train_start = time.time()
    step = 0
    best_val_bpb = float("inf")
    epoch = 0
    tokens_seen = 0

    while True:
        epoch += 1
        for x, y in train_loader:
            # Check time budget
            elapsed = time.time() - train_start
            if elapsed >= TRAIN_TIME_BUDGET:
                break

            x, y = x.to(DEVICE), y.to(DEVICE)

            # Learning rate schedule: warmup + cosine decay
            if step < WARMUP_STEPS:
                lr = LEARNING_RATE * (step + 1) / WARMUP_STEPS
            else:
                progress = (step - WARMUP_STEPS) / max(1, 2000 - WARMUP_STEPS)
                lr = LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward pass
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss = loss / grad_accum_steps
            loss.backward()

            tokens_seen += x.numel()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Evaluation
            if step > 0 and step % EVAL_INTERVAL == 0:
                val_bpb = evaluate_bpb(model, val_loader, DEVICE, VOCAB_SIZE)
                best_val_bpb = min(best_val_bpb, val_bpb)
                elapsed = time.time() - train_start
                print(
                    f"  step {step:5d} | "
                    f"train_loss {loss.item() * grad_accum_steps:.4f} | "
                    f"val_bpb {val_bpb:.4f} | "
                    f"best_bpb {best_val_bpb:.4f} | "
                    f"lr {lr:.2e} | "
                    f"tokens {tokens_seen:,} | "
                    f"time {elapsed:.0f}s"
                )

            step += 1

        # Check time budget after epoch
        if time.time() - train_start >= TRAIN_TIME_BUDGET:
            break

    # Final evaluation
    print("\n--- Final Evaluation ---")
    final_val_bpb = evaluate_bpb(model, val_loader, DEVICE, VOCAB_SIZE)
    best_val_bpb = min(best_val_bpb, final_val_bpb)
    total_time = time.time() - train_start

    print(f"Final val_bpb:  {final_val_bpb:.4f}")
    print(f"Best val_bpb:   {best_val_bpb:.4f}")
    print(f"Total steps:    {step}")
    print(f"Epochs:         {epoch}")
    print(f"Tokens seen:    {tokens_seen:,}")
    print(f"Training time:  {total_time:.1f}s")
    print(f"Pruned data:    {len(pruned_ds):,} examples")

    save_results(best_val_bpb, step, epoch, len(pruned_ds))


def save_results(val_bpb, steps, epochs, pruned_size):
    """Save results to JSON for the agent to read."""
    result = {
        "val_bpb": val_bpb,
        "steps": steps,
        "epochs": epochs,
        "pruned_dataset_size": pruned_size,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Append to results history
    history = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            history = json.load(f)

    history.append(result)

    with open(RESULTS_FILE, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"RESULT: val_bpb = {val_bpb:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train()
