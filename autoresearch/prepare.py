"""
Data preparation for auto data pruning experiments.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VOCAB_SIZE = 8192
MAX_SEQ_LEN = 512

HF_DATASET = "roneneldan/TinyStories"
HF_DATASET_TEXT_COLUMN = "text"
HF_DATASET_SPLIT_TRAIN = "train"
HF_DATASET_SPLIT_VAL = "validation"

TRAIN_SHARD = os.path.join(DATA_DIR, "train.bin")
VAL_SHARD = os.path.join(DATA_DIR, "val.bin")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
TOKENIZATION_BATCH_SIZE = 1024


def train_tokenizer(texts, vocab_size=VOCAB_SIZE, save_path=TOKENIZER_PATH):
    """Train a BPE tokenizer on the given texts."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path} (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


def load_tokenizer(path=TOKENIZER_PATH):
    """Load a pre-trained tokenizer."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def _batched(iterable, batch_size):
    """Yield lists of up to batch_size items from an iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_dataset_texts(dataset, text_column=HF_DATASET_TEXT_COLUMN):
    """Yield texts from a Hugging Face dataset without materializing the full column."""
    for example in dataset:
        yield example[text_column]


def write_tokenized_shard(tokenizer, texts, shard_path, batch_size=TOKENIZATION_BATCH_SIZE):
    """Stream-tokenize texts into a uint16 shard."""
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    total_tokens = 0

    with open(shard_path, "wb") as f:
        for text_batch in _batched(texts, batch_size):
            encoded_batch = tokenizer.encode_batch(text_batch)
            batch_tokens = []
            for encoded in encoded_batch:
                batch_tokens.extend(encoded.ids)
                batch_tokens.append(eot_id)

            arr = np.asarray(batch_tokens, dtype=np.uint16)
            arr.tofile(f)
            total_tokens += len(arr)

    return total_tokens


def download_and_tokenize():
    """Download the dataset, train the tokenizer, and write token shards."""
    from datasets import load_dataset

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(TRAIN_SHARD) and os.path.exists(VAL_SHARD):
        print("Data shards already exist, skipping download.")
        return

    print(f"Downloading {HF_DATASET}...")
    ds = load_dataset(HF_DATASET)

    print("Training tokenizer...")
    train_split = ds[HF_DATASET_SPLIT_TRAIN]
    sample_size = min(100_000, len(train_split))
    sample_texts = (
        train_split[i][HF_DATASET_TEXT_COLUMN]
        for i in range(sample_size)
    )
    tokenizer = train_tokenizer(sample_texts)

    for split_name, shard_path in [
        (HF_DATASET_SPLIT_TRAIN, TRAIN_SHARD),
        (HF_DATASET_SPLIT_VAL, VAL_SHARD),
    ]:
        print(f"Tokenizing {split_name}...")
        split = ds[split_name]
        n_tokens = write_tokenized_shard(
            tokenizer,
            iter_dataset_texts(split),
            shard_path,
        )
        print(f"  Wrote {n_tokens:,} tokens to {shard_path}")

    print("Data preparation complete.")


def load_hf_dataset(split="train", max_examples=None):
    """Load the raw Hugging Face dataset."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    return ds


class TokenDataset(TorchDataset):
    """Memory-mapped dataset of tokenized sequences from a binary shard."""

    def __init__(self, shard_path, seq_len=MAX_SEQ_LEN):
        self.data = np.memmap(shard_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n_sequences = len(self.data) // (seq_len + 1)

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def make_dataloader(dataset, batch_size, shuffle=True):
    """Create a DataLoader for a token dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
    )


@torch.no_grad()
def evaluate_bpb(model, val_loader, device, vocab_size=VOCAB_SIZE):
    """
    Evaluate validation bits-per-byte (val_bpb).
    Lower is better.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size), y.view(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens  # nats per token

    # Convert nats → bits, then normalize by bytes-per-token.
    # For BPE with vocab_size ~8K on English text, ~3.5 chars/token ≈ 3.5 bytes/token.
    # We estimate bytes_per_token from the tokenizer if available; else use 3.5.
    bytes_per_token = 3.5  # reasonable default
    bpb = avg_loss / np.log(2) / bytes_per_token

    model.train()
    return bpb


# ---------------------------------------------------------------------------
# Main: run once to prepare data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    download_and_tokenize()
    print("\nVerification:")
    tokenizer = load_tokenizer()
    print(f"  Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    train_ds = TokenDataset(TRAIN_SHARD)
    val_ds = TokenDataset(VAL_SHARD)
    print(f"  Train sequences: {len(train_ds):,}")
    print(f"  Val sequences:   {len(val_ds):,}")
    print("Done. Ready for training.")
