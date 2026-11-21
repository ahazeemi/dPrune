import math
import numpy as np
import pytest
from datasets import Dataset

from dprune import D2Pruner, TopKPruner


def _clustered_dataset():
    """Two tight clusters: a large, hard one and a small, easy one."""
    rng = np.random.default_rng(0)
    hard = rng.normal(loc=[10.0, 0.0], scale=0.01, size=(10, 2))
    easy = rng.normal(loc=[0.0, 10.0], scale=0.01, size=(5, 2))
    emb = np.vstack([hard, easy])
    scores = [1.0 + 0.01 * i for i in range(10)] + [0.5] * 5
    ds = Dataset.from_dict({
        "text": [f"t{i}" for i in range(15)],
        "score": scores,
        "embedding": emb.tolist(),
    })
    return ds


def test_d2_pruner_size_and_column():
    ds = _clustered_dataset()
    pruned = D2Pruner(k=6, n_neighbors=5, normalize_embeddings=False).prune(ds)
    assert len(pruned) == math.ceil(len(ds) * 0.4)
    assert len(set(pruned["text"])) == 6


def test_d2_pruner_covers_both_clusters():
    ds = _clustered_dataset()
    # Top-k on difficulty alone only picks from the hard cluster.
    topk = TopKPruner(k=3).prune(ds)
    assert all(int(t[1:]) < 10 for t in topk["text"])
    # D2 down-weights near-duplicates, so the easy cluster gets represented.
    pruned = D2Pruner(
        k=3, n_neighbors=9, normalize_embeddings=False, normalize_scores=False
    ).prune(ds)
    assert any(int(t[1:]) >= 10 for t in pruned["text"])


def test_d2_pruner_embeddings_kwarg_and_fraction():
    ds = _clustered_dataset()
    emb = np.asarray(ds["embedding"])
    ds = ds.remove_columns("embedding")
    pruned = D2Pruner(k=0.4).prune(ds, embeddings=emb)
    assert len(pruned) == math.ceil(len(ds) * 0.4)


def test_d2_pruner_hard_cutoff_drops_hardest():
    ds = _clustered_dataset()
    pruned = D2Pruner(k=8, hard_cutoff=0.2, normalize_embeddings=False).prune(ds)
    assert len(pruned) == 8
    assert max(pruned["score"]) < max(ds["score"])


def test_d2_pruner_edge_cases():
    ds = _clustered_dataset()
    assert len(D2Pruner(k=100).prune(ds)) == len(ds)
    assert len(D2Pruner(k=0).prune(ds)) == 0
    with pytest.raises(ValueError):
        D2Pruner(k=1.5).prune(ds)
    with pytest.raises(ValueError):
        D2Pruner(k=3).prune(ds.remove_columns("embedding"))
    with pytest.raises(ValueError):
        D2Pruner(k=3, n_neighbors=0)
    with pytest.raises(ValueError):
        D2Pruner(k=3).prune(ds, embeddings=np.zeros((3, 2)))
