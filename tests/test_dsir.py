import numpy as np
import pytest
from datasets import Dataset

from dprune import DSIRScorer, ImportanceResamplingPruner, PruningPipeline, TopKPruner


TARGET = [
    "the patient was given a dose of the drug",
    "clinical trial results show the drug reduces symptoms",
    "the doctor prescribed medication to the patient",
]


@pytest.fixture
def raw_dataset():
    texts = [
        "the patient received the drug in a clinical trial",   # in-domain
        "the striker scored a goal in the final minute",       # sports
        "doctor says the medication helps the patient",         # in-domain
        "the stock market fell sharply on monday",              # finance
        "the team won the match after extra time",              # sports
        "a new drug trial for patient symptoms",                # in-domain
    ]
    return Dataset.from_dict({"text": texts, "id": list(range(len(texts)))})


def test_dsir_scores_target_like_examples_higher(raw_dataset):
    scored = DSIRScorer(target_texts=TARGET, text_column="text").score(raw_dataset)
    assert "score" in scored.column_names
    scores = np.array(scored["score"])
    in_domain = scores[[0, 2, 5]]
    out_domain = scores[[1, 3, 4]]
    assert in_domain.min() > out_domain.max()


def test_dsir_pipeline_topk_selects_in_domain(raw_dataset):
    scorer = DSIRScorer(target_texts=TARGET, text_column="text", length_normalize=True)
    pruned = PruningPipeline(scorer=scorer, pruner=TopKPruner(k=3)).run(raw_dataset)
    assert sorted(pruned["id"]) == [0, 2, 5]


def test_dsir_with_external_raw_texts(raw_dataset):
    scorer = DSIRScorer(
        target_texts=TARGET, text_column="text", raw_texts=raw_dataset["text"]
    )
    a = np.array(scorer.score(raw_dataset)["score"])
    b = np.array(DSIRScorer(target_texts=TARGET, text_column="text").score(raw_dataset)["score"])
    np.testing.assert_allclose(a, b)


def test_dsir_is_deterministic(raw_dataset):
    s1 = DSIRScorer(TARGET, "text", num_buckets=97).score(raw_dataset)["score"]
    s2 = DSIRScorer(TARGET, "text", num_buckets=97).score(raw_dataset)["score"]
    assert s1 == s2


def test_dsir_validation(raw_dataset):
    with pytest.raises(ValueError):
        DSIRScorer(target_texts=[], text_column="text")
    with pytest.raises(ValueError):
        DSIRScorer(target_texts=TARGET, text_column="text", num_buckets=0)
    with pytest.raises(ValueError):
        DSIRScorer(target_texts=TARGET, text_column="text", smoothing=0)
    with pytest.raises(ValueError):
        DSIRScorer(target_texts=TARGET, text_column="missing").score(raw_dataset)


def test_importance_resampling_size_and_determinism():
    ds = Dataset.from_dict({"score": [float(i) for i in range(100)], "id": list(range(100))})
    p = ImportanceResamplingPruner(k=0.2, seed=0)
    out1, out2 = p.prune(ds), p.prune(ds)
    assert len(out1) == 20
    assert out1["id"] == out2["id"]
    assert len(set(out1["id"])) == 20


def test_importance_resampling_low_temperature_is_topk():
    ds = Dataset.from_dict({"score": [float(i) for i in range(50)], "id": list(range(50))})
    out = ImportanceResamplingPruner(k=5, temperature=1e-6).prune(ds)
    assert sorted(out["id"]) == [45, 46, 47, 48, 49]


def test_importance_resampling_favours_high_weights():
    # Half the examples have log-weight 5, the rest 0 -> ~e^5 times more likely.
    scores = [5.0] * 50 + [0.0] * 50
    ds = Dataset.from_dict({"score": scores, "id": list(range(100))})
    out = ImportanceResamplingPruner(k=20, seed=1).prune(ds)
    assert sum(i < 50 for i in out["id"]) >= 18


def test_importance_resampling_validation():
    ds = Dataset.from_dict({"score": [1.0, 2.0]})
    with pytest.raises(ValueError):
        ImportanceResamplingPruner(k=1, temperature=0)
    with pytest.raises(ValueError):
        ImportanceResamplingPruner(k=1.5).prune(ds)
    assert len(ImportanceResamplingPruner(k=5).prune(ds)) == 2
