import math

import numpy as np
import pytest
import torch
from datasets import Dataset
from transformers import GPT2Config, GPT2LMHeadModel

from dprune import IFDScorer, PruningPipeline, ThresholdPruner, TopKPruner


class CharTokenizer:
    """Minimal byte-level tokenizer so the tests need no downloads."""

    def __init__(self, bos=True):
        self.bos_token_id = 1 if bos else None
        self.eos_token_id = 2
        self.pad_token_id = 0

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[3 + (ord(c) % 120) for c in t] for t in texts]}


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=128, n_positions=256, n_embd=32, n_layer=2, n_head=2)
    return GPT2LMHeadModel(config).eval()


def _dataset():
    return Dataset.from_dict({
        "instruction": ["Say hi.", "Add 2 and 3.", "Name a color.", "Translate 'cat' to French."],
        "input": ["", "", "", ""],
        "output": ["hi", "5", "Blue is a color.", "chat"],
    })


def _manual_ifd(model, tok, prompt, response):
    """Reference implementation, one example at a time with no padding."""
    p = tok([prompt])["input_ids"][0]
    r = tok([response])["input_ids"][0]
    bos = [tok.bos_token_id]

    def mean_nll(seq, start):
        ids = torch.tensor([seq])
        with torch.no_grad():
            logp = torch.log_softmax(model(input_ids=ids).logits[0], dim=-1)
        return -np.mean([logp[t - 1, seq[t]].item() for t in range(start, len(seq))])

    cond = mean_nll(bos + p + r, 1 + len(p))
    direct = mean_nll(bos + r, 1)
    return math.exp(cond - direct)


def test_ifd_adds_score_column(model):
    ds = _dataset()
    scored = IFDScorer(model, CharTokenizer(), batch_size=2).score(ds)
    assert "score" in scored.column_names
    assert len(scored) == len(ds)
    assert all(np.isfinite(s) and s > 0 for s in scored["score"])


def test_ifd_matches_reference(model):
    ds = _dataset()
    tok = CharTokenizer()
    template = "Q: {instruction}\nA:"
    scored = IFDScorer(model, tok, prompt_template=template, batch_size=4).score(ds)
    for ex, s in zip(ds, scored["score"]):
        expected = _manual_ifd(model, tok, template.format(instruction=ex["instruction"]), ex["output"])
        assert s == pytest.approx(expected, rel=1e-4)


def test_ifd_independent_of_batch_size(model):
    ds = _dataset()
    a = IFDScorer(model, CharTokenizer(), batch_size=1).score(ds)["score"]
    b = IFDScorer(model, CharTokenizer(), batch_size=4).score(ds)["score"]
    np.testing.assert_allclose(a, b, rtol=1e-4)


def test_ifd_empty_prompt_is_one(model):
    ds = _dataset()
    scored = IFDScorer(model, CharTokenizer(), prompt_template=lambda ex: "").score(ds)
    np.testing.assert_allclose(scored["score"], 1.0, rtol=1e-5)


def test_ifd_components_and_ratio(model):
    ds = _dataset()
    scored = IFDScorer(model, CharTokenizer(), return_components=True).score(ds)
    ratio = np.array(scored["ppl_conditioned"]) / np.array(scored["ppl_direct"])
    np.testing.assert_allclose(scored["score"], ratio, rtol=1e-5)


def test_ifd_without_bos_skips_first_token(model):
    ds = Dataset.from_dict({"instruction": ["Say hi.", "Count."], "output": ["x", "123"]})
    scored = IFDScorer(model, CharTokenizer(bos=False), input_column=None).score(ds)
    # A one-token response has nothing to score without a BOS token.
    assert math.isnan(scored["score"][0])
    assert np.isfinite(scored["score"][1])


def test_ifd_truncation_respects_max_length(model):
    ds = Dataset.from_dict({"instruction": ["word " * 100], "output": ["answer " * 100]})
    scored = IFDScorer(model, CharTokenizer(), max_length=64).score(ds)
    assert np.isfinite(scored["score"][0])


def test_ifd_input_column_used_in_default_template(model):
    tok = CharTokenizer()
    with_input = Dataset.from_dict({"instruction": ["Sum."], "input": ["2 3"], "output": ["5"]})
    no_input = Dataset.from_dict({"instruction": ["Sum."], "input": [""], "output": ["5"]})
    a = IFDScorer(model, tok).score(with_input)["score"][0]
    b = IFDScorer(model, tok).score(no_input)["score"][0]
    assert a != pytest.approx(b)


def test_ifd_missing_column_raises(model):
    ds = Dataset.from_dict({"prompt": ["x"], "output": ["y"]})
    with pytest.raises(ValueError):
        IFDScorer(model, CharTokenizer()).score(ds)


def test_ifd_invalid_args(model):
    with pytest.raises(ValueError):
        IFDScorer(model, CharTokenizer(), max_length=1)
    with pytest.raises(ValueError):
        IFDScorer(model, CharTokenizer(), batch_size=0)


def test_ifd_cherry_selection_pipeline(model):
    ds = _dataset()
    scored = IFDScorer(model, CharTokenizer()).score(ds)
    valid = ThresholdPruner(threshold=1.0, keep="below").prune(scored)
    assert all(s < 1.0 for s in valid["score"])
    selected = PruningPipeline(scorer=IFDScorer(model, CharTokenizer()), pruner=TopKPruner(k=2)).run(ds)
    assert len(selected) == 2
    assert min(selected["score"]) >= sorted(scored["score"])[-2] - 1e-6
