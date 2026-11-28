"""
Scorers for selecting instruction-tuning data for (large) language models.
"""

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..base import Scorer


ALPACA_PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}"
    "\n\n### Response:"
)
ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{instruction}"
    "\n\n### Response:"
)


def _mean_response_nll(
    model: PreTrainedModel,
    sequences: List[List[int]],
    response_starts: List[int],
    pad_token_id: int,
    device: torch.device,
) -> np.ndarray:
    """
    Mean negative log-likelihood of the tokens ``seq[start:]`` of each sequence,
    where token ``t`` is predicted from ``seq[:t]``. Sequences are right-padded,
    so the result does not depend on how examples are batched.
    """
    max_len = max(len(s) for s in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    labels = torch.full((len(sequences), max_len), -100, dtype=torch.long)
    for i, (seq, start) in enumerate(zip(sequences, response_starts)):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, : len(seq)] = 1
        labels[i, start : len(seq)] = torch.tensor(seq[start:], dtype=torch.long)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].float()
    targets = labels[:, 1:]
    token_nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(targets.shape)
    mask = (targets != -100).float()
    counts = mask.sum(dim=1)
    mean_nll = (token_nll * mask).sum(dim=1) / counts.clamp(min=1.0)
    mean_nll[counts == 0] = float("nan")
    return mean_nll.cpu().numpy()


class IFDScorer(Scorer):
    """
    Instruction-Following Difficulty (IFD) scorer from Li et al., NAACL 2024,
    "From Quantity to Quality: Boosting LLM Performance with Self-Guided Data
    Selection for Instruction Tuning" (https://arxiv.org/abs/2308.12032).

    For an instruction/response pair (Q, A), a causal language model scores the
    response twice: once conditioned on the prompt built from Q, and once on its
    own. The IFD score is the ratio

        IFD(Q, A) = PPL(A | Q) / PPL(A)

    where PPL is the perplexity over the response tokens only. A low score means
    the instruction makes the response much easier to predict, so the model
    already follows it well. A high score means the instruction helps little and
    the example is hard (informative) to learn from. A score >= 1 means the
    instruction does not help at all, which usually signals a misaligned or
    noisy pair.

    The paper selects the examples with the highest IFD scores (e.g. the top
    5-10% of Alpaca/WizardLM) after discarding those with IFD >= 1::

        scored = IFDScorer(model, tokenizer).score(dataset)
        valid = ThresholdPruner(threshold=1.0, keep="below").prune(scored)
        selected = TopKPruner(k=0.1).prune(valid)

    Any causal LM works. Using a small model such as GPT-2 (as in
    "Superfiltering", Li et al., ACL 2024, https://arxiv.org/abs/2402.00530)
    gives scores that are consistent with those of much larger models at a
    fraction of the cost.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        instruction_column: str = "instruction",
        response_column: str = "output",
        input_column: Optional[str] = "input",
        prompt_template: Optional[Union[str, Callable[[Dict], str]]] = None,
        max_length: int = 512,
        batch_size: int = 8,
        add_bos_token: bool = True,
        return_components: bool = False,
    ):
        """
        Args:
            model (PreTrainedModel): A causal language model (e.g.
                ``AutoModelForCausalLM``). Ideally the base model you are about
                to instruction-tune, or a small proxy model such as GPT-2.
            tokenizer (PreTrainedTokenizer): The model's tokenizer.
            instruction_column (str): Column holding the instruction.
            response_column (str): Column holding the target response.
            input_column (Optional[str]): Column holding the optional extra
                input (as in Alpaca). Ignored if missing from the dataset or None.
            prompt_template (Optional[Union[str, Callable]]): How to build the
                prompt Q. Either a format string using ``{instruction}`` and
                optionally ``{input}``, or a callable mapping an example (dict)
                to the prompt string. Defaults to the Alpaca template used in
                the paper.
            max_length (int): Maximum number of tokens of the conditioned
                sequence. The response is truncated first if needed and the
                prompt is truncated from the left, so both scores use exactly
                the same response tokens.
            batch_size (int): Batch size for scoring.
            add_bos_token (bool): Prepend the tokenizer's BOS token (if it has
                one) so that the first response token can be scored without a
                prompt. If no BOS token is used, the first response token is
                excluded from both perplexities.
            return_components (bool): If True, also add ``ppl_conditioned`` and
                ``ppl_direct`` columns.
        """
        if not isinstance(max_length, int) or max_length < 2:
            raise ValueError("max_length must be an integer >= 2")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if prompt_template is not None and not (
            isinstance(prompt_template, str) or callable(prompt_template)
        ):
            raise ValueError("prompt_template must be a format string or a callable")
        self.model = model
        self.tokenizer = tokenizer
        self.instruction_column = instruction_column
        self.response_column = response_column
        self.input_column = input_column
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.batch_size = batch_size
        self.add_bos_token = add_bos_token
        self.return_components = return_components

    def _build_prompt(self, example: Dict) -> str:
        if callable(self.prompt_template):
            return self.prompt_template(example)
        instruction = example[self.instruction_column]
        extra = ""
        if self.input_column is not None and self.input_column in example:
            extra = example[self.input_column] or ""
        if isinstance(self.prompt_template, str):
            return self.prompt_template.format(instruction=instruction, input=extra)
        if extra.strip():
            return ALPACA_PROMPT_INPUT.format(instruction=instruction, input=extra)
        return ALPACA_PROMPT_NO_INPUT.format(instruction=instruction)

    def _encode(self, texts: List[str]) -> List[List[int]]:
        encoded = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [list(ids) for ids in encoded]

    def _pad_token_id(self) -> int:
        for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
            value = getattr(self.tokenizer, attr, None)
            if value is not None:
                return int(value)
        return 0

    def _device(self) -> torch.device:
        if getattr(self.model, "hf_device_map", None):
            # Model was dispatched across devices (e.g. device_map="auto").
            return next(self.model.parameters()).device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        return device

    def score(self, dataset: Dataset, **kwargs) -> Dataset:
        """
        Adds a ``score`` column with the IFD score of each example. Examples
        whose response has no scorable token get a score of NaN.
        """
        required = [self.response_column]
        if not callable(self.prompt_template):
            required.append(self.instruction_column)
        missing = [c for c in required if c not in dataset.column_names]
        if missing:
            raise ValueError(f"Dataset is missing column(s): {missing}")

        self.model.eval()
        device = self._device()
        pad_id = self._pad_token_id()
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        bos = [int(bos_id)] if self.add_bos_token and bos_id is not None else []
        # Without a BOS token the first response token cannot be predicted in
        # the direct case, so it is skipped in both cases.
        skip = 0 if bos else 1

        cond_nll, direct_nll = [], []
        for start in tqdm(range(0, len(dataset), self.batch_size), desc="Scoring IFD"):
            batch = dataset[start : start + self.batch_size]
            n = len(batch[self.response_column])
            examples = [{k: v[i] for k, v in batch.items()} for i in range(n)]
            prompts = self._encode([self._build_prompt(ex) for ex in examples])
            responses = self._encode(
                [str(ex[self.response_column]) for ex in examples]
            )

            cond_seqs, cond_starts, dir_seqs, dir_starts = [], [], [], []
            for prompt, response in zip(prompts, responses):
                response = response[: self.max_length - len(bos)]
                budget = self.max_length - len(bos) - len(response)
                prompt = prompt[max(0, len(prompt) - budget) :] if budget > 0 else []
                cond_seqs.append(bos + prompt + response)
                cond_starts.append(len(bos) + len(prompt) + skip)
                dir_seqs.append(bos + response)
                dir_starts.append(len(bos) + skip)

            cond_nll.append(
                _mean_response_nll(self.model, cond_seqs, cond_starts, pad_id, device)
            )
            direct_nll.append(
                _mean_response_nll(self.model, dir_seqs, dir_starts, pad_id, device)
            )

        cond_nll = np.concatenate(cond_nll) if cond_nll else np.array([])
        direct_nll = np.concatenate(direct_nll) if direct_nll else np.array([])
        ppl_cond = np.exp(cond_nll)
        ppl_direct = np.exp(direct_nll)
        ifd = np.exp(cond_nll - direct_nll)

        scored = dataset.add_column("score", [float(x) for x in ifd])
        if self.return_components:
            scored = scored.add_column("ppl_conditioned", [float(x) for x in ppl_cond])
            scored = scored.add_column("ppl_direct", [float(x) for x in ppl_direct])
        return scored
