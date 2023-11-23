from typing import Optional, Tuple

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM

from .cache_regimes import KVCache


def evalute_on_text(
    model: GPTJForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    kv_cache: Optional[KVCache] = None,
    max_ctx_len: int = 2048,
) -> torch.Tensor:
    """Evaluate the model on a text sequence.

    Args:
        model (GPTJForCausalLM): The model to evaluate.
        tokenizer (AutoTokenizer): The tokenizer to use.
        text (str): The text to evaluate on.
        kv_cache (Optional[KVCache]): The cache to use for the evaluation.
        max_ctx_len (int): The maximum context length for the model.

    Returns:
        torch.Tensor: The negative log likelihoods for each token in the text.
    """
    loss_fn = CrossEntropyLoss(reduction="none")
    encodings = tokenizer(text, return_tensors="pt")
    use_cache = kv_cache is not None

    seq_len = encodings.input_ids.size(1)
    past_key_values: Tuple[torch.Tensor] = None
    pbar = tqdm(range(0, seq_len - 1))
    nlls = []

    for idx in pbar:
        input_ids = []
        if use_cache:
            input_ids = encodings.input_ids[:, idx : idx + 1].to(model.device)
        else:
            input_ids = encodings.input_ids[:, 0 : idx + 1].to(model.device)

            if input_ids.size(1) >= max_ctx_len:
                input_ids = input_ids[:, -max_ctx_len:]

        with torch.no_grad():
            outputs = model(
                input_ids, past_key_values=past_key_values, use_cache=use_cache
            )

            logits = outputs.logits[:, -1, :].view(-1, model.config.vocab_size)
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)

            nll = loss_fn(logits, label)

            if use_cache:
                past_key_values = kv_cache.prune_kv_cache(outputs.past_key_values)

        nlls.append(nll)
        pbar.set_description(f"nll: {nll.item():.2f}, ppl: {torch.exp(nll).item():.2f}")

    return torch.stack(nlls)
