from typing import List, Optional

import matplotlib.pyplot as plt
import torch


def graph_sliding_window_ppl(
    nlls: torch.Tensor, cache_regime: str, window_size: int, max_ppl: int = 500
):
    n = nlls.size(0)
    window_ppls = []

    for idx in range(0, n, window_size):
        window_nlls = nlls[idx : idx + window_size].to("cpu")
        window_ppl = min(torch.exp(window_nlls.mean()), max_ppl)
        window_ppls.append(window_ppl)

    x = torch.arange(len(window_ppls)) * window_size

    plt.figure(figsize=(20, 10))
    plt.title(cache_regime, fontsize=18)

    plt.xlabel("Token")
    plt.xticks(x[::10], fontsize=12)

    plt.ylabel(f"{window_size}-token average perplexity")
    plt.yticks(fontsize=12)

    plt.plot(x, window_ppls)
    plt.show()


def graph_attentions(
    attn: torch.Tensor, n_cols: int = 7, head_idxs: Optional[List[int]] = None
):
    n_layers = attn.size(0)
    if head_idxs is None:
        head_idxs = torch.arange(n_layers)

    n_rows = len(head_idxs) // n_cols

    _, axs = plt.subplots(n_rows, n_cols, figsize=(20, 20))

    graph_idx = 0
    for idx in head_idxs:
        row_idx = graph_idx // n_cols
        col_idx = graph_idx % n_cols

        ax = axs[row_idx, col_idx]
        graph_idx += 1

        ax.set_title(f"Layer {idx + 1}")
        head = attn[idx]
        ax.imshow(head, cmap="viridis")

    plt.show()
