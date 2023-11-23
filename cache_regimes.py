from abc import ABC, abstractmethod
from typing import Tuple

import torch


class KVCache(ABC):
    @abstractmethod
    def prune_kv_cache(self, past_key_values: Tuple[torch.Tensor]):
        pass


class SlidingWindowCache(KVCache):
    def __init__(self, cache_size: int, k_seq_dim: int = 2):
        self.cache_size = cache_size
        self.k_seq_dim = k_seq_dim

    def prune_kv_cache(self, past_key_values: Tuple[torch.Tensor]):
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        if seq_len <= self.cache_size:
            return past_key_values

        return [
            [
                k[:, :, seq_len - self.cache_size : seq_len, :],
                v[:, :, seq_len - self.cache_size : seq_len, :],
            ]
            for k, v in past_key_values
        ]


class AttentionSinkCache(KVCache):
    def __init__(
        self, start_size: int, recent_size: int, k_seq_dim: int = 1, v_seq_dim: int = 2
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim

    def prune_kv_cache(self, past_key_values: Tuple[torch.Tensor]):
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        if seq_len <= self.recent_size + self.start_size:
            return past_key_values

        return [
            [
                torch.cat(
                    [
                        k[:, 0 : self.start_size, :, :],
                        k[:, seq_len - self.recent_size : seq_len, :, :],
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        v[:, :, 0 : self.start_size, :],
                        v[:, :, seq_len - self.recent_size : seq_len, :],
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
