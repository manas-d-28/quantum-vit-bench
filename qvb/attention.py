"""Attention modules. Each maps (B, T, E) -> (B, T, E) and is plugged into the
same backbone. Every module splits the embedding into `n_heads` slices of
d = E / n_heads features. Each head only sees its own slice, as in the quantum
papers, where one d-qubit circuit handles one slice.

    classic   standard scaled dot-product attention (per-head linear Q, K, V)
    gaussian  classical mimic of QSANN / Unlu: scalar q, k per token,
              score_ij = -(q_i - k_j)^2 / sqrt(d)
    unlu      Unlu et al. (2024): q, k, v produced by d-qubit circuits,
              same Gaussian-style score as `gaussian`
"""
import math
import torch
import torch.nn as nn


class _MultiHead(nn.Module):
    """Splits (B, T, E) into heads, runs `head` on each, concatenates."""

    def __init__(self, embed_dim, n_heads, make_head):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.d = embed_dim // n_heads
        self.heads = nn.ModuleList([make_head(self.d) for _ in range(n_heads)])

    def forward(self, x):
        return torch.cat([h(x[..., i * self.d:(i + 1) * self.d])
                          for i, h in enumerate(self.heads)], dim=-1)


def _attend(scores, v, d):
    return torch.softmax(scores / math.sqrt(d), dim=-1) @ v


class ClassicHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.q, self.k, self.v = (nn.Linear(d, d, bias=False) for _ in range(3))

    def forward(self, x):                               # x: (B, T, d)
        q, k, v = self.q(x), self.k(x), self.v(x)
        return _attend(q @ k.transpose(1, 2), v, self.d)


class GaussianHead(nn.Module):
    """Classical mimic: same scoring rule as the quantum head, linear maps instead of circuits."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.q = nn.Linear(d, 1, bias=False)
        self.k = nn.Linear(d, 1, bias=False)
        self.v = nn.Linear(d, d, bias=False)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)        # (B,T,1), (B,T,1), (B,T,d)
        scores = -(q - k.transpose(1, 2)) ** 2           # (B, T, T)
        return _attend(scores, v, self.d)


class UnluHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        from .quantum import QLayer, unlu_query_key, unlu_value
        self.d = d
        self.q = QLayer(unlu_query_key, d)
        self.k = QLayer(unlu_query_key, d)
        self.v = QLayer(unlu_value, d)

    def forward(self, x):
        b, t, d = x.shape
        flat = x.reshape(b * t, d)                        # one circuit run per token
        q = self.q(flat).reshape(b, t, 1)
        k = self.k(flat).reshape(b, t, 1)
        v = self.v(flat).reshape(b, t, d)
        scores = -(q - k.transpose(1, 2)) ** 2
        return _attend(scores, v.to(x.dtype), d)


HEADS = {"classic": ClassicHead, "gaussian": GaussianHead, "unlu": UnluHead}
QUANTUM = {"unlu": {"circuits_per_token_per_head": 3}}   # q, k and v circuits


def make_attention(name, embed_dim, n_heads):
    head = HEADS[name]
    return lambda: _MultiHead(embed_dim, n_heads, head)
