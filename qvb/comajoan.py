"""Comajoan Cara et al. (2024), arXiv:2405.10284: quantum ViT where the Q, K, V and
output projections, and a layer inside the MLP, are variational circuits.

Their circuit (official code: github.com/ML4SCI/QMLHEP, Quantum_Transformers_*/
quantum_transformers/quantum_layer.py), on n qubits for an n-feature vector x:
    Rx(x_j) on every qubit                        (angle embedding)
    Rx(w_j) on every qubit, then a CNOT ring      (one BasicEntangler layer, their default)
    read <Z_i> on every qubit                     -> n outputs

Key observation: before the CNOTs the state is a product state with
<Z_j> = cos(x_j + w_j), and CNOTs only map Z-strings to Z-strings. So every output is
exactly a product of cosines:

    <Z_i> = prod_{j in S_i} cos(x_j + w_j)

where S_i is found by pushing Z_i back through the CNOT ring. This module computes that
formula directly, so the paper's 16- and 32-qubit layers cost no more than a classical
layer. `tools/test_circuits.py` checks the formula against a real TensorCircuit
simulation of the same circuit.
"""
import torch
import torch.nn as nn


def cnot_ring(n):
    """The CNOT pattern of their basic_vqc."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    return [(j, (j + 1) % n) for j in range(n)]


def z_support(n):
    """S_i for every output i: the qubits whose cosines multiply into <Z_i>.
    Heisenberg picture: walk the gates backwards; CNOT(c, t) maps Z_t -> Z_c Z_t."""
    sets = []
    for i in range(n):
        s = {i}
        for c, t in reversed(cnot_ring(n)):
            if t in s:
                s ^= {c}
        sets.append(sorted(s))
    return sets


class CosineProductLayer(nn.Module):
    """Exact, fast evaluation of Comajoan Cara's n-qubit quantum layer.
    (..., n) -> (..., n). n trainable angles, Xavier-normal init as in their Flax code."""

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.w = nn.Parameter(torch.randn(n) * (2.0 / (1 + n)) ** 0.5)
        mask = torch.zeros(n, n, dtype=torch.bool)
        for i, s in enumerate(z_support(n)):
            mask[i, s] = True
        self.register_buffer("mask", mask)

    def forward(self, x):
        c = torch.cos(x + self.w).unsqueeze(-2)                  # (..., 1, n)
        return torch.where(self.mask, c, torch.ones_like(c)).prod(dim=-1)


class ComajoanAttention(nn.Module):
    """Q, K, V from full-width quantum layers (one qubit per embedding feature), then
    ordinary scaled dot-product attention per head, as in their transformers.py."""

    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.h, self.d = n_heads, embed_dim // n_heads
        self.q, self.k, self.v = (CosineProductLayer(embed_dim) for _ in range(3))

    def forward(self, x):
        b, t, e = x.shape
        q, k, v = (f(x).reshape(b, t, self.h, self.d).transpose(1, 2)
                   for f in (self.q, self.k, self.v))
        a = torch.softmax(q @ k.transpose(-2, -1) / self.d ** 0.5, dim=-1)
        return (a @ v).transpose(1, 2).reshape(b, t, e)


class ComajoanFFN(nn.Module):
    """Their FeedForward: Dense -> quantum layer -> GELU -> Dense."""

    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, ff_dim), CosineProductLayer(ff_dim),
                                 nn.GELU(), nn.Linear(ff_dim, embed_dim))

    def forward(self, x):
        return self.net(x)
