"""HQViT (arXiv:2504.02730): whole-image amplitude encoding + swap-test attention.

What the paper specifies, and how it maps onto the shared backbone (T tokens of
E = 16 features each, after LayerNorm):
  * All tokens are concatenated and amplitude-encoded as ONE state, normalised by the
    whole image's norm: |psi> = sum_{i,l} x_il / ||X|| |i>|l>. Index register: patch i
    (here ceil(log2 7) = 3 qubits); patch register: feature l (log2 16 = 4 qubits).
  * U_Q, U_K, U_V are block-diagonal: the same PQC acts on the patch register of every
    patch. So |q_i> = U_Q x_i / ||X||, etc., which are unnormalised sub-vectors.
  * Attention coefficient = |<q_i|k_j>|^2, read out with a swap test between the Q and K
    registers. Softmax over j. No 1/sqrt(d) scaling is given, so none is applied.
  * V is measured, so v_il = |(U_V x_i)_l|^2 / ||X||^2 (probabilities). One head (the
    paper doesn't use multi-head attention).
What the paper does NOT give is the gate list of the PQC. Assumption here: D = 2 layers
of [Ry, Rz on every qubit, CNOT ring], then a final Ry on every qubit (paper: "single-
and two-qubit gates, and a final set of R_y gates", D = 2 for MNIST).

Because the PQC acts only on the 4-qubit patch register, its effect is a 16x16 unitary.
This module builds that unitary in PyTorch and applies it, which is an exact simulation
of the noiseless circuit with exact (infinite-shot) swap-test and measurement
probabilities. `tools/test_circuits.py` checks the unitary against TensorCircuit.
"""
import math
import torch
import torch.nn as nn

HQVIT_DEPTH = 2


def _ry(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(torch.complex64)


def _rz(t):
    z = torch.zeros_like(t)
    e = torch.polar(torch.ones_like(t), -t / 2)
    return torch.stack([torch.stack([e, z.to(e.dtype)]), torch.stack([z.to(e.dtype), e.conj()])])


def _kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out


def _cnot(n, c, t):
    """Permutation matrix of CNOT(c, t) on n qubits; qubit 0 is the most significant bit."""
    dim = 2 ** n
    perm = []
    for b in range(dim):
        if (b >> (n - 1 - c)) & 1:
            b ^= 1 << (n - 1 - t)
        perm.append(b)
    m = torch.zeros(dim, dim, dtype=torch.complex64)
    m[perm, list(range(dim))] = 1
    return m


def ring_cnots(n):
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    return [(j, (j + 1) % n) for j in range(n)]


def n_params(n, depth=HQVIT_DEPTH):
    return depth * 2 * n + n


def pqc_unitary(w, n, depth=HQVIT_DEPTH, cnot_mats=None):
    """16x16 (2^n x 2^n) unitary of the assumed HQViT PQC, differentiable in w."""
    if cnot_mats is None:
        cnot_mats = [_cnot(n, c, t) for c, t in ring_cnots(n)]
    u = torch.eye(2 ** n, dtype=torch.complex64)
    k = 0
    for _ in range(depth):
        u = _kron_all([_ry(w[k + i]) for i in range(n)]) @ u
        k += n
        u = _kron_all([_rz(w[k + i]) for i in range(n)]) @ u
        k += n
        for m in cnot_mats:
            u = m @ u
    return _kron_all([_ry(w[k + i]) for i in range(n)]) @ u


class _Unitary(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.n = int(round(math.log2(dim)))
        assert 2 ** self.n == dim, "HQViT patch register needs a power-of-2 feature size"
        self.w = nn.Parameter(torch.rand(n_params(self.n)) * 2 * math.pi)
        self.register_buffer("cnots", torch.stack([_cnot(self.n, c, t) for c, t in ring_cnots(self.n)]))

    def forward(self):
        return pqc_unitary(self.w, self.n, cnot_mats=list(self.cnots))


def _normalise_whole_image(x):
    """Amplitude encoding of all tokens together: divide by the norm of the whole image."""
    return x / x.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None]


class HQViTAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.uq, self.uk, self.uv = _Unitary(embed_dim), _Unitary(embed_dim), _Unitary(embed_dim)

    def forward(self, x):                                   # (B, T, E), real
        xn = _normalise_whole_image(x).to(torch.complex64)
        q = xn @ self.uq().T                                # |q_i> = U_Q x_i / ||X||
        k = xn @ self.uk().T
        coef = (q.conj() @ k.transpose(1, 2)).abs() ** 2    # |<q_i|k_j>|^2 (swap test)
        v = (xn @ self.uv().T).abs() ** 2                   # measured probabilities
        return torch.softmax(coef, dim=-1) @ v


class OverlapMimic(nn.Module):
    """Classical mimic of HQViT: identical formula, but W_Q, W_K, W_V are free real matrices
    instead of PQC unitaries. Separates 'the scoring rule' from 'the quantum circuit'."""

    def __init__(self, embed_dim):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(3))

    def forward(self, x):
        xn = _normalise_whole_image(x)
        coef = (self.q(xn) @ self.k(xn).transpose(1, 2)) ** 2
        return torch.softmax(coef, dim=-1) @ self.v(xn) ** 2


class _LowRank(nn.Module):
    """x -> x @ (A @ B)^T with A: E x r, B: r x E  (2 E r parameters instead of E^2)."""

    def __init__(self, dim, rank):
        super().__init__()
        self.a = nn.Linear(rank, dim, bias=False)
        self.b = nn.Linear(dim, rank, bias=False)

    def forward(self, x):
        return self.a(self.b(x))


class OverlapMimicSmall(OverlapMimic):
    """Same as OverlapMimic, but W_Q, W_K, W_V are rank-2, so the whole model has about
    as many parameters as HQViT (3 x 64 = 192 per block, vs HQViT's 60 and the full
    mimic's 768). Answers the reviewer point that the full mimic is ~38% larger."""

    def __init__(self, embed_dim, rank=2):
        nn.Module.__init__(self)
        self.q, self.k, self.v = (_LowRank(embed_dim, rank) for _ in range(3))
