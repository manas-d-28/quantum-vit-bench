"""Quantum circuits (simulated with TensorCircuit-NG on JAX) wrapped as PyTorch layers.

Only imported by the quantum attention modules, so classical runs don't need
tensorcircuit or jax installed.
"""
import torch
import torch.nn as nn
import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")


# ----------------------------------------------------------------------------
# Unlu et al. (2024), arXiv:2402.00776. Faithful to their official repo
# (github.com/EyupBunlu/QViT_HEP_ML4Sci, QViT/circuits.py), written out explicitly.
# ----------------------------------------------------------------------------
def _encode(c, x, n):
    """Data loading: Hadamard then Rx(x_i) on qubit i (one feature per qubit)."""
    for i in range(n):
        c.H(i)
        c.rx(i, theta=x[i])


def _cnot_ring(c, n):
    for i in range(n - 1):
        c.cnot(i, i + 1)
    c.cnot(n - 1, 0)


def unlu_query_key(x, w, n):
    """3n+1 parameters. Rx layer, Ry layer, CNOT ring, Ry layer, CNOT ring,
    Ry on qubit 0. Output: <Z_0>, a single number in [-1, 1]."""
    c = tc.Circuit(n)
    _encode(c, x, n)
    for i in range(n):
        c.rx(i, theta=w[i])
    for i in range(n):
        c.ry(i, theta=w[n + i])
    _cnot_ring(c, n)
    for i in range(n):
        c.ry(i, theta=w[2 * n + i])
    _cnot_ring(c, n)
    c.ry(0, theta=w[3 * n])
    return K.real(c.expectation_ps(z=[0]))


def unlu_value(x, w, n):
    """3n parameters. Rx layer, Ry layer, CNOT ring, Ry layer.
    Output: (<Z_0>, ..., <Z_{n-1}>), a vector of n numbers in [-1, 1]."""
    c = tc.Circuit(n)
    _encode(c, x, n)
    for i in range(n):
        c.rx(i, theta=w[i])
    for i in range(n):
        c.ry(i, theta=w[n + i])
    _cnot_ring(c, n)
    for i in range(n):
        c.ry(i, theta=w[2 * n + i])
    return K.stack([K.real(c.expectation_ps(z=[i])) for i in range(n)])


# ----------------------------------------------------------------------------
# QSANN: Li, Zhao & Wang (2022), arXiv:2205.05625. No official code; built from
# the paper (Eqs. 3-4, Fig. 2, Table I).
#   * The encoder takes d = n(D_enc + 2) features on n qubits: Hadamards, then the
#     features are the angles of an Rx layer and an Ry layer, followed by D_enc
#     blocks of (CNOT ring, Ry layer). For our 4-feature heads: n = 2, D_enc = 0.
#   * U_q, U_k, U_v use the same ansatz with trainable angles and depth D = 1
#     (Table I uses D = 1 for most datasets): n(D + 2) = 3n parameters each.
#   * Query and key measure <Z_1>. The value measures d Pauli observables taken
#     in the paper's order Z_1..Z_n, X_1..X_n, Y_1..Y_n (first d of them).
# ----------------------------------------------------------------------------
QSANN_DEPTH = 1


def _qsann_ansatz(c, angles, n, depth):
    """Rx layer, Ry layer, then `depth` blocks of (CNOT ring, Ry layer)."""
    for i in range(n):
        c.rx(i, theta=angles[i])
    for i in range(n):
        c.ry(i, theta=angles[n + i])
    for b in range(depth):
        if n > 1:
            _cnot_ring(c, n)
        for i in range(n):
            c.ry(i, theta=angles[(b + 2) * n + i])


def _qsann_state(x, w, n):
    c = tc.Circuit(n)
    for i in range(n):
        c.H(i)
    _qsann_ansatz(c, x, n, depth=x.shape[0] // n - 2)      # encoder, D_enc from d = n(D_enc+2)
    _qsann_ansatz(c, w, n, depth=QSANN_DEPTH)              # trainable U_q / U_k / U_v
    return c


def qsann_query_key(x, w, n):
    return K.real(_qsann_state(x, w, n).expectation_ps(z=[0]))


def qsann_value(x, w, n):
    c = _qsann_state(x, w, n)
    obs = ([{"z": [i]} for i in range(n)] + [{"x": [i]} for i in range(n)]
           + [{"y": [i]} for i in range(n)])[: x.shape[0]]
    return K.stack([K.real(c.expectation_ps(**o)) for o in obs])


N_PARAMS = {unlu_query_key: lambda n: 3 * n + 1, unlu_value: lambda n: 3 * n,
            qsann_query_key: lambda n: n * (QSANN_DEPTH + 2),
            qsann_value: lambda n: n * (QSANN_DEPTH + 2)}
INIT_STD = {qsann_query_key: 0.1, qsann_value: 0.1}   # paper: N(0, 0.01) i.e. std 0.1


_COMPILED = {}


def _batched(circuit, n_qubits):
    """One jit-compiled, batch-vectorised, PyTorch-callable function per
    (circuit, n_qubits). The parameters are an argument, so every layer that
    uses the same circuit shares one compiled function and compiles only once."""
    key = (circuit.__name__, n_qubits)
    if key not in _COMPILED:
        f = lambda x, w: circuit(x, w, n_qubits)
        _COMPILED[key] = tc.interfaces.torch_interface(K.vmap(f, vectorized_argnums=0), jit=True)
    return _COMPILED[key]


class QLayer(nn.Module):
    """Runs one circuit on every row of a (batch, n) input. The circuit is
    vectorised over the batch with JAX vmap, jit-compiled, and exposed to
    PyTorch so gradients flow into the circuit parameters `w`."""

    def __init__(self, circuit, n_qubits):
        super().__init__()
        n_params = N_PARAMS[circuit](n_qubits)
        std = INIT_STD.get(circuit, 1 / n_params ** 0.5)               # default: Unlu's init
        self.w = nn.Parameter(torch.randn(n_params) * std)
        self.f = _batched(circuit, n_qubits)

    def forward(self, x):
        return self.f(x, self.w)
