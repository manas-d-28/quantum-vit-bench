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


N_PARAMS = {unlu_query_key: lambda n: 3 * n + 1, unlu_value: lambda n: 3 * n}


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
        self.w = nn.Parameter(torch.randn(n_params) / n_params ** 0.5)   # Unlu's init
        self.f = _batched(circuit, n_qubits)

    def forward(self, x):
        return self.f(x, self.w)
