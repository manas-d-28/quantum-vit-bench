"""Check every circuit in qvb/quantum.py against an independent NumPy statevector
simulation of the same gate list, on random inputs and parameters.

    python -m tools.test_circuits          (needs tensorcircuit-ng + jax)

The circuit functions are run twice: once with a recording fake Circuit (to get
the gate list, then simulated with NumPy below), and once for real in TensorCircuit.
"""
import numpy as np

from qvb import quantum as Q

I2 = np.eye(2)
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
PX = np.array([[0, 1], [1, 0]])
PY = np.array([[0, -1j], [1j, 0]])
PZ = np.diag([1, -1])


def rx(t): return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * PX
def ry(t): return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * PY


def apply1(psi, U, q, n):
    psi = psi.reshape([2] * n)
    psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)
    return psi.reshape(-1)


def cnot(psi, a, b, n):
    psi = psi.reshape([2] * n).copy()
    idx = [slice(None)] * n
    idx[a] = 1
    sub = psi[tuple(idx)]
    bb = b if b < a else b - 1
    psi[tuple(idx)] = np.flip(sub, axis=bb)
    return psi.reshape(-1)


def expval(psi, P, q, n):
    return float(np.real(np.vdot(psi, apply1(psi, P, q, n))))


class Recorder:
    """Stands in for tc.Circuit and records gates; qubit 0 is the most significant bit,
    as in TensorCircuit."""
    def __init__(self, n): self.n, self.g = n, []
    def H(self, i): self.g.append(("H", i))
    def rx(self, i, theta): self.g.append(("rx", i, float(theta)))
    def ry(self, i, theta): self.g.append(("ry", i, float(theta)))
    def cnot(self, a, b): self.g.append(("cnot", a, b))
    def expectation_ps(self, x=(), y=(), z=()):
        psi = np.zeros(2 ** self.n, complex); psi[0] = 1
        for g in self.g:
            if g[0] == "H": psi = apply1(psi, H, g[1], self.n)
            elif g[0] == "rx": psi = apply1(psi, rx(g[2]), g[1], self.n)
            elif g[0] == "ry": psi = apply1(psi, ry(g[2]), g[1], self.n)
            else: psi = cnot(psi, g[1], g[2], self.n)
        (P, q), = [(P, q[0]) for P, q in ((PX, x), (PY, y), (PZ, z)) if q]
        return expval(psi, P, q, self.n)


def numpy_run(fn, x, w, n):
    real_circuit, real_K = Q.tc.Circuit, Q.K
    Q.tc.Circuit = Recorder
    Q.K = type("K", (), {"real": staticmethod(lambda v: v), "stack": staticmethod(np.array)})
    try:
        return np.atleast_1d(fn(x, w, n))
    finally:
        Q.tc.Circuit, Q.K = real_circuit, real_K


def main():
    rng = np.random.default_rng(0)
    cases = [(Q.unlu_query_key, 4, 4), (Q.unlu_value, 4, 4), (Q.unlu_query_key, 2, 2),
             (Q.qsann_query_key, 2, 4), (Q.qsann_value, 2, 4), (Q.qsann_value, 3, 6)]
    ok = True
    for fn, n, d in cases:
        for _ in range(3):
            x = rng.uniform(-np.pi, np.pi, d).astype(np.float32)
            w = rng.uniform(-np.pi, np.pi, Q.N_PARAMS[fn](n)).astype(np.float32)
            ref = numpy_run(fn, x, w, n)
            got = np.atleast_1d(np.asarray(fn(x, w, n)))
            err = np.max(np.abs(ref - got))
            ok &= err < 1e-4
        print(f"{fn.__name__:16s} n={n} d={d}: max |tc - numpy| = {err:.2e}")
    # Comajoan Cara: exact cosine-product formula vs TensorCircuit vs NumPy
    import torch
    from qvb.comajoan import CosineProductLayer
    for n in [2, 3, 4, 6, 8]:
        x = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
        layer = CosineProductLayer(n)
        w = layer.w.detach().numpy()
        formula = layer(torch.tensor(x)).detach().numpy()
        got = np.asarray(Q.comajoan_layer(x, w, n))
        ref = numpy_run(Q.comajoan_layer, x, w, n)
        err = max(np.max(np.abs(formula - got)), np.max(np.abs(formula - ref)))
        ok &= err < 1e-4
        print(f"comajoan_layer   n={n}: max |formula - tc/numpy| = {err:.2e}")
    # analytic check: QSANN with all-zero inputs and weights gives |++>, so <Z>=0 and <X>=1
    v = np.asarray(Q.qsann_value(np.zeros(4, np.float32), np.zeros(6, np.float32), 2))
    print("qsann_value(0, 0) =", np.round(v, 4), "(expected [0, 0, 1, 1])")
    ok &= np.allclose(v, [0, 0, 1, 1], atol=1e-5)
    print("ALL CIRCUIT CHECKS PASSED" if ok else "CIRCUIT CHECK FAILED")


if __name__ == "__main__":
    main()
