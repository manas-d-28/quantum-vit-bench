"""Gate-by-gate check: our explicit Unlu circuits vs the official repo's loops.
Uses a fake `tensorcircuit` that records gates, so no torch/jax needed.
Run with: python3 -I tools/check_unlu_circuits.py <this_repo> <path to QViT_HEP_ML4Sci clone>"""
import sys, types, importlib.util

bench, unlu = sys.argv[1], sys.argv[2]


class Rec:
    def __init__(self, n): self.g = []
    def H(self, i): self.g.append(("H", i))
    def rx(self, i, theta): self.g.append(("rx", i, theta))
    def ry(self, i, theta): self.g.append(("ry", i, theta))
    def cnot(self, a, b): self.g.append(("cnot", a, b))
    def expectation_ps(self, z): self.g.append(("Z", tuple(z))); return 0.0


fake = types.ModuleType("tensorcircuit")
fake.Circuit = Rec
fake.set_backend = lambda name: types.SimpleNamespace(real=lambda v: v, stack=lambda l: l,
                                                      vmap=None)
fake.set_dtype = lambda d: None
sys.modules["tensorcircuit"] = fake
for m in ["torch", "torch.nn", "jax", "jax.numpy"]:
    mod = types.ModuleType(m)
    sys.modules[m] = mod
sys.modules["torch"].nn = sys.modules["torch.nn"]
sys.modules["torch.nn"].Module = object
sys.modules["jax"].pmap = None
sys.modules["jax.numpy"].array = lambda l: l


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ours = load(f"{bench}/qvb/quantum.py", "ours")
theirs = load(f"{unlu}/QViT/circuits.py", "theirs")


def trace_ours(fn, n, n_params):
    c = Rec(n); orig = fake.Circuit
    fake.Circuit = lambda n_: c
    fn([f"x{i}" for i in range(n)], [f"w{i}" for i in range(n_params)], n)
    fake.Circuit = orig
    return c.g


def trace_theirs(ansatz, n, n_params, zs):
    c = Rec(n)
    theirs.encode_token(c, [f"x{i}" for i in range(n)], n)
    ansatz(c, None, Params(n_params), n)
    for z in zs: c.expectation_ps(z=[z])
    return c.g


class Params(list):
    def __init__(self, k): super().__init__(f"w{i}" for i in range(k)); self.shape = (k,)


ok = True
for n in [2, 3, 4]:
    a = trace_ours(ours.unlu_query_key, n, 3 * n + 1)
    b = trace_theirs(theirs.qk_ansatz, n, 3 * n + 1, [0])
    a2 = trace_ours(ours.unlu_value, n, 3 * n)
    b2 = trace_theirs(theirs.v_ansatz, n, 3 * n, range(n))
    used = {g[-1] for g in a if g[0] in ("rx", "ry") and str(g[-1]).startswith("w")}
    print(f"n={n}: query/key match={a == b} (gates={len(a)}, params used={len(used)}/{3*n+1}); "
          f"value match={a2 == b2} (gates={len(a2)})")
    ok &= a == b and a2 == b2 and len(used) == 3 * n + 1
print("ALL MATCH" if ok else "MISMATCH")
