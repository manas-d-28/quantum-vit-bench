"""Train one model on MNIST and append a row to a results CSV.

    python -m qvb.train --model unlu --seed 0 --epochs 60
    python -m qvb.train --model classic --quick      # 2 epochs on 640 images, smoke test
"""
import argparse
import copy
import csv
import json
import os
import time

import torch
import torch.nn as nn

from .attention import QUANTUM, make_attention
from .backbone import TinyViT
from .data import load_mnist, load_pairs


def predict(model, loader, device):
    """Predicted class and true label for every example, in loader order."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).argmax(-1).cpu())
            labels.append(y)
    model.train()
    return torch.cat(preds), torch.cat(labels)


def evaluate(model, loader, device):
    p, y = predict(model, loader, device)
    return (p == y).float().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["classic", "gaussian", "unlu", "qsann", "comajoan", "hqvit", "overlap", "uniform"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--embed", type=int, default=16)
    p.add_argument("--heads", type=int, default=4)       # 4 heads x 4 features -> 4-qubit circuits
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--ff", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results")
    p.add_argument("--task", default="digits", choices=["digits", "pair_same", "pair_larger"],
                   help="digits: 10-class classification; pair_*: two digits side by side, 2 classes")
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion"])
    p.add_argument("--img", type=int, default=14, help="image side after resizing")
    p.add_argument("--patch", default="2x14", help="patch height x width, e.g. 2x14 or 4x4")
    p.add_argument("--quick", action="store_true", help="smoke test: 2 epochs, 640/160/160 images")
    a = p.parse_args()

    if a.quick:
        a.epochs = 2
    torch.manual_seed(a.seed)
    split = (640, 160, 160) if a.quick else None
    data_kw = dict(dataset=a.dataset, img=a.img, patch=tuple(int(v) for v in a.patch.split("x")))
    if a.task == "digits":
        tr, va, te, n_tokens, token_dim = (load_mnist(n_images=sum(split), split=split, **data_kw)
                                           if a.quick else load_mnist(**data_kw))
        n_classes = 10
    else:
        sizes = {"train": 640, "val": 160, "test": 160} if a.quick else None
        tr, va, te, n_tokens, token_dim = load_pairs(a.task, img=a.img, patch=data_kw["patch"],
                                                     sizes=sizes)
        n_classes = 2

    if a.model == "comajoan":     # quantum Q/K/V, quantum W_O and a quantum layer in the FFN
        from .comajoan import ComajoanAttention, ComajoanFFN, CosineProductLayer
        parts = dict(make_attention=lambda: ComajoanAttention(a.embed, a.heads),
                     make_proj=lambda: CosineProductLayer(a.embed),
                     make_ffn=lambda: ComajoanFFN(a.embed, a.ff))
    elif a.model in ("hqvit", "overlap"):   # single full-width head, as in HQViT
        from .hqvit import HQViTAttention, OverlapMimic
        cls = HQViTAttention if a.model == "hqvit" else OverlapMimic
        parts = dict(make_attention=lambda: cls(a.embed))
    else:
        parts = dict(make_attention=make_attention(a.model, a.embed, a.heads))
    model = TinyViT(n_tokens, token_dim, embed_dim=a.embed, n_layers=a.layers, ff_dim=a.ff,
                    n_classes=n_classes, **parts).to(a.device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()

    d = a.embed // a.heads
    q = QUANTUM.get(a.model)
    if a.model == "hqvit":        # 2 registers of ceil(log2(T*E)) qubits + 1 swap-test ancilla;
        import math               # per layer: one swap-test circuit (Q,K) and one V circuit
        nq = math.ceil(math.log2(n_tokens * a.embed))
        meta = {"qubits_per_circuit": f"{2 * nq + 1}", "circuits_per_image": 2 * a.layers}
    elif a.model == "comajoan":     # per layer and token: Q, K, V, W_O (embed qubits) + FFN (ff qubits)
        meta = {"qubits_per_circuit": f"{a.embed}/{a.ff}", "circuits_per_image": 5 * n_tokens * a.layers}
    else:
        meta = {"qubits_per_circuit": q["qubits"](d) if q else 0,
                "circuits_per_image": q["circuits_per_token_per_head"] * n_tokens * a.heads * a.layers if q else 0}
    print(f"model={a.model} seed={a.seed} params={n_params} {meta}", flush=True)

    best_val, best_epoch, best_state, curve = -1.0, -1, None, []
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        te0 = time.time()
        for x, y in tr:
            loss = loss_fn(model(x.to(a.device)), y.to(a.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = evaluate(model, va, a.device)
        curve.append(round(val, 4))
        if val > best_val:
            best_val, best_epoch, best_state = val, epoch, copy.deepcopy(model.state_dict())
        print(f"epoch {epoch:3d}  loss {loss.item():.3f}  val {val:.4f}  ({time.time() - te0:.1f}s)",
              flush=True)
    wall = time.time() - t0

    model.load_state_dict(best_state)
    test_pred, test_y = predict(model, te, a.device)
    tag = f"{a.model}_{a.task}_{a.dataset}{a.img}_{a.patch}_lr{a.lr:g}_s{a.seed}"
    row = {"model": a.model, "seed": a.seed, "task": a.task, "lr": a.lr, "dataset": a.dataset, "img": a.img, "patch": a.patch,
           "tokens": n_tokens, "epochs": a.epochs, "params": n_params,
           "embed": a.embed, "heads": a.heads, "layers": a.layers, **meta,
           "best_val": round(best_val, 4), "best_epoch": best_epoch,
           "test_acc": round((test_pred == test_y).float().mean().item(), 4),
           "sec_per_epoch": round(wall / a.epochs, 1), "wall_min": round(wall / 60, 1),
           "quick": a.quick}
    print("RESULT " + json.dumps(row), flush=True)

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    # per-example test predictions, for paired statistics across models (tools/stats.py)
    torch.save({"pred": test_pred.to(torch.int8), "label": test_y.to(torch.int8)},
               os.path.join(a.out, f"preds_{tag}.pt"))
    with open(os.path.join(a.out, f"curve_{tag}.json"), "w") as f:
        json.dump(curve, f)


if __name__ == "__main__":
    main()
