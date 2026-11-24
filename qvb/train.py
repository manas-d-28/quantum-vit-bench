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
from .data import load_mnist


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(-1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)
    model.train()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["classic", "gaussian", "unlu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--embed", type=int, default=16)
    p.add_argument("--heads", type=int, default=4)       # 4 heads x 4 features -> 4-qubit circuits
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--ff", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true", help="smoke test: 2 epochs, 640/160/160 images")
    a = p.parse_args()

    if a.quick:
        a.epochs = 2
    torch.manual_seed(a.seed)
    split = (640, 160, 160) if a.quick else None
    tr, va, te, n_tokens, token_dim = (load_mnist(n_images=sum(split), split=split)
                                       if a.quick else load_mnist())

    model = TinyViT(n_tokens, token_dim, make_attention(a.model, a.embed, a.heads),
                    embed_dim=a.embed, n_layers=a.layers, ff_dim=a.ff).to(a.device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()

    d = a.embed // a.heads
    q = QUANTUM.get(a.model)
    meta = {"qubits_per_circuit": d if q else 0,
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
    row = {"model": a.model, "seed": a.seed, "epochs": a.epochs, "params": n_params,
           "embed": a.embed, "heads": a.heads, "layers": a.layers, **meta,
           "best_val": round(best_val, 4), "best_epoch": best_epoch,
           "test_acc": round(evaluate(model, te, a.device), 4),
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
    with open(os.path.join(a.out, f"curve_{a.model}_s{a.seed}.json"), "w") as f:
        json.dump(curve, f)


if __name__ == "__main__":
    main()
