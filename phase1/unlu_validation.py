# Phase 1 validation: does Unlu et al.'s hybrid quantum ViT reach accuracy
# close to a classical ViT of the same size on MNIST (as the paper reports)?
# Repo: https://github.com/EyupBunlu/QViT_HEP_ML4Sci
# Data follows the repo's MNIST cells: 14x14 images, 2x14 patches (7 tokens x 28),
# 4000 train / 1000 val / 1000 test, batch 32, Adam lr 1e-3.
# Both models: 2 layers, Embed_Dim 16, head dim 8, ff 32 (~3k params each).

import sys, time, json, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

sys.path.insert(0, "/kaggle/working/QViT_HEP_ML4Sci")
from QViT import *  # HViT, train, patcher, simple_dataset

torch.set_default_dtype(torch.float32)  # float64 hits a Double/Float mismatch in the repo
DEVICE = "cpu"                           # Kaggle CPU session; also sed the repo's cuda strings
EPOCHS = 100

ds = MNIST("./data", train=True, download=True,
           transform=transforms.Compose([transforms.Resize(14), transforms.ToTensor()]))
N = 6000
X = torch.stack([ds[i][0][0] for i in range(N)]).float()
X = (X - X.mean()) / X.std()
y = torch.tensor([ds[i][1] for i in range(N)])
P = patcher(X, [2, 14])
tr, va, te = random_split(simple_dataset(P, y), [4000, 1000, 1000],
                          generator=torch.Generator().manual_seed(0))
tr_dl = DataLoader(tr, batch_size=32, shuffle=True)
va_dl = DataLoader(va, batch_size=32)
te_dl = DataLoader(te, batch_size=32)

def evaluate(m, dl):
    m.eval(); correct = total = 0
    with torch.no_grad():
        for b in dl:
            pred = m(b["input"].to(DEVICE)).argmax(-1).cpu()
            correct += (pred == b["output"]).sum().item(); total += len(pred)
    m.train()
    return correct / total

results = []
for name in ["classic", "hybrid2"]:
    torch.manual_seed(0)
    m = HViT(7, 28, 8, 2, [10], name, True, "max", 16, 32).to(DEVICE)
    with torch.no_grad():                      # materialise lazy layers
        m(next(iter(tr_dl))["input"].to(DEVICE))
    n_params = sum(p.numel() for p in m.parameters())
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    t0 = time.time()
    h = train(m, tr_dl, va_dl, nn.CrossEntropyLoss(reduction="none"), opt, EPOCHS, device=DEVICE)
    wall = time.time() - t0
    m.load_state_dict(torch.load("best_state_on_training_acc"))   # best on validation
    r = {"model": name, "params": n_params, "epochs": EPOCHS,
         "wall_min": round(wall / 60, 1), "sec_per_epoch": round(wall / EPOCHS, 1),
         "best_val_acc": round(max(float(a) for a in h["val_acc"]), 4),
         "test_acc": round(evaluate(m, te_dl), 4),
         "val_acc_curve": [round(float(a), 4) for a in h["val_acc"]]}
    print(json.dumps(r), flush=True)
    results.append(r)
    json.dump(results, open("/kaggle/working/phase1_validation.json", "w"), indent=2)

print("DONE")
