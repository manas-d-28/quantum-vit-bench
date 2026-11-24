# Diagnostic: time individual training steps of Unlu's hybrid2 model
import sys, time, torch, torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, "/kaggle/working/QViT_HEP_ML4Sci")
from QViT import *
torch.set_default_dtype(torch.float32)

P = torch.randn(32 * 6, 7, 28)            # same shape as our MNIST patches
y = torch.randint(0, 10, (32 * 6,))
dl = DataLoader(simple_dataset(P, y), batch_size=32)

cfg = sys.argv[1] if len(sys.argv) > 1 else "big"
if cfg == "big":   # repo's MNIST hybrid2 setting
    m = HViT(7, 28, 8, 4, [10], "hybrid2", True, "max", 32, 32).to("cpu")
else:              # same size as the classic model (2 layers, Embed_Dim 16)
    m = HViT(7, 28, 8, 2, [10], "hybrid2", True, "max", 16, 32).to("cpu")

lossf, opt = nn.CrossEntropyLoss(), None
for k, b in enumerate(dl):
    t = time.time()
    out = m(b["input"].to("cpu"))
    if opt is None:
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    l = lossf(out, b["output"].to("cpu"))
    opt.zero_grad(); l.backward(); opt.step()
    print(cfg, "batch", k, "sec", round(time.time() - t, 2), flush=True)
print("params", sum(p.numel() for p in m.parameters()))
