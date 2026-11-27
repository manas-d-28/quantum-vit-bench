"""MNIST / Fashion-MNIST data pipeline shared by every model.

Images are resized to 14x14 and cut into 7 row-patches of 2x14 pixels, so each
image becomes 7 tokens of 28 features (same as Phase 1). 6000 training images
are split 4000 / 1000 / 1000 with a FIXED split seed, so every model and every
training seed sees exactly the same train / val / test images.
Other datasets, image sizes and patch sizes can be chosen (see train.py flags).
"""
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

N_IMAGES = 6000
SPLIT = (4000, 1000, 1000)
SPLIT_SEED = 0
IMG = 14
PATCH = (2, 14)


def patchify(x, ph, pw):
    """(N, H, W) -> (N, n_patches, ph*pw), row-major patch order."""
    n, h, w = x.shape
    x = x[:, : h - h % ph, : w - w % pw]
    x = x.reshape(n, h // ph, ph, w // pw, pw).permute(0, 1, 3, 2, 4)
    return x.reshape(n, (h // ph) * (w // pw), ph * pw)


def load_mnist(root="./data", batch_size=32, n_images=N_IMAGES, split=SPLIT,
               dataset="mnist", img=IMG, patch=PATCH):
    """dataset: "mnist" or "fashion" (Fashion-MNIST). img: side length after resizing.
    patch: (height, width) of each token's patch."""
    from torchvision.datasets import MNIST, FashionMNIST
    from torchvision import transforms

    cls = {"mnist": MNIST, "fashion": FashionMNIST}[dataset]
    ds = cls(root, train=True, download=True,
             transform=transforms.Compose([transforms.Resize(img), transforms.ToTensor()]))
    x = torch.stack([ds[i][0][0] for i in range(n_images)]).float()
    y = torch.tensor([ds[i][1] for i in range(n_images)])
    x = (x - x.mean()) / x.std()
    tokens = patchify(x, *patch)                      # default: (N, 7, 28)
    full = TensorDataset(tokens, y)
    tr, va, te = random_split(full, list(split),
                              generator=torch.Generator().manual_seed(SPLIT_SEED))
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(va, batch_size=batch_size),
            DataLoader(te, batch_size=batch_size),
            tokens.shape[1], tokens.shape[2])


# ----------------------------------------------------------------------------
# Pair tasks: two MNIST digits side by side, so the answer depends on comparing
# the left half of the image with the right half.
#   pair_same    "do the two digits show the same number?"  (balanced 50/50)
#   pair_larger  "is the left digit larger than the right?" (distinct digits, 50/50)
# Each 14x14 digit is cut into 7 strips of 2x14 -> 14 tokens of 28 features.
# Train/val pairs use disjoint image pools from MNIST-train; test pairs are built
# only from the official MNIST test images. All pair sampling uses fixed seeds, so
# every model and training seed sees exactly the same pairs.
# ----------------------------------------------------------------------------
PAIR_SIZES = {"train": 10000, "val": 2000, "test": 10000}


def _mnist_tensor(root, train, img):
    from torchvision.datasets import MNIST
    from torchvision.transforms.functional import resize
    ds = MNIST(root, train=train, download=True)
    x = resize(ds.data.float().div(255).unsqueeze(1), [img, img], antialias=True).squeeze(1)
    return x, ds.targets.clone()


def _make_pairs(task, x, y, pool, n, gen):
    by_digit = [pool[(y[pool] == d)] for d in range(10)]
    pick = lambda d: by_digit[d][torch.randint(len(by_digit[d]), (1,), generator=gen)].item()
    left, right, label = [], [], []
    for _ in range(n):
        if task == "pair_same":
            same = torch.randint(2, (1,), generator=gen).item()
            d1 = torch.randint(10, (1,), generator=gen).item()
            d2 = d1 if same else (d1 + torch.randint(1, 10, (1,), generator=gen).item()) % 10
            lab = same
        else:  # pair_larger: two different digits in random order
            d1, d2 = torch.randperm(10, generator=gen)[:2].tolist()
            lab = int(d1 > d2)
        i1, i2 = pick(d1), pick(d2)
        while i2 == i1:
            i2 = pick(d2)
        left.append(i1); right.append(i2); label.append(lab)
    pair = torch.cat([x[left], x[right]], dim=2)          # (n, img, 2*img)
    return pair, torch.tensor(label)


def load_pairs(task, root="./data", batch_size=32, img=IMG, patch=PATCH, sizes=None):
    sizes = sizes or PAIR_SIZES
    xtr, ytr = _mnist_tensor(root, True, img)
    xte, yte = _mnist_tensor(root, False, img)
    gen = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(xtr), generator=gen)
    pool_tr, pool_va = perm[:50000], perm[50000:]           # disjoint image pools
    a, ya = _make_pairs(task, xtr, ytr, pool_tr, sizes["train"], gen)
    b, yb = _make_pairs(task, xtr, ytr, pool_va, sizes["val"], gen)
    c, yc = _make_pairs(task, xte, yte, torch.arange(len(xte)), sizes["test"], gen)
    mean, std = a.mean(), a.std()
    tok = lambda z: patchify((z - mean) / std, *patch)
    a, b, c = tok(a), tok(b), tok(c)
    return (DataLoader(TensorDataset(a, ya), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(b, yb), batch_size=batch_size),
            DataLoader(TensorDataset(c, yc), batch_size=batch_size),
            a.shape[1], a.shape[2])
