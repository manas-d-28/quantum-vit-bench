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
