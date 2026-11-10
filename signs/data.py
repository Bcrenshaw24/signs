"""
Download and prepare GTSRB (German Traffic Sign Recognition Benchmark, 43 classes).

Run once to download the dataset and build the cached tensors:

    python -m signs.data                                  # torchvision source (no account needed)
    python -m signs.data --source kaggle                  # Kaggle copy via kagglehub
    python -m signs.data --source kaggle --crop-roi       # crop to sign bounding boxes
    python -m signs.data --img-size 48                    # different resolution

Sources
-------
torchvision : torchvision.datasets.GTSRB, downloaded from the official mirror
              into <root>/gtsrb/ (~350 MB). No credentials needed.
kaggle      : "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign" via kagglehub
              (cached in ~/.cache/kagglehub). Has Train.csv/Test.csv with sign
              bounding boxes, so --crop-roi is available. Some kagglehub
              versions need Kaggle credentials (KAGGLE_USERNAME / KAGGLE_KEY
              or ~/.kaggle/kaggle.json).

Images are resized once and cached as uint8 tensors
(<root>/gtsrb_<source>[_roi]_<split>_<size>.pt), so training is just tensor indexing.
"""

import argparse
import csv
import math
import os

import torch
import torch.nn.functional as F

NUM_CLASSES = 43
KAGGLE_HANDLE = "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def download_torchvision(root="./data"):
    import torchvision
    os.makedirs(root, exist_ok=True)
    for split in ("train", "test"):
        torchvision.datasets.GTSRB(root, split=split, download=True)
    return root


def download_kaggle():
    import kagglehub
    return kagglehub.dataset_download(KAGGLE_HANDLE)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #

class KaggleGTSRB(torch.utils.data.Dataset):
    """Kaggle layout: root/Train.csv, root/Test.csv with columns
    Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path"""

    def __init__(self, root, split, size, crop_roi=False):
        self.root, self.size, self.crop_roi = root, size, crop_roi
        path = os.path.join(root, "Train.csv" if split == "train" else "Test.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found -- expected the folder containing Train.csv/Test.csv")
        with open(path, newline="") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        import torchvision.transforms.functional as TF
        r = self.rows[i]
        img = Image.open(os.path.join(self.root, r["Path"])).convert("RGB")
        if self.crop_roi:
            img = img.crop((int(r["Roi.X1"]), int(r["Roi.Y1"]), int(r["Roi.X2"]) + 1, int(r["Roi.Y2"]) + 1))
        img = img.resize((self.size, self.size), Image.BILINEAR)
        return TF.pil_to_tensor(img), int(r["ClassId"])


def _stack_dataset(ds, workers=2):
    loader = torch.utils.data.DataLoader(ds, batch_size=512, num_workers=workers)
    xs, ys = [], []
    for i, (x, y) in enumerate(loader):
        xs.append(x)
        ys.append(torch.as_tensor(y))
        print(f"\r[data]   {min((i + 1) * 512, len(ds))}/{len(ds)} images", end="", flush=True)
    print()
    return torch.cat(xs), torch.cat(ys).long()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def fake_split(split, img_size=32):
    """Random images/labels for smoke tests."""
    n = 2048 if split == "train" else 512
    g = torch.Generator().manual_seed(0 if split == "train" else 1)
    return (torch.randint(0, 256, (n, 3, img_size, img_size), dtype=torch.uint8, generator=g),
            torch.randint(0, NUM_CLASSES, (n,), generator=g))


def load_split(source="torchvision", split="train", root="./data", img_size=32,
               crop_roi=False, cache_dir=None):
    """
    Returns (images uint8 N x 3 x S x S, labels int64 N). Downloads and builds
    the cache on first use.

    source: "torchvision" | "kaggle" | "fake"
    root:   torchvision -> download dir; kaggle -> folder with Train.csv,
            or "auto" to fetch with kagglehub.
    """
    if source == "fake":
        return fake_split(split, img_size)
    if source == "torchvision" and crop_roi:
        raise ValueError("--crop-roi needs the kaggle source (torchvision has no bounding boxes)")

    if source == "kaggle" and root == "auto":
        root = download_kaggle()
        cache_dir = cache_dir or "./data"

    cache_dir = cache_dir or root
    os.makedirs(cache_dir, exist_ok=True)
    tag = f"{source}{'_roi' if crop_roi else ''}_{split}_{img_size}"
    cache = os.path.join(cache_dir, f"gtsrb_{tag}.pt")
    if os.path.exists(cache):
        return torch.load(cache)

    print(f"[data] building {split} cache -> {cache} (one-off)")
    if source == "torchvision":
        import torchvision
        from torchvision import transforms as T
        tf = T.Compose([T.Resize((img_size, img_size)), T.PILToTensor()])
        ds = torchvision.datasets.GTSRB(root, split=split, transform=tf, download=True)
    elif source == "kaggle":
        ds = KaggleGTSRB(root, split, img_size, crop_roi)
    else:
        raise ValueError(f"unknown source '{source}'")
    x, y = _stack_dataset(ds)
    torch.save((x, y), cache)
    return x, y


def augment(x):
    """GPU batch augmentation for float images in [0,1]: small rotation/scale/
    shift + brightness/contrast. No horizontal flips -- they change the
    meaning of many signs."""
    N, dev = x.shape[0], x.device
    ang = (torch.rand(N, device=dev) * 2 - 1) * math.radians(12)
    scale = 1 + (torch.rand(N, device=dev) * 2 - 1) * 0.1
    shift = (torch.rand(N, 2, device=dev) * 2 - 1) * 0.15
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1),
                         torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    b = 0.7 + 0.6 * torch.rand(N, 1, 1, 1, device=dev)
    c = 0.7 + 0.6 * torch.rand(N, 1, 1, 1, device=dev)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - mean) * c + mean * b).clamp(0, 1)


# --------------------------------------------------------------------------- #
# CLI: download + build caches
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["torchvision", "kaggle"], default="torchvision")
    p.add_argument("--root", default=None,
                   help="torchvision: download dir (default ./data); kaggle: folder with Train.csv (default: kagglehub)")
    p.add_argument("--cache-dir", default="./data")
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--crop-roi", action="store_true")
    args = p.parse_args()

    root = args.root or ("./data" if args.source == "torchvision" else "auto")
    for split in ("train", "test"):
        x, y = load_split(args.source, split, root, args.img_size, args.crop_roi, args.cache_dir)
        counts = torch.bincount(y, minlength=NUM_CLASSES)
        print(f"[data] {split}: {tuple(x.shape)}  classes={int((counts > 0).sum())}  "
              f"per-class min/max={int(counts.min())}/{int(counts.max())}")
    print("[data] ready")


if __name__ == "__main__":
    main()
