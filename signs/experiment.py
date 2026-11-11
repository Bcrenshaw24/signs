"""
Train and compare every (edge filter x model) combination on GTSRB.

Examples
--------
# Download the data first (or let the experiment do it on first run)
python -m signs.data

# Full run
python -m signs.experiment

# Kaggle copy of the dataset (folder containing Train.csv / Test.csv / Train/ / Test/)
python -m signs.experiment --dataset kaggle --data-root /content/gtsrb --crop-roi
# ...or let kagglehub fetch it
python -m signs.experiment --dataset kaggle --data-root auto

# Sweep bit-widths and sparsity levels
python -m signs.experiment --filters learned --models quant sparse --w-bits 8 4 2 1 --sparsity 0.8 0.9 0.95

# Quick CPU smoke test with random data
python -m signs.experiment --dataset fake --epochs 1 --subset 512

From Python (e.g. Colab):

    import signs
    results = signs.run(dataset="fake", epochs=1, subset=512)
    results = signs.run(filters=["learned"], models=["quant"], w_bits=[4, 2])

Outputs (in --out): results.csv, results.json, filters.png, ckpt/*.pt
"""

import argparse
import copy
import csv
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from .filters import build_filter, FILTERS
from .models import build_model, cubic_sparsity_schedule, MODELS, SparseCNN
from .data import load_split, augment


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model, x_u8, y, device, bs=1024):
    model.eval()
    correct = 0
    for i in range(0, len(x_u8), bs):
        xb = x_u8[i:i + bs].to(device).float() / 255.0
        correct += (model(xb).argmax(1).cpu() == y[i:i + bs]).sum().item()
    return correct / len(x_u8)


def train(model, data, args, device, log_prefix=""):
    (tx, ty), (vx, vy) = data
    tx, ty = tx.to(device), ty.to(device)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(tx) / args.batch_size)
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.15)

    sparse = isinstance(model, SparseCNN)
    prune_start, prune_end = int(0.10 * total), int(0.70 * total)
    prune_every = max(1, steps_per_epoch // 4)

    step = 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tx), device=device)
        run_loss, t0 = 0.0, time.time()
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            xb = tx[idx].float() / 255.0
            if not args.no_augment:
                xb = augment(xb)
            loss = F.cross_entropy(model(xb), ty[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if sparse:
                if prune_start <= step <= prune_end + prune_every and (step % prune_every == 0 or step >= prune_end):
                    model.set_sparsity(cubic_sparsity_schedule(step, prune_end, model.target_sparsity, prune_start))
                model.apply_masks()
            run_loss += loss.item()

        msg = f"{log_prefix} epoch {epoch + 1:>2}/{args.epochs}  loss {run_loss / steps_per_epoch:.3f}"
        if args.eval_every and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs):
            msg += f"  test acc {evaluate(model, vx, vy, device):.4f}"
        if sparse:
            msg += f"  sparsity {model.sparsity():.3f}"
        print(msg + f"  ({time.time() - t0:.1f}s)", flush=True)
    return model


@torch.no_grad()
def latency_ms(model, img_size, device, batch=256, reps=20):
    """Per-image inference latency (ms) on `device`, incl. the edge filter.
    Note: QAT models use *fake* quantization in fp32, so their latency is NOT
    representative of int kernels -- compare their storage_kb instead."""
    m = model.to(device).eval()
    x = torch.rand(batch, 3, img_size, img_size, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(5):
        m(x)
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        m(x)
    sync()
    return (time.perf_counter() - t0) / (reps * batch) * 1000.0


# --------------------------------------------------------------------------- #
# Filter visualisation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def save_filter_viz(front_ends, x_u8, path, n=6):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not installed, skipping")
        return
    x = x_u8[:n].float() / 255.0
    rows = ["input"] + list(front_ends)
    fig, axes = plt.subplots(len(rows), n, figsize=(1.6 * n, 1.6 * len(rows)))
    for r, name in enumerate(rows):
        if name == "input":
            imgs = x.permute(0, 2, 3, 1)
        else:
            fe = copy.deepcopy(front_ends[name]).cpu().eval()
            e = fe.filt(x)
            if name == "sobel":
                e = e[:, 2:3]                       # magnitude channel
            elif name == "canny":
                e = e[:, 0:1]                       # binary edges
            elif name == "none":
                e = e.mean(1, keepdim=True)
            else:
                e = e.norm(dim=1, keepdim=True)     # learned: response energy
            e = e / e.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            imgs = e[:, 0]
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[c].numpy(), cmap=None if name == "input" else "gray")
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[viz] saved {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def model_variants(args):
    """Yields (tag, kind, kwargs) for every model configuration requested."""
    for kind in args.models:
        if kind == "dense":
            yield "dense-fp32", "dense", {}
        elif kind == "quant":
            for wb in args.w_bits:
                yield (f"quant-w{wb}a{args.a_bits}", "quant",
                       dict(w_bits=wb, a_bits=args.a_bits, first_last_bits=args.first_last_bits))
        elif kind == "sparse":
            for s in args.sparsity:
                yield f"sparse-{int(round(s * 100))}", "sparse", dict(target_sparsity=s)


def filter_kwargs(name, args):
    if name == "learned":
        return dict(in_channels=args.learned_in_channels, out_channels=args.learned_channels,
                    kernel_size=args.learned_kernel, zero_mean=not args.learned_free)
    if name == "canny":
        return dict(return_magnitude=args.canny_magnitude)
    return {}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["torchvision", "kaggle", "fake"], default="torchvision")
    p.add_argument("--data-root", default="./data", help="'auto' with --dataset kaggle uses kagglehub")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--crop-roi", action="store_true", help="kaggle only: crop to the sign bounding box")
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--subset", type=int, default=0, help="use only N training images (quick tests)")

    p.add_argument("--filters", nargs="+", default=["sobel", "canny", "learned"], choices=FILTERS)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--concat-input", action="store_true", help="feed edges AND raw RGB to the CNN")
    p.add_argument("--width", type=int, default=32, help="base channel width of the CNN")

    p.add_argument("--w-bits", nargs="+", type=int, default=[4], help="weight bits for quant model(s); 1 = binary")
    p.add_argument("--a-bits", type=int, default=4)
    p.add_argument("--first-last-bits", type=int, default=8)
    p.add_argument("--sparsity", nargs="+", type=float, default=[0.9], help="final sparsity for sparse model(s)")

    p.add_argument("--learned-channels", type=int, default=8)
    p.add_argument("--learned-kernel", type=int, default=3)
    p.add_argument("--learned-in-channels", type=int, default=3, choices=[1, 3])
    p.add_argument("--learned-free", action="store_true", help="drop the zero-mean (edge-only) constraint")
    p.add_argument("--canny-magnitude", action="store_true", help="canny outputs edges + NMS magnitude")

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--no-latency", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    return p


def run(**kwargs):
    """
    Run the experiment from Python. Keyword arguments mirror the CLI flags
    with dashes replaced by underscores, e.g.

        run(dataset="fake", epochs=1, subset=512)
        run(filters=["sobel", "learned"], models=["dense", "quant"], w_bits=[4, 2])
        run(models=["sparse"], sparsity=[0.8, 0.95], no_latency=True)

    Returns the list of result dicts (also written to `out`/results.csv/json).
    """
    p = build_parser()
    args = p.parse_args([])
    unknown = set(kwargs) - set(vars(args))
    if unknown:
        raise TypeError(f"unknown option(s) {sorted(unknown)}; valid: {sorted(vars(args))}")
    list_opts = {"filters", "models", "w_bits", "sparsity"}
    for k, v in kwargs.items():
        if k in list_opts and isinstance(v, (str, int, float)):
            v = [v]
        setattr(args, k, v)
    for name, choices in (("filters", FILTERS), ("models", MODELS)):
        bad = [v for v in getattr(args, name) if v not in choices]
        if bad:
            raise ValueError(f"{name}: {bad} not in {choices}")
    return _run(args)


def main(argv=None):
    return _run(build_parser().parse_args(argv))


def _run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}  torch={torch.__version__}")
    os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)

    train_data, test_data = (load_split(args.dataset, s, args.data_root, args.img_size, args.crop_roi, args.cache_dir)
                             for s in ("train", "test"))
    if args.subset:
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(train_data[0]), generator=g)[:args.subset]
        train_data = (train_data[0][idx], train_data[1][idx])
    print(f"[data] train {tuple(train_data[0].shape)}  test {tuple(test_data[0].shape)}")

    results, trained_front_ends = [], {}
    for fname in args.filters:
        for tag, kind, mkw in model_variants(args):
            torch.manual_seed(args.seed)
            name = f"{fname}/{tag}"
            fe = build_filter(fname, concat_input=args.concat_input, **filter_kwargs(fname, args))
            target = mkw.pop("target_sparsity", None)
            model = build_model(kind, fe, width=args.width, **mkw)
            if target is not None:
                model.target_sparsity = target

            print(f"\n=== {name}  ({sum(p.numel() for p in model.parameters()):,} params, "
                  f"filter out_channels={fe.out_channels}) ===")
            t0 = time.time()
            train(model, (train_data, test_data), args, device, log_prefix=f"[{name}]")
            train_time = time.time() - t0

            acc = evaluate(model, *test_data, device)
            n_params = sum(p.numel() for p in model.parameters())
            row = dict(filter=fname, model=tag, test_acc=round(acc, 4), params=n_params,
                       nnz_weights=sum(m.weight.count_nonzero().item() for m in model.weight_layers()),
                       storage_kb=round(model.storage_bits() / 8 / 1024, 1),
                       train_s=round(train_time, 1))
            if not args.no_latency:
                row["ms_per_img"] = round(latency_ms(model, args.img_size, device), 4)
            results.append(row)
            torch.save(model.state_dict(), os.path.join(args.out, "ckpt", name.replace("/", "__") + ".pt"))
            trained_front_ends.setdefault(fname, model.front_end)

            if kind == "sparse":
                # real CSR inference -- should match accuracy, latency is the point
                sm = copy.deepcopy(model).to_sparse().eval()
                s_acc = evaluate(sm, *test_data, device)
                srow = dict(row, model=tag + "-csr", test_acc=round(s_acc, 4))
                if not args.no_latency:
                    srow["ms_per_img"] = round(latency_ms(sm, args.img_size, device), 4)
                results.append(srow)

            print(f"[{name}] " + "  ".join(f"{k}={v}" for k, v in row.items() if k not in ("filter", "model")))

    # ---- report ----
    cols = ["filter", "model", "test_acc", "params", "nnz_weights", "storage_kb", "ms_per_img", "train_s"]
    cols = [c for c in cols if any(c in r for r in results)]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in results)) for c in cols}
    print("\n" + "  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in results:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(dict(args=vars(args), results=results), f, indent=2)
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)
    print(f"\n[done] results in {args.out}/")

    save_filter_viz(trained_front_ends, test_data[0], os.path.join(args.out, "filters.png"))
    return results


if __name__ == "__main__":
    main()
