"""Train and evaluate a CubeLearn model on the HAR dataset.

Upstream train.py has no test loop at all -- it does train/validation only, and
selects the model by editing the source. This adds both test sets and a CLI.

The DFT-vs-CubeLearn comparison that the paper is built on is one flag:

    --lpp-lr 0       transform layers frozen at DFT init  -> "DFT" baseline
    --lpp-lr 1e-4    transform layers trained             -> "CubeLearn"

Everything else follows the paper: Adam, 30 epochs, lr 3e-4, best weights chosen
by validation accuracy with validation loss as the tie-break.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from har_dataset import ACTIVITY_NAMES, HARDataset
from network_har import CROPPED, LPP_MODULES, MODELS


def evaluate(net, loader, criterion, device, n_classes):
    net.eval()
    total = correct = 0
    loss_sum = 0.0
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            output = net(data)
            loss_sum += criterion(output, label).item() * len(label)
            pred = torch.argmax(output, 1)
            correct += (pred == label).sum().item()
            total += len(label)
            for t, p in zip(label.cpu().numpy(), pred.cpu().numpy()):
                confusion[t, p] += 1
    return correct / total, loss_sum / total, confusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dat_2dcnn_lstm", choices=sorted(MODELS))
    p.add_argument("--lpp-lr", type=float, required=True,
                   help="0 for the DFT baseline, 1e-4 for CubeLearn")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache-dir", default="/mnt/weka/fgeikyan/cubelearn_data/har_cache")
    p.add_argument("--split-dir", default="dataset_split")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arm = "DFT (frozen)" if args.lpp_lr == 0 else f"CubeLearn (lpp_lr={args.lpp_lr})"
    print(f"model={args.model}  arm={arm}  seed={args.seed}  device={device}", flush=True)

    crop = args.model in CROPPED
    n_classes = len(ACTIVITY_NAMES)
    loaders = {}
    for split in ("train", "val", "test_in", "test_out"):
        ds = HARDataset(args.cache_dir, args.split_dir, split, crop=crop)
        loaders[split] = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=(split == "train"),
            num_workers=args.workers, pin_memory=True)
        print(f"  {split:9s} {len(ds):5d} samples", flush=True)

    net = MODELS[args.model](n_classes=n_classes).to(device)

    lpp_params = [q for name in LPP_MODULES if hasattr(net, name)
                  for q in getattr(net, name).parameters()]
    lpp_ids = {id(q) for q in lpp_params}
    base_params = [q for q in net.parameters() if id(q) not in lpp_ids]
    print(f"  params: {sum(q.numel() for q in base_params)} classifier + "
          f"{sum(q.numel() for q in lpp_params)} transform", flush=True)

    optimizer = optim.Adam([
        {"params": base_params, "lr": args.lr},
        {"params": lpp_params, "lr": args.lpp_lr},
    ])
    criterion = nn.CrossEntropyLoss()

    ckpt_path = os.path.join(args.output_dir, "best.pth")
    best_acc, best_loss, best_epoch = -1.0, float("inf"), -1
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        net.train()
        total = correct = 0
        loss_sum = 0.0
        for data, label in loaders["train"]:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(label)
            correct += (torch.argmax(output, 1) == label).sum().item()
            total += len(label)
        train_acc, train_loss = correct / total, loss_sum / total

        val_acc, val_loss, _ = evaluate(net, loaders["val"], criterion, device, n_classes)
        # paper's rule: best validation accuracy, ties broken by lower loss
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
            torch.save({"epoch": epoch, "model_state_dict": net.state_dict(),
                        "val_acc": val_acc, "val_loss": val_loss}, ckpt_path)
            saved = " *saved"
        else:
            saved = ""
        history.append({"epoch": epoch, "train_acc": train_acc, "train_loss": train_loss,
                        "val_acc": val_acc, "val_loss": val_loss})
        print(f"[{epoch + 1:3d}/{args.epochs}] train {train_acc:.4f} ({train_loss:.4f})  "
              f"val {val_acc:.4f} ({val_loss:.4f}){saved}  {time.time() - t0:.0f}s", flush=True)

    net.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    print(f"\nbest epoch {best_epoch + 1}, val acc {best_acc:.4f}", flush=True)

    results = {"model": args.model, "lpp_lr": args.lpp_lr, "arm": arm, "seed": args.seed,
               "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
               "best_epoch": best_epoch + 1, "val_acc": best_acc,
               "train_seconds": round(time.time() - t0, 1), "history": history}
    for split in ("test_in", "test_out"):
        acc, loss, confusion = evaluate(net, loaders[split], criterion, device, n_classes)
        results[split] = {"acc": acc, "loss": loss, "confusion": confusion.tolist()}
        print(f"{split:9s} acc {acc * 100:.2f}%  loss {loss:.4f}", flush=True)
        per_class = confusion.diagonal() / confusion.sum(axis=1)
        for i, name in enumerate(ACTIVITY_NAMES):
            print(f"    {per_class[i] * 100:6.2f}%  {name}", flush=True)

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.output_dir}/results.json", flush=True)


if __name__ == "__main__":
    main()
