#!/usr/bin/env python3
"""
finetune.py
-----------
Resume LogBERT from a saved checkpoint and continue training
with a new (typically lower) learning rate schedule.
"""
import argparse, pickle, torch, random, numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import WindowDataset, collate_mask_fn
from model   import LogBERT
from losses  import MLKPLoss, VHMLoss
import os

from clearml import Task
import matplotlib.pyplot as plt

RUN_ID = os.getenv("RUN_ID", "manual")
task = Task.init(
    project_name="LogBERT-OpenStack/FineTuning",
    task_name=f"finetune_run_{RUN_ID}",  # replace dynamically
    tags=["finetune", "run-set-1"]
)

logger = task.get_logger()

# reproducibility helper
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt from base training")
    ap.add_argument("--train_pkl", required=True)
    ap.add_argument("--val_pkl",   required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr",     type=float, default=5e-5)
    ap.add_argument("--alpha",  type=float, default=0.1)
    ap.add_argument("--mask_ratio", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=256)

    # schedule controls
    ap.add_argument("--schedule", choices=["none", "step", "cosine"], default="none")
    ap.add_argument("--step_size", type=int, default=5)
    ap.add_argument("--gamma",     type=float, default=0.5)
    ap.add_argument("--eta_min",   type=float, default=2e-6)  # for cosine
    ap.add_argument("--embed_dim", type=int, default=128)     # must match ckpt
    ap.add_argument("--seed", type=int, default=42)

    return ap.parse_args()


def get_vocab_size(pkl_path):
    seqs = pickle.load(open(pkl_path, "rb"))
    if isinstance(seqs[0], tuple):  # (seq,label)
        seqs = [s for s,_ in seqs]
    max_id = max(max(s) for s in seqs)
    return max_id + 1

def main():
    args = parse_args(); set_seed(args.seed)

    cfg = {
        "ckpt": args.ckpt,
        "train_pkl": args.train_pkl,
        "val_pkl": args.val_pkl,
        "epochs": args.epochs,
        "lr": args.lr,
        "alpha": args.alpha,
        "mask_ratio": args.mask_ratio,
        "batch": args.batch,
        "schedule": args.schedule,
        "step_size": args.step_size,
        "gamma": args.gamma,
        "eta_min": args.eta_min,
        "embed_dim": args.embed_dim,
        "seed": args.seed
    }
    task.connect(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load checkpoint first ---
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # infer vocab size & embed dim from checkpoint
    vocab_size_ckpt = state["token_emb.weight"].shape[0]
    embed_dim_ckpt  = state["token_emb.weight"].shape[1]
    if embed_dim_ckpt != args.embed_dim:
        print(f"[warn] embed_dim mismatch: ckpt={embed_dim_ckpt}, arg={args.embed_dim}; using ckpt value.")
        args.embed_dim = embed_dim_ckpt

    # sanity check data vocab
    vocab_size_data = get_vocab_size(args.train_pkl)
    if vocab_size_data > vocab_size_ckpt:
        print(f"[warn] data has unseen ids up to {vocab_size_data-1}; "
              f"checkpoint vocab {vocab_size_ckpt-1}. "
              f"MASK token will be capped to checkpoint vocab.")

    # --- build & load model ---
    model = LogBERT(vocab_size=vocab_size_ckpt, embed_dim=args.embed_dim).to(device)
    model.load_state_dict(state, strict=True)
    print(f"[i] Loaded checkpoint '{args.ckpt}' (vocab={vocab_size_ckpt}).")

    # --- losses ---
    mlkp = MLKPLoss().to(device)
    vhm  = VHMLoss(embed_dim=args.embed_dim, alpha=args.alpha).to(device)
    if isinstance(ckpt, dict) and "vhm" in ckpt:
        vhm.load_state_dict(ckpt["vhm"])

    # --- data ---
    train_ds = WindowDataset(args.train_pkl, with_labels=False, mask_ratio=args.mask_ratio)
    val_ds   = WindowDataset(args.val_pkl,   with_labels=False, mask_ratio=args.mask_ratio)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=lambda b: collate_mask_fn(b, args.mask_ratio))
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          collate_fn=lambda b: collate_mask_fn(b, args.mask_ratio))

    # --- optimizer & scheduler ---
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    if args.schedule == "none":
        sched = None
    elif args.schedule == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)
    elif args.schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.eta_min)

    best_val = float("inf")
    train_losses, val_losses = [], [] 
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for x, lbl in train_dl:
            x, lbl = x.to(device), lbl.to(device)
            opt.zero_grad()
            logits, h_dist = model(x)
            loss = mlkp(logits, lbl) + vhm(h_dist)
            loss.backward(); opt.step()
            total += loss.item()
        if sched: sched.step()

        # validation
        model.eval(); vtot = 0.0
        with torch.no_grad():
            for x, lbl in val_dl:
                x, lbl = x.to(device), lbl.to(device)
                logits, h_dist = model(x)
                vtot += (mlkp(logits, lbl) + vhm(h_dist)).item()
        vavg = vtot / len(val_dl)
        
        train_losses.append(total / len(train_dl))
        val_losses.append(vavg)

        # log to ClearML    
        logger.report_scalar(title="loss", series="train", iteration=epoch, value=total / len(train_dl))
        logger.report_scalar(title="loss", series="val", iteration=epoch, value=vavg)

        print(f"[FT{epoch+1:02d}] train={total/len(train_dl):.4f}  val={vavg:.4f} (lr={opt.param_groups[0]['lr']:.2e})")

        if vavg < best_val:
            best_val = vavg
            out = "checkpoints/best_ft.pt"
            torch.save({"model": model.state_dict(),
                        "vhm":   vhm.state_dict(),
                        "cfg":   vars(args)}, out)
            print("  ↳ saved new fine‑tuned best:", out)
        
    
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(range(cfg["epochs"]), train_losses, label="train")
    ax.plot(range(cfg["epochs"]), val_losses,   label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Loss curves")
    ax.legend(loc="upper right")
    
    # ...existing code...
    logger.report_matplotlib_figure(
    title="loss_curves",
    series="train_vs_val",
    iteration=epoch,          # or cfg["epochs"]
    figure=fig                # a matplotlib.figure.Figure
    )
    plt.close(fig)          # free memory

if __name__ == "__main__":
    main()
