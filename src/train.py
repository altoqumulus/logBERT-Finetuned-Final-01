import torch, torch.optim as optim, torch.nn as nn
from torch.utils.data import DataLoader
from dataset import WindowDataset, collate_mask_fn
from model   import LogBERT
from losses  import MLKPLoss, VHMLoss
from pathlib import Path
import pickle
import os
import matplotlib.pyplot as plt

from clearml import Task
import argparse

RUN_ID = os.getenv("RUN_ID", "manual")
task = Task.init(
    project_name="LogBERT-OpenStack/Pretraining",
    task_name=f"pretrain_run_{RUN_ID}", # replace dynamically
    tags=["pretrain", "run-set-1"]
)

logger = task.get_logger()


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1.  vocab size (§3 notes)
    vocab_size = cfg["vocab_size"] + 2

    model  = LogBERT(vocab_size).to(device)
    mlkp   = MLKPLoss()
    #vhm    = VHMLoss(embed_dim=128, alpha=cfg["alpha"])
    vhm = VHMLoss(embed_dim=128, alpha=cfg["alpha"]).to(device) 

    opt    = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)
    sched  = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    # 2.  DataLoaders
    train_ds = WindowDataset(cfg["train_pkl"], with_labels=False,
                             mask_ratio=cfg["mask_ratio"])
    val_ds   = WindowDataset(cfg["val_pkl"],   with_labels=False,
                             mask_ratio=cfg["mask_ratio"])

    train_dl = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True,
                          collate_fn=lambda b: collate_mask_fn(b, cfg["mask_ratio"]))
    val_dl   = DataLoader(val_ds,   batch_size=cfg["batch"], shuffle=False,
                          collate_fn=lambda b: collate_mask_fn(b, cfg["mask_ratio"]))

    best_val = float("inf")
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    train_losses, val_losses = [], [] 

    for epoch in range(cfg["epochs"]):
        model.train()
        total = 0
        for x, lbl in train_dl:
            x, lbl = x.to(device), lbl.to(device)
            opt.zero_grad()
            logits, h_dist = model(x)
            loss = mlkp(logits, lbl) + vhm(h_dist)
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()

        # ---- validation loss on normal sequences ----
        model.eval(); val_loss = 0
        with torch.no_grad():
            for x, lbl in val_dl:
                x, lbl = x.to(device), lbl.to(device)
                logits, h_dist = model(x)
                val_loss += (mlkp(logits, lbl) + vhm(h_dist)).item()
        val_loss /= len(val_dl)

        # store for plot
        train_losses.append(total / len(train_dl))
        val_losses.append(val_loss)

        # log to ClearML    
        logger.report_scalar(title="loss", series="train", iteration=epoch, value=total / len(train_dl))
        logger.report_scalar(title="loss", series="val", iteration=epoch, value=val_loss)

        print(f"[{epoch+1:02d}] train={total/len(train_dl):.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = ckpt_dir / f"best.pt"
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            task.upload_artifact(name="best_model", artifact_object=ckpt_path)
            print("  ↳ saved new best")
    
    
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
    parser = argparse.ArgumentParser(description="Train LogBERT model")
    parser.add_argument("--train_pkl", type=str, default="data/train.pkl", help="Path to training pickle file")
    parser.add_argument("--val_pkl", type=str, default="data/val.pkl", help="Path to validation pickle file")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha for VHMLoss")
    parser.add_argument("--mask_ratio", type=float, default=0.3, help="Mask ratio")

    args = parser.parse_args()

    cfg = {
        "train_pkl": args.train_pkl,
        "val_pkl": args.val_pkl,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "alpha": args.alpha,
        "mask_ratio": args.mask_ratio
    }
    
    task.connect(cfg) # connect config dict to ClearML

    seqs = pickle.load(open(cfg["train_pkl"], "rb"))
    cfg["vocab_size"] = max(max(s) for s in seqs) + 2   # +MASK

    print (f"[i] Training LogBERT with config: {cfg}")
    #import pdb;pdb.set_trace()
    train(cfg)
    
    