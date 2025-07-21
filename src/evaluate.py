#!/usr/bin/env python3
"""
evaluate.py
-----------
End-to-end anomaly detection evaluation for fine-tuned LogBERT.

Computes miss-ratio anomaly scores on a labeled test set and reports
Precision / Recall / F1 at an automatically tuned threshold (or a user-specified one).

Usage:
  python src/evaluate.py \
      --ckpt checkpoints/best_ft.pt \
      --test_pkl data/test.pkl \
      --mask_ratio 0.3 \
      --topk 5 \
      --tune_frac 0.1   # use 10% of test to pick tau

"""
import argparse, pickle, random, torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from dataset import WindowDataset   # no masking collate; we mask inside scorer
from model   import LogBERT
from losses  import VHMLoss, MLKPLoss  # only if you want center stats

import os

from clearml import Task
import matplotlib.pyplot as plt

RUN_ID = os.getenv("RUN_ID", "manual")
task = Task.init(
    project_name="LogBERT-OpenStack/evaluate",
    task_name=f"evaluate_run_${RUN_ID}",  # replace dynamically
    tags=["evaluate", "run-set-1"]
)

logger = task.get_logger()

# --------------------
# Utilities
# --------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_vocab_size_from_ckpt(ckpt):
    state = ckpt["model"] if "model" in ckpt else ckpt
    return state["token_emb.weight"].shape[0]

def deterministic_mask(seqs, mask_ratio, rng):
    """
    Given a 2D LongTensor [B,L], return (masked, labels, mask_positions_bool).
    We never mask position 0 (DIST).
    labels = original token IDs where masked else -100.
    """
    B, L = seqs.shape
    num_can = L - 1
    k = max(1, int(num_can * mask_ratio))
    # mask same #positions per sequence; choose independently per row
    masked = seqs.clone()
    labels = torch.full_like(seqs, fill_value=-100)
    mask_bool = torch.zeros_like(seqs, dtype=torch.bool)
    for i in range(B):
        idxs = list(range(1, L))
        rng.shuffle(idxs)
        sel = idxs[:k]
        mask_bool[i, sel] = True
        labels[i, sel] = seqs[i, sel]
    return masked, labels, mask_bool

def mask_with_token(masked, mask_bool, mask_token):
    masked[mask_bool] = mask_token
    return masked

@torch.no_grad()
def batch_scores(model, batch, mask_ratio, topk, mask_token, device, rng):
    """
    Compute anomaly scores (miss_ratio) for a batch of sequences.
    """
    seqs = torch.stack(batch).to(device)  # [B,L]

    # build masked copies
    masked, labels, mask_bool = deterministic_mask(seqs, mask_ratio, rng)
    masked = mask_with_token(masked, mask_bool, mask_token)

    logits, _ = model(masked)             # [B,L,V]

    # gather top-k predictions per position
    topv, topi = torch.topk(logits, k=topk, dim=-1)  # [B,L,k]

    # true-in-topk?
    expanded = topi.eq(seqs.unsqueeze(-1))            # [B,L,k] bool
    hit = expanded.any(-1)                            # [B,L] bool
    # consider only masked positions
    miss = (~hit) & mask_bool.to(hit.device)
    miss_count = miss.sum(dim=1)                      # [B]
    denom = mask_bool.sum(dim=1).clamp(min=1)
    miss_ratio = miss_count.float() / denom.float()

    return miss_ratio.cpu()

def sweep_threshold(scores, labels, steps=101):
    """
    Return best (f1, tau, precision, recall).
    """
    best = (-1, None, None, None)
    for t in np.linspace(0, 1, steps):
        preds = (scores >= t).astype(int)
        p,r,f,_ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        if f > best[0]:
            best = (f, t, p, r)
    return best  # tuple

def evaluate(model, test_ds, mask_ratio, topk, mask_token, device,
             tune_frac=0.0, seed=42):
    """
    Full pipeline:
      1. Compute scores for all test items.
      2. If tune_frac>0: split off small slice, sweep τ on that slice, apply to rest.
         Else: sweep τ on full test set.
    """
    rng = random.Random(seed)
    loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        collate_fn=lambda b: list(zip(*b))  # returns (seqs,label_tensors)
    )

    all_scores = []
    all_labels = []
    for seqs, lbls in loader:
        seqs_list = list(seqs)                        # list of [L] tensors
        lbls_t    = torch.stack(lbls).cpu()           # [B]
        miss_ratio = batch_scores(model, seqs_list, mask_ratio, topk,
                                  mask_token, device, rng)  # Tensor [B]
        all_scores.append(miss_ratio.cpu())
        all_labels.append(lbls_t)

    scores_t = torch.cat(all_scores)                  # [N]
    labels_t = torch.cat(all_labels)                  # [N]

    import numpy as np
    scores = scores_t.numpy()
    labels = labels_t.numpy()

    # Tuning vs full evaluation
    if tune_frac > 0.0 and 0 < tune_frac < 1.0:
        n = len(scores)
        rng_idx = np.arange(n)
        rng.shuffle(rng_idx)
        k = max(1, int(n * tune_frac))
        tune_idx, eval_idx = rng_idx[:k], rng_idx[k:]

        best_f, tau, p, r = sweep_threshold(scores[tune_idx], labels[tune_idx])
        preds = (scores[eval_idx] >= tau).astype(int)
        p2,r2,f2,_ = precision_recall_fscore_support(labels[eval_idx], preds,
                                                     average="binary", zero_division=0)
        cm = confusion_matrix(labels[eval_idx], preds)



        return {
            "tau": tau,
            "tune_f1": best_f, "tune_precision": p, "tune_recall": r,
            "test_precision": p2, "test_recall": r2, "test_f1": f2,
            "confusion": cm, "scores": scores, "labels": labels,
            "eval_indices": eval_idx, "tune_indices": tune_idx
        }
    else:
        best_f, tau, p, r = sweep_threshold(scores, labels)
        preds = (scores >= tau).astype(int)
        cm = confusion_matrix(labels, preds)
        return {
            "tau": tau,
            "precision": p, "recall": r, "f1": best_f,
            "confusion": cm, "scores": scores, "labels": labels
        }

def log_confusion(logger, cm, tag="test", iter_=0):

    #cm_plot = [[cm[1][1], cm[0][1]], [cm[1][0], cm[0][0]]]
    cm_plot = [[cm[1][0], cm[1][1]], [cm[0][0], cm[0][1]]]
    #cm_plot = cm.tolist()  # convert to list for ClearML compatibility

    logger.report_confusion_matrix(
        title="confusion",
        series=tag,
        iteration=iter_,
        matrix=cm_plot,
        xaxis=["Normal", "Anomaly"],   # predicted labels
        yaxis=["Normal", "Anomaly"]    # true labels
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_pkl", required=True)
    ap.add_argument("--mask_ratio", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--tau", type=float, default=None, help="if set, skip sweep and use fixed threshold")
    ap.add_argument("--tune_frac", type=float, default=0.0, help="fraction of test to tune tau; overrides --tau when >0")
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = {
        "ckpt": args.ckpt,
        "test_pkl": args.test_pkl,
        "mask_ratio": args.mask_ratio,
        "topk": args.topk,
        "tau": args.tau,
        "tune_frac": args.tune_frac,
        "embed_dim": args.embed_dim,
        "seed": args.seed
    }
    task.connect(cfg)
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    test_ds = WindowDataset(args.test_pkl, with_labels=True)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    vocab_size = get_vocab_size_from_ckpt(ckpt)
    embed_dim_ckpt = state["token_emb.weight"].shape[1]
    if embed_dim_ckpt != args.embed_dim:
        print(f"[warn] embed_dim mismatch; using ckpt value {embed_dim_ckpt}")
        args.embed_dim = embed_dim_ckpt

    # Build & load model
    model = LogBERT(vocab_size=vocab_size, embed_dim=args.embed_dim).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Mask token = last id in vocab
    mask_token = vocab_size - 1

    # Evaluate
    if args.tau is not None and args.tune_frac == 0.0:
        # compute scores once, apply fixed tau
        results = evaluate(model, test_ds, args.mask_ratio, args.topk,
                           mask_token, device, tune_frac=0.0, seed=args.seed)
        # override tau
        tau = args.tau
        preds = (results["scores"] >= tau).astype(int)
        p,r,f,_ = precision_recall_fscore_support(results["labels"], preds,
                                                 average="binary", zero_division=0)
        cm = confusion_matrix(results["labels"], preds)
        print(f"[EvalFixed] τ={tau:.3f}  Precision={p:.4f} Recall={r:.4f} F1={f:.4f}")
        print(cm)
    else:
        results = evaluate(model, test_ds, args.mask_ratio, args.topk,
                           mask_token, device, tune_frac=args.tune_frac, seed=args.seed)
        if args.tune_frac > 0:
            print(f"[Tune] τ*={results['tau']:.3f}  tune_F1={results['tune_f1']:.4f}  "
                  f"P={results['tune_precision']:.4f} R={results['tune_recall']:.4f}")
            print(f"[Test] P={results['test_precision']:.4f}  R={results['test_recall']:.4f}  "
                  f"F1={results['test_f1']:.4f}")
            print(results["confusion"])
        else:
            print(f"[EvalSweepFullTest] τ*={results['tau']:.3f} "
                  f"P={results['precision']:.4f} R={results['recall']:.4f} F1={results['f1']:.4f}")
            print(results["confusion"])

    # ----------- log to ClearML -----------
    if args.tune_frac > 0:
            
            logger.report_scalar("tau",  "tuned",  results["tau"], 0)
            logger.report_scalar("F1",   "tune", results["tune_f1"], 0 )
            logger.report_scalar("Prec", "tune", results["tune_precision"], 0)
            logger.report_scalar("Rec",  "tune", results["tune_recall"], 0)
            logger.report_scalar("F1",   "test", results["test_f1"], 0)
            logger.report_scalar("Prec", "test", results["test_precision"], 0)
            logger.report_scalar("Rec",  "test", results["test_recall"], 0)
            log_confusion(logger, results["confusion"], tag="test", iter_=0)
    else:
            logger.report_scalar("tau",  "selected",results["tau"], 0 )
            logger.report_scalar("F1",   "test", results["f1"], 0)
            logger.report_scalar("Prec", "test", results["precision"], 0 )
            logger.report_scalar("Rec",  "test", results["recall"], 0 )
            log_confusion(logger, results["confusion"], tag="test", iter_=0)

if __name__ == "__main__":
    main()
