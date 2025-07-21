#!/usr/bin/env python3
"""
FastAPI inference service for LogBERT anomaly detection (batch of 32 lines).

Detection logic
---------------
1. Any Drain3 cluster_id >= vocab_size-1  ->  "unseen_template" anomaly.
2. Otherwise:
     • mask_ratio of positions are masked;
     • if miss_count (top-g misses) > r  ->  anomaly ("miss_ratio").

Author: Capstone-01
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import argparse, torch, uvicorn
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from .model import LogBERT

# ------------------------------ helpers --------------------------------
IGNORE_IDX = -100

@torch.no_grad()
def miss_count(model: LogBERT,
               seq: torch.Tensor,
               topk: int = 5,
               mask_ratio: float = 0.3):
    """Return (miss_count, num_masked) for one sequence [1,L] (DIST at pos0)."""
    L = seq.size(1)
    num_mask = max(1, int((L - 1) * mask_ratio))
    pos = torch.randperm(L - 1, device=seq.device)[:num_mask] + 1  # skip DIST
    labels = torch.full_like(seq, IGNORE_IDX)
    labels[0, pos] = seq[0, pos]
    seq[0, pos] = model.token_emb.num_embeddings - 1               # MASK id
    logits, _ = model(seq)
    top = logits.topk(k=topk, dim=-1).indices                      # [1,L,k]
    hit = top.eq(labels.unsqueeze(-1)).any(-1)                     # [1,L]
    miss = (~hit)[0, pos]
    return int(miss.sum()), num_mask
# ----------------------------------------------------------------------

class LogBatch(BaseModel):
    lines: List[str] = Field(..., min_items=32, max_items=32)


def build_app(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Drain3 -----------------
    miner = TemplateMiner(persistence_handler=FilePersistence(cfg.drain_state))
    miner.drain.add_catch_all_cluster = False      # freeze template tree
    # ---------------- Model ------------------
    model = LogBERT(vocab_size=cfg.vocab_size).to(device)
    state = torch.load(cfg.checkpoint, map_location=device)
    state = state.get("model", state)              # accept dict or raw
    model.load_state_dict(state, strict=True)
    model.eval()
    # ----------------------------------------
    DIST_ID = 0
    MASK_ID = cfg.vocab_size - 1

    app = FastAPI(title="LogBERT Batch Inference")

    @app.post("/score_batch")
    def score_batch(batch: LogBatch):
        ids = []
        for line in batch.lines:
            cid = miner.add_log_message(line)["cluster_id"]
            print (miner.add_log_message(line)["template_mined"])
            #import pdb; pdb.set_trace()  # for debugging, remove in production
            if cid >= MASK_ID:
                # unseen template  -> anomaly
                return {"is_anomaly": True,
                        "reason": "unseen_template",
                        "cluster_id": int(cid),
                        "score": 1.0}

            ids.append(cid + 1)  # shift by +1 so DIST remains 0

        seq = torch.tensor([[DIST_ID] + ids], dtype=torch.long, device=device)
        miss_cnt, num_mask = miss_count(model, seq,
                                        topk=cfg.topk,
                                        mask_ratio=cfg.mask_ratio)
        is_anom = miss_cnt > cfg.r
        return {"is_anomaly": bool(is_anom),
                "reason": "miss_ratio",
                "miss_count": miss_cnt,
                "masked": num_mask,
                "g": cfg.topk,
                "r": cfg.r,
                "score": round(miss_cnt / num_mask, 4)}

    @app.get("/")
    def root():
        return {"msg": "POST JSON {'lines': [32 raw lines]} to /score_batch"}

    return app


def cli():
    import argparse, uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--drain_state", required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--r", type=int, default=3)
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(args)  # optional
    app = build_app(args)  # must return FastAPI instance

    # ✅ THIS IS MISSING IN YOUR CODE
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    cfg = cli()
