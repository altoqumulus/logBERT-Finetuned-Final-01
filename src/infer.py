from fastapi import FastAPI
import uvicorn, torch, pickle
from model import LogBERT
from drain3 import TemplateMiner; from pathlib import Path

app = FastAPI()
miner = TemplateMiner(persistence_path="drain_state.bin")
model = LogBERT(vocab_size=...).load_state_dict(torch.load("checkpoints/best.pt"))
model.eval().cuda()

WINDOW, DIST = 32, 0
current_ids = []            # slide window across calls

@app.post("/score")
def score(log_line: str):
    rid = miner.add_log_message(log_line)["cluster_id"] + 1
    current_ids.append(rid)
    if len(current_ids) < WINDOW:
        return {"ready": False}

    seq = [DIST] + current_ids[-WINDOW:]
    # inference
    with torch.no_grad():
        x = torch.tensor([seq]).cuda()
        logits, _ = model(x)
        score = anomaly_score(model, x).item()
    return {"ready": True, "score": score, "is_anomaly": score >= 0.5}
