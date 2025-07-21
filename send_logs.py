#!/usr/bin/env python3
"""
send_logs.py  –  stream a logfile in 32-line batches to the
                 /score_batch endpoint and save every anomalous batch.

Server JSON contract (see updated infer.py):
{
  "is_anomaly": true|false,
  "reason":      "unseen_template" | "miss_ratio",
  "score":       1.0 | <miss_cnt / masked>,
  "cluster_id":  <int>  (only for unseen_template),
  "miss_count":  <int>, "masked": <int>, "g": <int>, "r": <int>
}

The client:
1. reads exactly 32 lines at a time;
2. POSTs them as JSON  {"lines": [line1,…,line32]};
3. if is_anomaly == true, writes the 32-line context to --out,
   preceded by a header summarising why it was flagged.
"""

import argparse, requests, json, sys, time
WINDOW = 32                           # must match server --window

# ---------- CLI ----------
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--log_file", required=True,
                   help="path to raw OpenStack log file")
    p.add_argument("--api", default="http://localhost:8000/score_batch",
                   help="inference endpoint")
    p.add_argument("--out", default="anomaly_batches.log",
                   help="file to save anomalous 32-line windows")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="seconds to pause between batches (simulate stream)")
    return p.parse_args()
# -------------------------

def main():
    a = cli()
    batch, start_line = [], 1

    with open(a.log_file, encoding="utf-8") as fin, \
         open(a.out, "w", encoding="utf-8") as fout:

        for ln, raw in enumerate(fin, 1):
            batch.append(raw.rstrip("\n"))
            if len(batch) < WINDOW:
                continue

            try:
                r = requests.post(a.api,
                                  headers={"Content-Type":"application/json"},
                                  data=json.dumps({"lines": batch}))
                r.raise_for_status()
                ans = r.json()
                #import pdb; pdb.set_trace()  # for debugging, remove in production
            except Exception as e:
                print(f"[ERR] batch {start_line}-{ln}: {e}", file=sys.stderr)
                break

            if ans.get("is_anomaly"):
                hdr_parts = [f"reason={ans['reason']}",
                             f"score={ans.get('score',1):.4f}"]
                if ans["reason"] == "unseen_template":
                    hdr_parts.append(f"cluster={ans['cluster_id']}")
                else:  # miss_ratio path
                    hdr_parts.append(f"miss={ans['miss_count']}/{ans['masked']}"
                                     f"  g={ans['g']} r={ans['r']}")
                header = " ".join(hdr_parts)

                print(f"[ALERT] lines {start_line}-{ln}  {header}")
                fout.write(f"# ---- anomaly lines {start_line}-{ln}  {header} ----\n")
                fout.write("\n".join(batch) + "\n\n")

            # slide to next batch
            batch, start_line = [], ln + 1
            if a.sleep:
                time.sleep(a.sleep)

    print("done ➜", a.out)

if __name__ == "__main__":
    main()
