#!/usr/bin/env python3
"""
merge_anomalies_into_test.py
----------------------------
After you've run:
    python parse_openstack.py --input abnormal.log --output abnormal.csv --state drain_state.bin

Execute this script to:
  • build windows from abnormal.csv
  • assign label 1 to each
  • merge with the normal-only test.pkl
  • save an updated, labeled test set

Usage
-----
python merge_anomalies_into_test.py \
    --abn_csv abnormal.csv \
    --test_pkl data/test.pkl \
    --window 32
"""
import argparse, pickle, random
from pathlib import Path
import pandas as pd

DIST = 0  # same special token you used before

def build_windows(ids, T, dist_token=0):
    wins = []
    for i in range(0, len(ids) - T + 1):
        wins.append([dist_token] + ids[i : i+T])
    return wins

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abn_csv", required=True, help="abnormal.csv from Drain")
    ap.add_argument("--test_pkl", required=True, help="existing test.pkl (normal-only)")
    ap.add_argument("--window", type=int, default=32)
    args = ap.parse_args()

    # 1. Build anomaly windows -------------------------------------------------
    df_abn = pd.read_csv(args.abn_csv, usecols=["cluster_id"])
    ids_abn = (df_abn["cluster_id"].astype(int) + 1).tolist()  # +1 for DIST=0
    abn_windows = build_windows(ids_abn, args.window, DIST)
    print(f"[i] Created {len(abn_windows):,} anomaly windows")

    # 2. Load normal test set --------------------------------------------------
    test_set = pickle.load(open(args.test_pkl, "rb"))
    # If your original test.pkl had NO labels yet, wrap normals with label 0
    if isinstance(test_set[0], list):
        test_set = [(seq, 0) for seq in test_set]
    
    

    # 3. Create labeled anomaly tuples ----------------------------------------
    abn_labeled = [(seq, 1) for seq in abn_windows]

    # 4. Merge & shuffle -------------------------------------------------------
    merged = test_set + abn_labeled
    random.shuffle(merged)

    
    # 5. Persist ---------------------------------------------------------------
    pickle.dump(merged, open(args.test_pkl, "wb"))
    print(f"[+] Merged test set saved to {args.test_pkl} "
          f"({len(merged):,} total windows; "
          f"{len(abn_labeled):,} anomalies)")

if __name__ == "__main__":
    main()
