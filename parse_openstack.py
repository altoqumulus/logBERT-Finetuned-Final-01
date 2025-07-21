#!/usr/bin/env python3
"""
parse_openstack.py
------------------
Drain‑based parser for OpenStack logs.

Examples
--------
# Parse a single raw log and persist state
python parse_openstack.py \
    --input  openstack_raw.log \
    --output parsed_logs.csv \
    --state  drain_state.bin

# Incrementally add more logs using existing state
python parse_openstack.py \
    --input  more_raw.log other.log \
    --output parsed_more.csv \
    --state  drain_state.bin
"""
import argparse, json, os, sys, time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig


def build_miner(state_path: Path) -> TemplateMiner:
    config = TemplateMinerConfig()
    config.drain_depth             = 5
    config.drain_similar_threshold = 0.4
    config.drain_extra_delimiters  = r'=\(\)"\']|,'
    
    persistence = FilePersistence(str(state_path))
    miner       = TemplateMiner(persistence, config=config)
    
    # --- status message that works on every Drain3 version ---
    if state_path.exists():
        # clusters_counter (0.9+)  ||  fallback to len(list)
        num = getattr(miner.drain, "clusters_counter", len(miner.drain.clusters))
        print(f"[i] Loaded existing Drain state with {num} clusters")
    else:
        print("[i] Starting with a fresh Drain state")
    
    return miner



def parse_file(miner: TemplateMiner, path: Path, rows: list):
    """
    Feed a single file into Drain and append parsed rows to `rows`.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc=path.name, unit="line"):
            line = line.rstrip("\n")
            if not line:
                continue
            result = miner.add_log_message(line)
            if not result:               # safety check
                continue

            rows.append({
                "LogId":      line,                     # original text
                "cluster_id": result["cluster_id"],     # Drain template id
                "template":   result["template_mined"], # mined template
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  nargs="+", required=True,
                    help="Path(s) to raw *.log or *.txt files")
    ap.add_argument("--output", default="parsed_logs.csv",
                    help="CSV destination (default: parsed_logs.csv)")
    ap.add_argument("--state",  default="drain_state.bin",
                    help="File to load/save Drain state (default: drain_state.bin)")
    args = ap.parse_args()

    state_path = Path(args.state)
    miner      = build_miner(state_path)

    parsed_rows = []
    for file_path in map(Path, args.input):
        if not file_path.exists():
            sys.exit(f"[!] File not found: {file_path}")
        parse_file(miner, file_path, parsed_rows)

    # Persist Drain clusters for later reuse
    #miner.drain.persist_state(FilePersistence(str(state_path)))
    miner.save_state("end_of_parse")
    print(f"[+] Persisted Drain state to {state_path}")

    # Dump parsed rows to CSV
    df = pd.DataFrame(parsed_rows)
    df.to_csv(args.output, index=False)
    print(f"[+] Wrote {len(df):,} parsed rows → {args.output}")


if __name__ == "__main__":
    main()
