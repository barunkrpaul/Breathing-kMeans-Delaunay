#!/usr/bin/env bash
# Runs every experiment of the paper and rebuilds the tables and figures.
# Finished rows of the main protocol are skipped, so the script can be restarted.
set -e
cd "$(dirname "$0")"
python3 protocol.py "quake:0:64:0+1+2,quake:0:256:0+1+2,uber:50000:64:0+1+2,uber:50000:256:0+1+2,uber:200000:256:0+1+2,uber:50000:1024:0+1+2"
python3 pkg.py
python3 rswap.py
python3 make_outputs.py
