#!/usr/bin/env python
"""Remove NVTX ranges broken by an nsys recording window from an exported nsys SQLite file.

With --duration, ranges still open when recording stops get an end timestamp before
their start (all set to the same instant). nsys stats then reports negative durations,
and nvtx_gpu_proj_sum fails with "rtree constraint failed: NVTX_EVENTS_RIDX.(startTS<=endTS)".
Dropping those rows (a handful per run) fixes both; complete ranges are untouched.

Usage:
    nsysFixRanges.py profiling/nsys/GA_MHc130_MA90_Combined.sqlite
"""

import argparse
import os
import sqlite3
import sys


def main():
    parser = argparse.ArgumentParser(description="Drop NVTX ranges with end < start from an nsys SQLite export")
    parser.add_argument("sqlite", help="SQLite file from 'nsys export --type sqlite'")
    args = parser.parse_args()
    if not os.path.isfile(args.sqlite):
        raise FileNotFoundError(f"SQLite file not found: {args.sqlite}")

    con = sqlite3.connect(args.sqlite)
    try:
        total = con.execute("SELECT count(*) FROM NVTX_EVENTS").fetchone()[0]
        broken = con.execute("SELECT count(*) FROM NVTX_EVENTS WHERE end IS NOT NULL AND end < start").fetchone()[0]
        con.execute("DELETE FROM NVTX_EVENTS WHERE end IS NOT NULL AND end < start")
        con.commit()
    finally:
        con.close()
    print(f"nsysFixRanges: removed {broken} of {total} NVTX ranges with end < start (cut by the recording window)")


if __name__ == "__main__":
    sys.exit(main())
