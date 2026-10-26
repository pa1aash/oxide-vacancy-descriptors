"""Queue status: counts per state, ETA from the mean wall time of done jobs, failures with tracebacks."""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import QUEUE

from dftgnn import jobqueue as Q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=str(QUEUE))
    ap.add_argument("--requeue-stale", action="store_true")
    a = ap.parse_args()
    root = Path(a.queue)
    if a.requeue_stale:
        print("requeued:", Q.requeue_stale(root))
    s = Q.status(root)
    print("  ".join(f"{k} {v}" for k, v in s["counts"].items()))
    if s["mean_wall_time_s"] is not None:
        print(f"mean wall time {s['mean_wall_time_s']:.1f} s; ETA {s['eta_s'] / 60:.1f} min "
              f"at concurrency {s['concurrency_assumed']}")
    for f in s["failures"]:
        print(f"\nFAILED {f['run_id']} ({f['model']})\n{f['traceback']}")


if __name__ == "__main__":
    main()
