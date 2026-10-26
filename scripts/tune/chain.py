"""Unattended scheduler for the twelve tuning studies (runs ON the pod, outside the git checkout).

    nohup python /workspace/chain/chain.py >> /workspace/tuning/chain.log 2>&1 &

Operational only: it changes when studies start, never how they run. Every ``--poll`` seconds it counts the
running ``run_study.py`` processes and, while fewer than ``--max-running`` (4) run, starts the next study of
the fixed order a654, a200, a50 x (S, D-state, D-late, P1) with the wave-1 command line (``run_study.py
--model <m> --anchor <a> --device cuda``, detached, appending to ``<storage>/logs/<m>_a<a>.log``). A study is
skipped if it already has the configured number of COMPLETE trials (read from its SQLite file) or if a process
for it is running; studies started by hand are found by their process. A study that has been started
``--max-launches`` times by the chain without completing is given up and reported.

Budget guard. Before each start the total tuning GPU-hours are projected as the elapsed wall time since
``<storage>/wave1_start`` (one GPU, billed while the studies run) plus, for every unfinished trial, the GPU
time per trial of its anchor: the mean trial wall time measured at that anchor divided by the number of
concurrent studies, or, for an anchor with no finished trial yet, the measured a654 value scaled by the
cost table's upper-bound ratio for that anchor. If the projection exceeds ``--guard`` the chain writes
``<storage>/HALT`` with the reason and starts nothing more.

Each finished study (all trials complete and its ``results/tuning/tuning_<study>.json`` written) is packed
once into the checkout's gitignored ``outbox/`` with ``scripts/queue/outbox.py pack-files``, for
``pull_results.sh``. When all twelve studies are complete, packed and no study process is left, the chain
writes ``<storage>/ALL_DONE`` and exits. It never stops, kills or deletes anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

MODELS = ("S", "D-state", "D-late", "P1")
ANCHORS = (654, 200, 50)
ORDER = tuple((m, a) for a in ANCHORS for m in MODELS)
# cost-table upper-bound GPU-hours per anchor, used only as ratios
UPPER_GPU_H = {654: 32.60756791114305, 200: 10.979064128892567, 50: 2.3856209891444853}


def name(m: str, a: int) -> str:
    return f"{m}_a{a}"


def log(msg: str) -> None:
    print(f"{datetime.now(UTC).isoformat(timespec='seconds')} {msg}", flush=True)


# ---------------------------------------------------------------- probes (replaced by mocks in tests)

def running_studies() -> list[tuple[str, int]]:
    """(model, anchor) of every live run_study.py process that is not a dry run."""
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, check=False).stdout.splitlines()
    found = []
    for line in out:
        parts = line.split()
        if not any(p.endswith("scripts/tune/run_study.py") for p in parts) or "--dry-run" in parts:
            continue
        try:
            found.append((parts[parts.index("--model") + 1], int(parts[parts.index("--anchor") + 1])))
        except (ValueError, IndexError):
            continue
    return found


def complete_trials(storage: Path, study: str) -> int:
    db = storage / f"{study}.db"
    if not db.is_file():
        return 0
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    try:
        row = con.execute("SELECT COUNT(*) FROM trials t JOIN studies s USING(study_id) "
                          "WHERE s.study_name = ? AND t.state = 'COMPLETE'", (study,)).fetchone()
        return int(row[0])
    except sqlite3.OperationalError:
        return 0
    finally:
        con.close()


def trial_walls(checkout: Path, study: str) -> list[float]:
    f = checkout / "results" / "tuning" / f"trials_{study}.csv"
    if not f.is_file():
        return []
    import csv

    with open(f, newline="") as fh:
        return [float(r["wall_s"]) for r in csv.DictReader(fh) if r.get("wall_s")]


class Chain:
    def __init__(self, checkout: Path, storage: Path, python: str, *, guard: float, n_trials: int = 30,
                 max_running: int = 4, max_launches: int = 3, state: Path | None = None,
                 running=running_studies, complete=None, walls=None, launch=None, pack=None, now=time.time):
        self.checkout, self.storage, self.python = Path(checkout), Path(storage), python
        self.guard, self.n_trials, self.max_running, self.max_launches = guard, n_trials, max_running, max_launches
        self.state_path = Path(state) if state else self.storage / "chain_state.json"
        self.running = running
        self.complete = complete or (lambda s: complete_trials(self.storage, s))
        self.walls = walls or (lambda s: trial_walls(self.checkout, s))
        self.launch = launch or self._launch
        self.pack = pack or self._pack
        self.now = now
        self.state = (json.loads(self.state_path.read_text()) if self.state_path.is_file()
                      else {"launches": {}, "packed": []})

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True))
        os.replace(tmp, self.state_path)

    # ---- actions
    def _launch(self, m: str, a: int) -> None:
        logs = self.storage / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        with open(logs / f"{name(m, a)}.log", "a") as fh, open(os.devnull) as nul:
            subprocess.Popen([self.python, "scripts/tune/run_study.py", "--model", m, "--anchor", str(a),
                              "--device", "cuda"], cwd=self.checkout, stdout=fh, stderr=subprocess.STDOUT,
                             stdin=nul, start_new_session=True)

    def _pack(self, study: str) -> bool:
        rel = [f"results/tuning/trials_{study}.csv", f"results/tuning/tuning_{study}.json",
               f"results/tuning/optuna/{study}.db"]
        if not all((self.checkout / r).is_file() for r in rel):
            return False
        r = subprocess.run([self.python, "scripts/queue/outbox.py", "pack-files", "--name", f"tuning_{study}", *rel],
                           cwd=self.checkout, capture_output=True, text=True, check=False)
        log(f"pack {study}: exit {r.returncode} {r.stdout.strip()} {r.stderr.strip()[-300:]}")
        return r.returncode == 0

    # ---- budget guard
    def start_time(self) -> float | None:
        f = self.storage / "wave1_start"
        if not f.is_file():
            return None
        return datetime.fromisoformat(f.read_text().strip()).timestamp()

    def projection(self) -> dict:
        t0 = self.start_time()
        elapsed_h = 0.0 if t0 is None else max(0.0, self.now() - t0) / 3600
        per_trial = {}
        for a in ANCHORS:
            w = [x for m in MODELS for x in self.walls(name(m, a))]
            per_trial[a] = (sum(w) / len(w)) / 3600 / self.max_running if w else None
        if per_trial[654] is None:
            return {"elapsed_h": elapsed_h, "remaining_h": None, "total_h": None, "per_trial_gpu_h": per_trial}
        for a in ANCHORS:
            if per_trial[a] is None:
                per_trial[a] = per_trial[654] * UPPER_GPU_H[a] / UPPER_GPU_H[654]
        remaining = sum(max(0, self.n_trials - self.complete(name(m, a))) * per_trial[a] for m, a in ORDER)
        return {"elapsed_h": elapsed_h, "remaining_h": remaining, "total_h": elapsed_h + remaining,
                "per_trial_gpu_h": per_trial}

    def halt(self, reason: str) -> None:
        (self.storage / "HALT").write_text(reason + "\n")
        log(f"HALT: {reason}")

    # ---- one scheduling cycle; returns "halt", "done" or "wait"
    def step(self) -> str:
        if (self.storage / "HALT").exists():
            return "halt"
        done = {s: self.complete(s) >= self.n_trials for s in (name(m, a) for m, a in ORDER)}
        for s, ok in done.items():
            if ok and s not in self.state["packed"] and self.pack(s):
                self.state["packed"].append(s)
                self.save()
        run = set(self.running())
        if all(done.values()) and len(self.state["packed"]) == len(ORDER) and not (run & set(ORDER)):
            (self.storage / "ALL_DONE").write_text(f"{datetime.now(UTC).isoformat(timespec='seconds')}\n")
            log("ALL_DONE: 12 studies complete and packed")
            return "done"
        given_up = []
        for m, a in ORDER:
            if len(run) >= self.max_running:
                break
            s = name(m, a)
            if done[s] or (m, a) in run:
                continue
            if self.state["launches"].get(s, 0) >= self.max_launches:
                given_up.append(s)
                continue
            p = self.projection()
            if p["total_h"] is not None and p["total_h"] > self.guard:
                self.halt(f"projected tuning GPU-hours {p['total_h']:.2f} exceed the guard {self.guard:.2f} "
                          f"(elapsed {p['elapsed_h']:.2f} h, remaining {p['remaining_h']:.2f} h, per-trial GPU-h "
                          f"{json.dumps(p['per_trial_gpu_h'])}); not starting {s}")
                return "halt"
            self.launch(m, a)
            self.state["launches"][s] = self.state["launches"].get(s, 0) + 1
            self.save()
            run.add((m, a))
            tot = "n/a" if p["total_h"] is None else f"{p['total_h']:.2f}"
            log(f"launched {s} (launch {self.state['launches'][s]}); projected total {tot} GPU-h")
        if given_up and not (run & set(ORDER)):
            self.halt(f"studies {given_up} were started {self.max_launches} times without completing")
            return "halt"
        return "wait"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", default="/workspace/dft-gnn")
    ap.add_argument("--storage", default="/workspace/tuning")
    ap.add_argument("--python", default="/workspace/miniforge3/envs/dftgnn-gpu/bin/python")
    ap.add_argument("--guard", type=float, default=46.0, help="tune-stage upper bound, docs/cost_table.md")
    ap.add_argument("--poll", type=float, default=60.0)
    a = ap.parse_args()
    ch = Chain(Path(a.checkout), Path(a.storage), a.python, guard=a.guard)
    log(f"chain start (pid {os.getpid()}): guard {a.guard} GPU-h, running {sorted(ch.running())}")
    while True:
        try:
            r = ch.step()
        except Exception as exc:  # noqa: BLE001 - a transient error must not end the chain
            log(f"step error: {exc!r}")
            r = "wait"
        if r in ("halt", "done"):
            log(f"chain exit: {r}")
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
