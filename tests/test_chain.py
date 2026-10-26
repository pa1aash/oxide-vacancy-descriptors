"""Tests of the unattended study scheduler (scripts/tune/chain.py) with mocked processes and studies."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("chain", ROOT / "scripts" / "tune" / "chain.py")
chain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chain)


class Pod:
    """A mock pod: running study processes, COMPLETE counts, trial wall times, launches and packs."""

    def __init__(self, running=(), complete=None, walls=None):
        self.run = set(running)
        self.done = dict(complete or {})
        self.wall = dict(walls or {})
        self.launched, self.packed = [], []

    def kw(self):
        return {"running": lambda: sorted(self.run), "complete": lambda s: self.done.get(s, 0),
                "walls": lambda s: self.wall.get(s, []), "launch": self.do_launch, "pack": self.do_pack}

    def do_launch(self, m, a):
        assert (m, a) not in self.run, "double launch"
        self.run.add((m, a))
        self.launched.append(chain.name(m, a))

    def do_pack(self, s):
        self.packed.append(s)
        return True

    def finish(self, m, a):
        self.run.discard((m, a))
        self.done[chain.name(m, a)] = 30


def _chain(tmp_path, pod, guard=1e9, start="2026-10-06T21:38:14Z", now=None):
    (tmp_path / "wave1_start").write_text(start + "\n")
    return chain.Chain(tmp_path, tmp_path, "python", guard=guard, state=tmp_path / "state.json",
                       now=now or (lambda: chain.datetime.fromisoformat("2026-10-07T02:38:14+00:00").timestamp()),
                       **pod.kw())


A654 = [("S", 654), ("D-state", 654), ("D-late", 654), ("P1", 654)]


def test_existing_a654_processes_are_left_alone_and_order_is_fixed(tmp_path):
    pod = Pod(running=A654)
    ch = _chain(tmp_path, pod)
    assert ch.step() == "wait" and pod.launched == []          # four running: nothing to start
    pod.finish("D-late", 654)
    assert ch.step() == "wait" and pod.launched == ["S_a200"]
    assert pod.packed == ["D-late_a654"]
    for m, a in A654:
        pod.finish(m, a)
    ch.step()
    assert pod.launched == ["S_a200", "D-state_a200", "D-late_a200", "P1_a200"]
    for m in chain.MODELS:
        pod.finish(m, 200)
    ch.step()
    assert pod.launched[4:] == ["S_a50", "D-state_a50", "D-late_a50", "P1_a50"]


def test_no_double_launch_across_cycles(tmp_path):
    pod = Pod()
    ch = _chain(tmp_path, pod)
    for _ in range(5):
        ch.step()
    assert pod.launched == [chain.name(m, a) for m, a in A654]     # the mock asserts on any double launch


def test_resume_after_chain_kill_and_relaunch_of_a_dead_study(tmp_path):
    pod = Pod(running=A654[:3], complete={"P1_a654": 30})
    ch = _chain(tmp_path, pod)
    ch.step()
    assert pod.launched == ["S_a200"]
    # the chain is killed; S_a200's process dies with 7 trials done; a new chain resumes from its state file
    pod.run.discard(("S", 200))
    pod.done["S_a200"] = 7
    ch2 = _chain(tmp_path, pod)
    assert ch2.state["launches"] == {"S_a200": 1} and ch2.state["packed"] == ["P1_a654"]
    ch2.step()
    assert pod.launched == ["S_a200", "S_a200"] and ch2.state["launches"]["S_a200"] == 2
    assert pod.packed == ["P1_a654"]                               # packed once only


def test_gives_up_after_max_launches(tmp_path):
    pod = Pod(complete={chain.name(m, a): 30 for m, a in chain.ORDER if (m, a) != ("S", 50)})
    ch = _chain(tmp_path, pod)
    for _ in range(3):
        ch.step()
        pod.run.clear()                                            # each launch dies at once
    assert pod.launched == ["S_a50"] * 3
    assert ch.step() == "halt" and "S_a50" in (tmp_path / "HALT").read_text()


def test_guard_halts_before_launching(tmp_path):
    walls = {chain.name(m, 654): [1440.0] * 10 for m in chain.MODELS}   # 0.1 GPU-h per a654 trial
    pod = Pod(running=A654[1:], complete={chain.name(m, 654): 10 for m in chain.MODELS}, walls=walls)
    ch = _chain(tmp_path, pod, guard=10.0)                         # elapsed 5 h + remaining > 10
    p = ch.projection()
    assert abs(p["elapsed_h"] - 5.0) < 1e-9
    assert abs(p["per_trial_gpu_h"][654] - 0.1) < 1e-12
    assert abs(p["per_trial_gpu_h"][50] - 0.1 * chain.UPPER_GPU_H[50] / chain.UPPER_GPU_H[654]) < 1e-12
    assert ch.step() == "halt"
    assert (tmp_path / "HALT").exists() and pod.launched == []
    assert ch.step() == "halt" and pod.launched == []              # stays halted


def test_guard_passes_when_within_budget(tmp_path):
    walls = {chain.name(m, 654): [1440.0] for m in chain.MODELS}
    pod = Pod(running=A654[1:], walls=walls)
    ch = _chain(tmp_path, pod, guard=46.0)
    assert ch.step() == "wait" and pod.launched == ["S_a654"]


def test_all_done(tmp_path):
    pod = Pod(complete={chain.name(m, a): 30 for m, a in chain.ORDER})
    ch = _chain(tmp_path, pod)
    assert ch.step() == "done" and (tmp_path / "ALL_DONE").exists()
    assert sorted(pod.packed) == sorted(chain.name(m, a) for m, a in chain.ORDER) and pod.launched == []


def test_running_studies_parses_process_lines(monkeypatch):
    lines = ("/x/bin/python scripts/tune/run_study.py --model D-late --anchor 654 --device cuda\n"
             "/x/bin/python scripts/tune/run_study.py --model S --anchor 50 --dry-run --trials 2\n"
             "bash -c pgrep -f scripts/tune/run_study.py\n")

    class R:
        stdout = lines

    monkeypatch.setattr(chain.subprocess, "run", lambda *a, **k: R())
    assert chain.running_studies() == [("D-late", 654)]
