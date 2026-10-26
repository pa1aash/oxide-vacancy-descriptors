"""Data-free check of the job specifications in specs/ against the shipped splits, configurations and run records.

For every specs/<stage>.jsonl with its <stage>.manifest.json:

    file       sha256 and job count of the .jsonl equal the manifest
    splits     the manifest's splits_manifest_sha256 equals the sha256 of splits/MANIFEST.sha256, and every file
               listed in splits/MANIFEST.sha256 has the listed sha256
    graphs     graphs_manifest_sha256 equals the sha256 of data/graphs_v1_manifest.sha256 (the manifest of the graph
               store; the store itself is not in this repository)
    configs    the sha256 pinned for the tuned configuration (and the tuned P1 / tuned D file) equals the shipped file
    run ids    unique, and each equals the id recomputed from its spec: dftgnn.train.run_id(RunSpec.from_dict(spec),
               code_ref, graphs sha256) for training and evaluation jobs, dftgnn.infer.task_id(key) for the other jobs
    leakage    dftgnn.train.nrp.check_no_leakage (test hosts never among training or validation hosts), when it runs
               on the shipped split files alone; otherwise skipped with a note
    records    every run record results/**/<run id>.json of a job carries the job's spec (payload.spec, or the task
               key for jobs that have one)

Full regeneration of the spec files (scripts/queue/make_specs.py --check) also needs the processed inputs
(data/processed), which are not in this repository.

    python reproduce/check_specs.py
"""
from __future__ import annotations

import os
import sys

# reproduce/numbers.py is named like the standard-library module ``numbers``: keep this folder off the import path.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SPECS = ROOT / "specs"
RUN_ID = re.compile(r"^[0-9a-f]{16}$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_splits() -> tuple[bool, str]:
    """Every file listed in splits/MANIFEST.sha256 and splits/loco/MANIFEST.sha256 has its listed sha256."""
    bad, n, listed = [], 0, set()
    for man in (ROOT / "splits" / "MANIFEST.sha256", ROOT / "splits" / "loco" / "MANIFEST.sha256"):
        if not man.is_file():
            continue
        for line in man.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            h, name = line.split(maxsplit=1)
            n += 1
            p = man.parent / name.strip()
            listed.add(p.resolve())
            if not p.is_file() or sha256(p) != h:
                bad.append(name)
    unlisted = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "splits").rglob("*.json") if p.resolve() not in listed)
    note = f"{n} split files listed, {len(bad)} differ" + (f": {bad[:3]}" if bad else "")
    if unlisted:
        note += f"; not listed (no hash pin): {', '.join(unlisted)}"
    return not bad, note


def run_records() -> dict[str, list[Path]]:
    """Run id -> run record files under results/ (prediction tables and other files are ignored)."""
    out: dict[str, list[Path]] = {}
    for p in (ROOT / "results").rglob("*.json"):
        if RUN_ID.match(p.stem) and "predictions" not in p.parts:
            out.setdefault(p.stem, []).append(p)
    return out


def pinned_files(man: dict) -> list[tuple[str, str, str]]:
    """(label, path, sha256) of every file a manifest pins by hash."""
    out = []
    for k in ("tuned", "tuned_p1", "tuned_d", "descriptor_taxonomy", "hosts_file"):
        v = man.get(k)
        if isinstance(v, dict) and "path" in v:
            out.append((k, v["path"], v["sha256"]))
    return out


def check_stage(jsonl: Path, graphs_sha: str, splits_sha: str, records: dict, cfg) -> dict:
    from dftgnn.infer import task_id
    from dftgnn.train import RunSpec, nrp, run_id

    stage = jsonl.name.removesuffix(".jsonl")
    man = json.loads(jsonl.with_name(f"{stage}.manifest.json").read_text())
    body = jsonl.read_bytes()
    jobs = [json.loads(ln) for ln in body.decode().splitlines() if ln]
    row = {"stage": stage, "jobs": len(jobs), "notes": []}
    row["file"] = sha256(jsonl) == man["sha256"] and len(jobs) == man["count"]
    row["splits"] = man.get("splits_manifest_sha256") == splits_sha
    row["graphs"] = man.get("graphs_manifest_sha256") == graphs_sha
    ok_cfg = True
    for label, rel, h in pinned_files(man):
        p = ROOT / rel
        if not p.is_file():
            row["notes"].append(f"{label} {rel} is not in this repository (sha256 not checked)")
            continue
        ok_cfg &= sha256(p) == h
    row["configs"] = ok_cfg

    ids = [j["run_id"] for j in jobs]
    bad_ids = []
    for j in jobs:
        if "key" in j:
            got = task_id(j["key"])
            gs = j["key"].get("graphs_manifest_sha256")
            if gs is not None and gs != graphs_sha:
                bad_ids.append(j["run_id"])
        else:
            got = run_id(RunSpec.from_dict(j["spec"]), j["code_ref"], graphs_sha)
        if got != j["run_id"] or j.get("code_ref") != man["code_ref"]:
            bad_ids.append(j["run_id"])
    row["run_ids"] = len(set(ids)) == len(ids) and not bad_ids

    try:
        for j in jobs:
            nrp.check_no_leakage(j, cfg)
        row["leakage"] = True
    except (FileNotFoundError, OSError, KeyError) as e:
        row["leakage"] = None
        row["notes"].append(f"leakage check skipped: needs files not in this repository ({type(e).__name__}: "
                            f"{str(e)[:80]})")
    except AssertionError as e:
        row["leakage"] = False
        row["notes"].append(str(e))

    n_rec = n_same = n_diff = 0
    for j in jobs:
        for p in records.get(j["run_id"], []):
            pl = json.loads(p.read_text()).get("payload", {})
            if not isinstance(pl, dict) or ("spec" not in pl and "key" not in pl):
                continue
            n_rec += 1
            same = pl["spec"] == j["spec"] if "spec" in pl else pl["key"] == j.get("key")
            n_same += same
            n_diff += not same
    row["records"] = (n_rec, n_same)
    row["records_ok"] = n_diff == 0
    return row


def main() -> int:
    from dftgnn.config import load_config

    cfg = load_config()
    graphs_sha = sha256(ROOT / "data" / "graphs_v1_manifest.sha256")
    splits_sha = sha256(ROOT / "splits" / "MANIFEST.sha256")
    splits_ok, splits_note = check_splits()
    records = run_records()
    rows = [check_stage(p, graphs_sha, splits_sha, records, cfg) for p in sorted(SPECS.glob("*.jsonl"))]

    yn = lambda v: "skip" if v is None else ("ok" if v else "FAIL")
    head = ("stage", "jobs", "file", "splits", "graphs", "configs", "run ids", "leakage", "records (same spec)")
    body = [(r["stage"], str(r["jobs"]), yn(r["file"]), yn(r["splits"]), yn(r["graphs"]), yn(r["configs"]),
             yn(r["run_ids"]), yn(r["leakage"]),
             f"{r['records'][1]}/{r['records'][0]}" + ("" if r["records_ok"] else " FAIL")) for r in rows]
    w = [max(len(x[i]) for x in [head, *body]) for i in range(len(head))]
    line = lambda x: " | ".join(s.ljust(w[i]) for i, s in enumerate(x)).rstrip()
    print(line(head))
    print("-|-".join("-" * n for n in w))
    for b in body:
        print(line(b))
    print(f"\nsplit manifests: {splits_note}")
    print(f"data/graphs_v1_manifest.sha256: sha256 {graphs_sha}")
    for r in rows:
        for n in dict.fromkeys(r["notes"]):
            print(f"note, {r['stage']}: {n}")
    fails = [r["stage"] for r in rows if not (r["file"] and r["splits"] and r["graphs"] and r["configs"]
                                               and r["run_ids"] and r["leakage"] is not False and r["records_ok"])]
    n_jobs = sum(r["jobs"] for r in rows)
    print(f"{len(rows)} spec files, {n_jobs} jobs; " + ("all checks pass" if not fails and splits_ok
                                                          else f"FAILED: {fails or 'splits'}"))
    print("Full regeneration (python scripts/queue/make_specs.py --check) also needs data/processed.")
    return 1 if fails or not splits_ok else 0


if __name__ == "__main__":
    sys.exit(main())
