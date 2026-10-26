"""Write the job specs of one cluster stage.

    python scripts/queue/make_specs.py --stage cv --code-ref nrp-code-v1 --backbone v1 --out specs/cv.jsonl
    python scripts/queue/make_specs.py --stage cv --backbone v2                # specs/cv_v2.jsonl (configs/tuned_v2.yaml)
    python scripts/queue/make_specs.py --stage cv --check            # regenerate and compare, write nothing
    python scripts/queue/make_specs.py --stage moment_v2 --backbone v2         # specs/moment_v2.jsonl (v2 only)
    python scripts/queue/make_specs.py                 # list every available stage and backbone with its spec file
    python scripts/queue/make_specs.py --check         # --check of every available stage and backbone

Without ``--stage`` nothing is written. Stages whose input file is missing (p1v2b and pcompv2b before
``configs/tuned_p1_v2.yaml`` exists, dretrain_v2 before ``configs/tuned_d_v2.yaml``) are left out of both forms;
``--stage`` on one of them stops with the file name.

Writes ``specs/<stage>.jsonl`` (one queue job per line, sorted by priority; canonical JSON) and
``specs/<stage>.manifest.json`` (counts and the sha256 of the jsonl; no time stamp). Pure: no GPU, no network.
Run ids use ``--code-ref`` in place of the commit SHA. The count must equal the expected count of the stage, run ids
must be unique, and every job must pass the test-host leakage check, or nothing is written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT

from dftgnn.train import nrp


def generate(stage: str, code_ref: str, backbone: str, hosts_file: str | None, tuned: str, out: Path):
    hosts = nrp.read_hosts_file(Path(hosts_file)) if hosts_file else None
    jobs = sorted(nrp.build_stage(stage, code_ref=code_ref, backbone=backbone, hosts=hosts, tuned_path=Path(tuned)),
                  key=lambda j: j["priority"])
    ids = [j["run_id"] for j in jobs]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{stage}: {len(ids) - len(set(ids))} duplicate run ids")
    want = nrp.expected_count(stage, hosts)
    if len(jobs) != want:
        raise SystemExit(f"{stage}: {len(jobs)} jobs, expected {want}")
    for j in jobs:
        nrp.check_no_leakage(j)
    body = nrp.dumps_jsonl(jobs)
    man = nrp.manifest(stage, jobs, body, jsonl_name=out.name, code_ref=code_ref, backbone=backbone,
                       hosts_file=Path(hosts_file) if hosts_file else None, tuned_path=Path(tuned))
    return jobs, body, (json.dumps(man, indent=1, sort_keys=True) + "\n").encode()


def default_tuned(backbone: str) -> str:
    return str(nrp.TUNED_V2 if backbone == "v2" else ROOT / "configs" / "tuned_v1.yaml")


def all_targets(check: bool, code_ref: str) -> None:
    """No ``--stage``: list (or ``--check``) every (stage, backbone) of ``nrp.spec_targets``."""
    bad = 0
    for stage, backbone in nrp.spec_targets():
        out = nrp.SPECS_DIR / nrp.spec_name(stage, backbone)
        if not check:
            print(f"{stage:10s} {backbone}  {nrp.GROUP[stage]:4s} specs/{out.name}" + ("" if out.is_file() else " (absent)"))
            continue
        man_path = out.with_name(out.name.removesuffix(".jsonl") + ".manifest.json")
        _jobs, body, man = generate(stage, code_ref, backbone, None, default_tuned(backbone), out)
        ok = out.is_file() and man_path.is_file() and out.read_bytes() == body and man_path.read_bytes() == man
        bad += not ok
        print(f"{stage} {backbone} {json.loads(man)['count']} {json.loads(man)['sha256']} "
              + ("files match" if ok else "DIFFERS"), flush=True)
    if bad:
        raise SystemExit(f"{bad} spec files differ from regeneration")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=nrp.STAGES, help="default: every available stage (list, or --check)")
    ap.add_argument("--code-ref", default=nrp.CODE_REF)
    ap.add_argument("--backbone", choices=nrp.BACKBONES, help="default v1, or v2 for a v2-only stage")
    ap.add_argument("--hosts-file", help="relax / geomeval: host ids to relax (the top-up list), one per line")
    ap.add_argument("--tuned", help="default configs/tuned_v1.yaml (v1) or configs/tuned_v2.yaml (v2)")
    ap.add_argument("--out", help="default specs/<stage>.jsonl (specs/<stage>_v2.jsonl for --backbone v2, except for "
                                  "the v2-only stage moment_v2)")
    ap.add_argument("--check", action="store_true", help="regenerate and compare with the files; write nothing")
    a = ap.parse_args()
    if a.stage is None:
        if a.out or a.hosts_file or a.tuned:
            raise SystemExit("--out, --hosts-file and --tuned need --stage")
        return all_targets(a.check, a.code_ref)
    a.backbone = a.backbone or ("v2" if a.stage in nrp.V2_ONLY else "v1")
    out = Path(a.out) if a.out else nrp.SPECS_DIR / nrp.spec_name(a.stage, a.backbone)
    man_path = out.with_name(out.name.removesuffix(".jsonl") + ".manifest.json")
    tuned = a.tuned or default_tuned(a.backbone)
    try:
        _jobs, body, man = generate(a.stage, a.code_ref, a.backbone, a.hosts_file, tuned, out)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from None
    if a.check:
        if out.read_bytes() != body or man_path.read_bytes() != man:
            raise SystemExit(f"{out} or {man_path} differs from regeneration")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(body)
        man_path.write_bytes(man)
    m = json.loads(man)
    print(f"{a.stage} ({m['group']}, {m['label']}, backbone {a.backbone}): {m['count']} jobs "
          f"{m['counts_by_model']}; sha256 {m['sha256']}; code_ref {a.code_ref}"
          + ("; files match" if a.check else f"; wrote {out.name} and {man_path.name}"))


if __name__ == "__main__":
    main()
