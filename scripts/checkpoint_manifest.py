"""Write results/checkpoint_manifest.json: size and sha256 of every file in checkpoints/ (gitignored), cross-checked
against the checkpoint hash recorded in the run's result JSON.

    python scripts/checkpoint_manifest.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    recorded = {}
    for f in ROOT.glob("results/**/*.json"):
        try:
            pay = json.loads(f.read_text()).get("payload", {})
        except (ValueError, AttributeError):
            continue
        ck = pay.get("checkpoint") if isinstance(pay, dict) else None
        if isinstance(ck, dict):
            recorded[Path(ck["path"]).name] = ck["sha256"]
    files, bad = {}, []
    for p in sorted((ROOT / "checkpoints").glob("*.pt")):
        s = sha(p)
        files[p.name] = {"bytes": p.stat().st_size, "sha256": s}
        if p.name in recorded and recorded[p.name] != s:
            bad.append(p.name)
    missing = sorted(set(recorded) - set(files))
    out = {"n_files": len(files), "total_bytes": sum(v["bytes"] for v in files.values()),
           "mismatch_with_result_json": bad, "recorded_but_missing": missing, "files": files}
    (ROOT / "results" / "checkpoint_manifest.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"{len(files)} checkpoints, {out['total_bytes'] / 1e6:.0f} MB, {len(bad)} mismatches, {len(missing)} missing")


if __name__ == "__main__":
    main()
