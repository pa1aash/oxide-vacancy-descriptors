"""Execute one run spec end to end: train (or evaluate P), then write the result via write_result.

Layout under ``results_dir`` (``results/``, or ``results/smoke/`` for smoke runs):
    <run_id>.json                  write_result record; payload holds metrics and bookkeeping
    predictions/<run_id>.parquet   per-site test predictions (path and sha256 in the payload)
Checkpoints go to ``checkpoints/<run_id>.pt`` (gitignored); path and sha256 are in the payload.

Idempotent: a run whose result file exists and verifies (same run_id, prediction and checkpoint
hashes match) is skipped. A spec with ``code_ref`` (cluster specs) takes its run id from that pinned reference and
runs only on the code it names (``dftgnn.train.check_code_ref``); the payload then records both.
"""
from __future__ import annotations

import json
from pathlib import Path

from dftgnn.config import Config, load_config
from dftgnn.graphs.store import sha256_file
from dftgnn.hardware import hardware_info
from dftgnn.io.results import write_result
from dftgnn.train import (
    CKPT_DIR,
    REPO_ROOT,
    RunSpec,
    Store,
    check_code_ref,
    code_sha,
    evaluate_p,
    run_id,
    train_run,
)


def results_dir_for(spec: RunSpec, root: Path = REPO_ROOT) -> Path:
    return root / "results" / ("smoke" if spec.smoke else (spec.results_subdir or ""))


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def verify_result(path: Path, rid: str, root: Path = REPO_ROOT) -> bool:
    """True if ``path`` is the finished result of ``rid`` and its artefacts match their hashes."""
    if not path.is_file():
        return False
    try:
        pay = json.loads(path.read_text())["payload"]
    except (json.JSONDecodeError, KeyError):
        return False
    if pay.get("run_id") != rid:
        return False
    for key in ("predictions", "checkpoint"):
        art = pay.get(key)
        if art is None:
            continue
        f = Path(art["path"])
        f = f if f.is_absolute() else root / f
        if not f.is_file() or sha256_file(f) != art["sha256"]:
            return False
    return True


def execute(spec: RunSpec, data: Store, *, cfg: Config | None = None, results_dir: Path | None = None,
            ckpt_dir: Path = CKPT_DIR, device=None, allow_dirty: bool = False, log=print) -> dict:
    """Run ``spec`` unless already done. Returns {"run_id", "status", "result"}."""
    cfg = cfg if cfg is not None else load_config()
    code = code_sha()
    if spec.code_ref is not None:
        check_code_ref(spec.code_ref)
    rid = run_id(spec, spec.code_ref or code, data.manifest_sha)
    rdir = Path(results_dir) if results_dir is not None else results_dir_for(spec)
    out = rdir / f"{rid}.json"
    if verify_result(out, rid):
        return {"run_id": rid, "status": "skipped", "result": out}
    ckpt = ckpt_dir / f"{rid}.pt"
    if spec.model == "P":
        res = evaluate_p(spec, data, ckpt_dir, cfg, device)
        ckpt = None
    else:
        res = train_run(spec, data, cfg, device=device, ckpt_path=ckpt, log=log)
    frame = res.pop("predictions")
    pdir = rdir / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    ppath = pdir / f"{rid}.parquet"
    frame.to_parquet(ppath, index=False)
    payload = {
        "run_id": rid, "spec": spec.to_dict(), "smoke": spec.smoke, "code_sha": code,
        "graphs_manifest_sha256": data.manifest_sha, **({"code_ref": spec.code_ref} if spec.code_ref else {}),
        "hardware": hardware_info(device), **res,
        "predictions": {"path": _rel(ppath), "sha256": sha256_file(ppath), "rows": len(frame)},
        "checkpoint": None if ckpt is None else {"path": _rel(ckpt), "sha256": sha256_file(ckpt)},
    }
    if spec.smoke:
        payload["note"] = "smoke run: pipeline check only, excluded from every analysis"
    path = write_result(rid, payload, config=cfg, allow_dirty=allow_dirty, results_dir=rdir)
    return {"run_id": rid, "status": "done", "result": path}
