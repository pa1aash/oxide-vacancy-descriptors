"""Run (or resume) one tuning study until it holds 30 COMPLETE trials, then write its result.

    python scripts/tune/run_study.py --model S --anchor 654 --device cuda --threads 3
    python scripts/tune/run_study.py --model S --anchor 50 --dry-run --trials 2    # path check only

A study is one process; four run at once on one GPU (``scripts/tune/wave.sh``) and share the card through
the VRAM ledger in ``<storage-dir>/admit``. Trials append rows to
``results/tuning/trials_<model>_a<anchor>.csv``. A finished study copies its SQLite file to
``results/tuning/optuna/`` and writes ``write_result("tuning_<model>_a<anchor>")`` into ``results/tuning/``.

``--dry-run`` uses the study prefix ``dryrun_``, keeps storage and CSV under ``<storage-dir>_dryrun`` and
writes nothing into ``results/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["S", "D-state", "D-late", "P1"])
    ap.add_argument("--anchor", required=True, type=int)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0: CPU quota / 4)")
    ap.add_argument("--storage-dir", default="/workspace/tuning")
    ap.add_argument("--trials", type=int, default=None, help="COMPLETE trials to reach (default: config)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--v2", action="store_true",
                    help="tune the v2 backbone: config v2.selected's options, study names prefixed v2_")
    a = ap.parse_args()

    import torch

    from dftgnn.config import load_config
    from dftgnn.io.results import write_result
    from dftgnn.train import admission, available_cpus, code_sha
    from dftgnn.tune import (
        RESULTS_DIR,
        TuningStore,
        VramLedger,
        archive_study,
        run_study,
        storage_url,
        study_name,
        summarise,
    )

    cfg = load_config()
    if a.anchor not in cfg.tuning.anchors:
        raise SystemExit(f"anchor {a.anchor} is not one of {cfg.tuning.anchors}")
    if a.trials is not None and not a.dry_run:
        raise SystemExit("--trials is for dry runs only; real studies use the configured trial count")
    torch.set_num_threads(a.threads or max(1, int(available_cpus() // 4)))
    dev = torch.device(a.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but CUDA is unavailable")
    storage = Path(a.storage_dir + ("_dryrun" if a.dry_run else ""))
    csv_dir = storage / "csv" if a.dry_run else RESULTS_DIR
    if a.v2 and cfg.v2.selected is None:
        raise SystemExit("v2.selected is not set")
    arch = cfg.v2.variants[cfg.v2.selected] if a.v2 else None
    base = "v2_" if a.v2 else ""
    prefix = ("dryrun_" if a.dry_run else "") + base
    vram = torch.cuda.get_device_properties(0).total_memory / admission.GIB if dev.type == "cuda" else None
    table = admission.load_benchmark()
    ledger = VramLedger(Path(a.storage_dir) / "admit", vram)      # one ledger for real and dry studies
    commit = code_sha()
    data = TuningStore(resample=cfg.tuning.anchor_resample)
    print(f"{study_name(a.model, a.anchor, prefix)}: commit {commit}, device {dev}, threads "
          f"{torch.get_num_threads()}, {len(data.excluded_hosts)} test hosts ({data.n_excluded_sites} sites) "
          f"removed from memory, storage {storage_url(storage, study_name(a.model, a.anchor, prefix))}",
          flush=True)
    study = run_study(a.model, a.anchor, data, cfg=cfg, prefix=prefix, storage_dir=storage, csv_dir=csv_dir,
                      n_trials=a.trials, device=dev, ledger=ledger,
                      est_gb=lambda p: admission.est_peak_for_spec(table, p, int(p["batch_size"])),
                      pod_commit=commit, arch=arch, log=lambda m: print(m, flush=True))
    if a.dry_run:
        print("dry run finished; nothing written to results/", flush=True)
        return
    name = study_name(a.model, a.anchor, base)
    payload = summarise(study, a.model, a.anchor, cfg, arch=arch)
    payload["artefacts"] = archive_study(storage / f"{name}.db", csv_dir / f"trials_{name}.csv",
                                         RESULTS_DIR / "optuna")
    path = write_result(f"tuning_{name}", payload, config=cfg, results_dir=RESULTS_DIR)
    print(f"{name}: best {payload['best_value']:.4f} (trial {payload['best_trial']}); wrote {path}", flush=True)


if __name__ == "__main__":
    main()
