"""One line per tuning study: COMPLETE / RUNNING / FAIL counts and the best value so far.

    python scripts/tune/status.py [--storage-dir /workspace/tuning]
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
    ap.add_argument("--storage-dir", default="/workspace/tuning")
    a = ap.parse_args()
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    S = optuna.trial.TrialState
    out = []
    for db in sorted(Path(a.storage_dir).glob("*.db")):
        st = optuna.load_study(study_name=db.stem, storage=f"sqlite:///{db.resolve()}")
        ts = st.get_trials(deepcopy=False)
        n = {s: sum(t.state == s for t in ts) for s in (S.COMPLETE, S.RUNNING, S.FAIL)}
        done = [t.value for t in ts if t.state == S.COMPLETE]
        best = f"{min(done):.4f}" if done else "-"
        out.append(f"{db.stem} {n[S.COMPLETE]}c/{n[S.RUNNING]}r/{n[S.FAIL]}f best {best}")
    print(" | ".join(out) if out else "no studies")


if __name__ == "__main__":
    main()
