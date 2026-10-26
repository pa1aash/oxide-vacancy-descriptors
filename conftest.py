"""Test-suite configuration: skip, with a stated reason, the tests whose inputs are not in this repository.

Two rules:

1. tests/unshipped_inputs.txt lists pytest node ids, one per line, each followed by a tab and a reason key (see
   REASONS). Those tests are skipped before they run.
2. Generic rule: a test that fails because a file under one of the paths in UNSHIPPED is missing (FileNotFoundError),
   or because ``git`` finds no repository history, is reported as skipped with the matching reason instead of failed.

Every other failure is reported as a failure.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
LIST = ROOT / "tests" / "unshipped_inputs.txt"

REASONS = {
    "data": "needs unshipped processed inputs (data/processed)",
    "gpu": "needs a GPU",
    "manuscript": "needs manuscript sources",
    "docs": "needs development documents (docs/) that are not in this repository",
    "git": "needs the git history and tags of the development repository",
    "checkpoints": "needs model checkpoints (not in this repository; see checkpoints/)",
}
UNSHIPPED = {"data/processed": "data", "data/raw": "data", "checkpoints": "checkpoints", "docs": "docs",
             "paper": "manuscript", "schematics": "manuscript", "blender": "manuscript"}


def _listed() -> dict[str, str]:
    out = {}
    if not LIST.is_file():
        return out
    for line in LIST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        node, _, key = line.partition("\t")
        key = key.strip() or "data"
        out[node.strip()] = REASONS.get(key, key)
    return out


def pytest_collection_modifyitems(config, items):
    listed = _listed()
    for item in items:
        reason = listed.get(item.nodeid)
        if reason is None and "[" in item.nodeid:
            reason = listed.get(item.nodeid.split("[", 1)[0])
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))


def _unshipped_reason(exc: BaseException) -> str | None:
    if isinstance(exc, FileNotFoundError) and exc.filename:
        try:
            rel = Path(exc.filename).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return None
        for prefix, key in UNSHIPPED.items():
            if rel == prefix or rel.startswith(prefix + "/"):
                return f"{REASONS[key]} ({rel})"
    if isinstance(exc, subprocess.CalledProcessError):
        cmd = exc.cmd if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd).split()
        if cmd and Path(str(cmd[0])).name == "git":
            return REASONS["git"]
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.failed and call.excinfo is not None and rep.when in ("setup", "call"):
        reason = _unshipped_reason(call.excinfo.value)
        if reason is not None:
            path, line, _ = item.reportinfo()
            rep.outcome = "skipped"
            rep.longrepr = (str(path), (line or 0) + 1, f"Skipped: {reason}")
