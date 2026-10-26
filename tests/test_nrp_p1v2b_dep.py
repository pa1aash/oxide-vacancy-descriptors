"""pcompv2b depends on one p1v2b run: an ingested p1v2b record is a met dependency (dftgnn.train.nrp.
p1v2b_result_verified) only when its record, code commit, clean-tree flag and checkpoint sha256 all check out.
CPU only; the check against the ingested records skips when they are absent."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dftgnn import tasks
from dftgnn.train import nrp as NRP

ROOT = Path(__file__).resolve().parents[1]
RID, GOOD, CKSUM = "0123456789abcdef", "a" * 40, "b" * 64


@pytest.fixture
def p1v2b(tmp_path, monkeypatch):
    rec_dir = tmp_path / "p1v2b"
    rec_dir.mkdir()
    manifest = tmp_path / "p1v2b.sha256"
    monkeypatch.setattr(NRP, "P1V2B_DIR", rec_dir)
    monkeypatch.setattr(NRP, "P1V2B_CHECKPOINTS", manifest)
    monkeypatch.setattr(NRP, "accepted_code_commits", lambda ref: {GOOD})
    monkeypatch.setattr(tasks, "verified", lambda path, rid: False)

    def write(rid=RID, git_sha=GOOD, dirty=False, ck_sha=CKSUM, listed=CKSUM, record=True):
        if record:
            pay = {"run_id": rid, "code_ref": "nrp-code-v1", "checkpoint": {"path": f"checkpoints/{rid}.pt", "sha256": ck_sha}}
            (rec_dir / f"{rid}.json").write_text(json.dumps({"name": rid, "git_sha": git_sha, "git_dirty": dirty,
                                                            "payload": pay}))
        manifest.write_text(f"{listed}  123  checkpoints/{rid}.pt\n")
    return write


def test_accepts_verified_p1v2b_record(p1v2b):
    p1v2b()
    assert NRP.task_result_verified(RID)


@pytest.mark.parametrize("kw", [dict(record=False), dict(dirty=True), dict(git_sha="c" * 40), dict(listed="d" * 64),
                                dict(rid="fedcba9876543210")], ids=["missing", "dirty", "wrong-commit", "sha-mismatch",
                                                                    "other-id"])
def test_refuses_bad_p1v2b_record(p1v2b, kw):
    p1v2b(**kw)
    assert not NRP.task_result_verified(RID)


def test_refuses_checkpoint_missing_from_manifest(p1v2b, tmp_path):
    p1v2b()
    (tmp_path / "p1v2b.sha256").write_text("")
    assert not NRP.task_result_verified(RID)


def test_existing_stage_result_unchanged(p1v2b, monkeypatch):
    assert not NRP.task_result_verified(RID)                        # no task result, no p1v2b record
    monkeypatch.setattr(tasks, "verified", lambda path, rid: True)  # a verified task result still counts
    assert NRP.task_result_verified(RID)


@pytest.mark.skipif(not (ROOT / "results" / "p1v2b").is_dir(), reason="p1v2b results not ingested")
def test_all_pcompv2b_dependencies_resolve_against_ingested_records(tmp_path):
    jobs = NRP.load_spec_file(NRP.SPECS_DIR / "pcompv2b.jsonl")
    assert len(jobs) == 180
    assert all(len(j["after"]) == 1 for j in jobs)
    for i in range(30):                                              # 30 shards, as the cluster enqueues them
        out = NRP.resolve_deps(NRP.shard(jobs, f"{i}/30"), tmp_path / "jobs")
        assert all(j["after"] == [] for j in out)
