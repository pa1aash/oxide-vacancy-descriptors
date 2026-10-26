"""Cluster spec generator (scripts/queue/make_specs.py, dftgnn.train.nrp): counts, determinism, purity, leakage,
grid coverage, the pinned code reference and the specs mode of the queue (CPU only, no GPU, no network)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from dftgnn.split.cv import CV_KS
from dftgnn.train import RunSpec, check_code_ref, run_id
from dftgnn.train import nrp as NRP

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data/processed/universe_v1.parquet").is_file(),
                                reason="universe parquet not built")
sys.path.insert(0, str(ROOT / "scripts" / "queue"))
_spec = importlib.util.spec_from_file_location("make_specs", ROOT / "scripts" / "queue" / "make_specs.py")
MS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MS)
TUNED = str(ROOT / "configs" / "tuned_v1.yaml")
V1_STAGES = [s for s in NRP.STAGES if s not in NRP.V2_ONLY]
V2_OF_V1 = ("cv", "precision", "moment_v2")     # v2 versions of v1 stages


def _gen(stage, hosts_file=None, tmp=Path("x")):
    return MS.generate(stage, NRP.CODE_REF, "v1", hosts_file, TUNED, tmp / f"{stage}.jsonl")


@pytest.fixture(scope="module")
def built():
    return {s: _gen(s) for s in V1_STAGES}


def test_counts_per_stage(built):
    counts = {s: len(jobs) for s, (jobs, _, _) in built.items()}
    assert counts == {s: NRP.EXPECTED[s] for s in V1_STAGES}
    print("spec counts:", counts)


def test_run_ids_unique_and_stable(built):
    allids = [j["run_id"] for jobs, _, _ in built.values() for j in jobs]
    assert len(set(allids)) == len(allids)                  # unique across every stage
    for s in ("cv", "precision", "peval", "relax"):
        _jobs, body, man = _gen(s)
        assert body == built[s][1] and man == built[s][2]


def test_manifest_sha_reproducible(built):
    for jobs, body, man in built.values():
        m = json.loads(man)
        assert m["sha256"] == hashlib.sha256(body).hexdigest() and m["count"] == len(jobs)
        assert m["code_ref"] == NRP.CODE_REF and "time" not in json.dumps(m)


def test_two_cli_invocations_identical(tmp_path):
    outs = []
    for i in (0, 1):
        out = tmp_path / str(i) / "cv.jsonl"
        out.parent.mkdir()
        subprocess.run([sys.executable, str(ROOT / "scripts/queue/make_specs.py"), "--stage", "cv", "--out", str(out)],
                       check=True, capture_output=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin",
                                                             "PYTHONHASHSEED": str(i + 1), "OMP_NUM_THREADS": "1"})
        outs.append((out.read_bytes(), out.with_name("cv.manifest.json").read_bytes()))
    assert outs[0] == outs[1]


@pytest.mark.parametrize("stage", [s for s in V1_STAGES if s != "relax"])
def test_no_test_host_in_training_sample(built, stage):
    jobs = built[stage][0]
    for j in random.Random(0).sample(jobs, min(30, len(jobs))):
        pairs = NRP.leakage_hosts(j)
        assert pairs
        for test, seen in pairs:
            assert test and not test & seen


def test_every_job_carries_the_contract(built):
    for s, (jobs, _, _) in built.items():
        for j in jobs:
            assert j["code_ref"] == NRP.CODE_REF and j["backbone"] == "v1" and j["stage"] == s
            assert j["group"] == NRP.GROUP[s] and isinstance(j["priority"], int) and isinstance(j["after"], list)
            sp = j["spec"]
            assert {"model", "r", "budget", "seed"} <= set(sp)
            if "lr" in sp:
                assert sp["code_ref"] == NRP.CODE_REF and "est_peak_gb" in j
                assert run_id(RunSpec.from_dict(sp), NRP.CODE_REF, json.loads(built[s][2])["graphs_manifest_sha256"]) \
                    == j["run_id"]
    order = [NRP.GROUP[s] for s in V1_STAGES]
    assert order == sorted(order)
    lo = [min(j["priority"] for j in built[s][0]) for s in V1_STAGES]
    assert lo == sorted(lo)


def test_cv_covers_all_folds(built):
    from dftgnn.split import load_split

    cv = load_split("cv_v1")
    jobs = built["cv"][0]
    cells = {(j["spec"]["split"], j["spec"]["model"], j["spec"]["seed"]) for j in jobs}
    want = {(f["split"], m, s) for k in CV_KS for f in cv["by_k"][str(k)]["folds"] for m in ("S", "D-state")
            for s in (0, 1, 2)}
    assert cells == want and len(jobs) == 180
    for j in jobs:
        f = int(j["spec"]["split"].rsplit("_f", 1)[1])
        k = int(j["spec"]["split"].split("_k")[1].split("_")[0])
        assert j["spec"]["r"] == f and j["spec"]["budget"] == cv["by_k"][str(k)]["folds"][f]["n_train"]
        assert j["spec"]["results_subdir"] == "cv" and j["spec"]["lr"] in (0.001421106328718029, 0.0016731107509407918)


def test_precision_grid_exact(built):
    jobs = built["precision"][0]
    got = sorted((j["spec"]["model"], j["spec"]["r"], j["spec"]["budget"], j["spec"]["seed"]) for j in jobs)
    want = sorted([(m, r, b, s) for m in ("S", "D-state") for r in range(10, 30) for b in (400, 654) for s in range(5)]
                  + [(m, r, b, s) for m in ("S", "D-state") for r in range(10) for b in (400, 654) for s in (3, 4)])
    assert got == want and len(got) == 480
    assert all(j["spec"]["split"] == f"outer_r{j['spec']['r']}" and j["spec"]["results_subdir"] == "precision"
               for j in jobs)


def test_relax_hosts_file_and_geomeval_dependencies(tmp_path, built):
    relax_all = built["relax"][0]
    hosts = sorted(j["spec"]["host_id"] for j in relax_all)[::7]
    hf = tmp_path / "hosts.txt"
    hf.write_text("# top-up\n" + "\n".join(hosts) + "\n")
    relax, _, man = _gen("relax", str(hf))
    assert len(relax) == len(hosts) and json.loads(man)["hosts_file"]["sha256"]
    keep = {j["run_id"] for j in relax}
    assert keep <= {j["run_id"] for j in relax_all}         # same ids as in the full list
    geo, _, _ = _gen("geomeval", str(hf))
    assert {a for j in geo for a in j["after"]} <= keep
    assert [j["run_id"] for j in geo] == [j["run_id"] for j in built["geomeval"][0]]


@pytest.fixture(scope="module")
def built_v2():
    return {s: MS.generate(s, NRP.CODE_REF, "v2", None, str(NRP.TUNED_V2), Path("x") / f"{s}_v2.jsonl")
            for s in V2_OF_V1}


def test_backbone_v2_cv_and_precision(built, built_v2):
    import yaml

    arch = yaml.safe_load(NRP.TUNED_V2.read_text())["arch"]
    for s, (jobs, body, man) in built_v2.items():
        m, v1 = json.loads(man), s.removesuffix("_v2")      # moment_v2 is the v2 version of moment
        assert len(jobs) == NRP.EXPECTED[s] and m["backbone"] == "v2" and m["label"] == NRP.V2_LABEL
        assert m["backbones_with_this_stage"] == ({"v2": f"{s}.jsonl"} if s in NRP.V2_ONLY
                                                  else {"v1": f"{s}.jsonl", "v2": f"{s}_v2.jsonl"})
        assert {j["spec"]["results_subdir"] for j in jobs} == {f"{v1}_v2"}
        assert all({k: j["spec"]["hp"][k] for k in arch} == arch for j in jobs)
        assert not {j["run_id"] for j in jobs} & {j["run_id"] for j in built[v1][0]}
        assert [(j["spec"]["model"], j["spec"]["split"], j["spec"]["budget"], j["spec"]["seed"]) for j in jobs] == \
            [(j["spec"]["model"], j["spec"]["split"], j["spec"]["budget"], j["spec"]["seed"]) for j in built[v1][0]]
        assert [j["spec"]["exclude_sites"] for j in jobs] == [j["spec"]["exclude_sites"] for j in built[v1][0]]
        assert min(j["priority"] for j in jobs) > max(j["priority"] for j in built[v1][0])
        for j in random.Random(1).sample(jobs, 30):
            for test, seen in NRP.leakage_hosts(j):
                assert test and not test & seen
    with pytest.raises(ValueError):
        NRP.build_stage("relax", backbone="v2")
    with pytest.raises(ValueError):
        NRP.build_stage("moment_v2", backbone="v1")
    assert "loco" not in NRP.STAGES


def test_pure_no_gpu_no_network(monkeypatch):
    import torch

    def boom(*a, **k):
        raise AssertionError("spec generation touched the network or the GPU")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(torch.cuda, "is_available", boom)
    monkeypatch.setattr(torch.cuda, "init", boom)
    assert len(NRP.build_stage("cv")) == 180 and len(NRP.build_stage("embed")) == 240


def test_existing_run_ids_unchanged():
    """A spec without code_ref keeps the commit-SHA rule: committed results reproduce their ids."""
    n = 0
    for f in sorted((ROOT / "results" / "loco").glob("*.json"))[:5] + sorted(ROOT.joinpath("results").glob("*.json"))[:20]:
        pay = json.loads(f.read_text())["payload"]
        if not isinstance(pay, dict) or "spec" not in pay or "code_sha" not in pay:
            continue
        spec = RunSpec.from_dict(pay["spec"])
        assert spec.code_ref is None
        assert run_id(spec, pay["code_sha"], pay["graphs_manifest_sha256"]) == pay["run_id"]
        n += 1
    assert n >= 5


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin:/opt/homebrew/bin", "HOME": str(cwd)})


def test_check_code_ref(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "c1")
    with pytest.raises(RuntimeError, match="does not exist"):
        check_code_ref("nrp-code-v1", root=tmp_path)
    _git(tmp_path, "tag", "-a", "nrp-code-v1", "-m", "pin")
    c1 = check_code_ref("nrp-code-v1", root=tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "r.json").write_text("{}\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "results on top")
    assert check_code_ref("nrp-code-v1", root=tmp_path) == c1           # results on top of the tag are fine
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    _git(tmp_path, "commit", "-qam", "code change")
    with pytest.raises(RuntimeError, match="differs"):
        check_code_ref("nrp-code-v1", root=tmp_path)


def test_check_code_ref_accepts_the_bug_fix_tag(tmp_path):
    """Specs pinned to nrp-code-v1 run on a checkout of nrp-code-v2 (deviations row of 2026-10-14): v1 must stay an
    ancestor and the pinned paths must be unchanged since v2."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "c1")
    _git(tmp_path, "tag", "-a", "nrp-code-v1", "-m", "pin")
    (tmp_path / "src" / "a.py").write_text("x = 2\n")                     # the bug fix
    _git(tmp_path, "commit", "-qam", "fix")
    with pytest.raises(RuntimeError, match="differs"):                   # refused: fix not tagged yet
        check_code_ref("nrp-code-v1", root=tmp_path)
    _git(tmp_path, "tag", "-a", "nrp-code-v2", "-m", "fix")
    c2 = _git_out(tmp_path, "rev-parse", "nrp-code-v2^{commit}")
    assert check_code_ref("nrp-code-v1", root=tmp_path) == c2            # accepted: checkout of v2
    _git(tmp_path, "checkout", "-q", "--detach", "nrp-code-v2")
    assert check_code_ref("nrp-code-v1", root=tmp_path) == c2
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "r.json").write_text("{}\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "results on top")
    assert check_code_ref("nrp-code-v1", root=tmp_path) == c2            # results on top of v2 are fine
    (tmp_path / "specs").mkdir()
    (tmp_path / "specs" / "s.jsonl").write_text("{}\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "spec change")
    with pytest.raises(RuntimeError, match="nrp-code-v2"):              # refused: changed since v2
        check_code_ref("nrp-code-v1", root=tmp_path)
    with pytest.raises(RuntimeError, match="differs"):
        check_code_ref("nrp-code-v2", root=tmp_path)                     # v2 has no successor of its own


def test_check_code_ref_refuses_v2_without_v1_ancestor(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "c1")
    _git(tmp_path, "tag", "-a", "nrp-code-v1", "-m", "pin")
    _git(tmp_path, "checkout", "-q", "--orphan", "other")
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "unrelated history")
    _git(tmp_path, "tag", "-a", "nrp-code-v2", "-m", "fix")
    with pytest.raises(RuntimeError, match="not an ancestor"):
        check_code_ref("nrp-code-v1", root=tmp_path)


def _git_out(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

def test_spec_file_verification_shards_and_dependencies(tmp_path, built):
    for s in ("relax", "geomeval"):
        _jobs, body, man = built[s]
        (tmp_path / f"{s}.jsonl").write_bytes(body)
        (tmp_path / f"{s}.manifest.json").write_bytes(man)
    assert len(NRP.load_spec_file(tmp_path / "relax.jsonl")) == 818
    shards = [NRP.shard(built["relax"][0], f"{i}/4") for i in range(4)]
    ids = [j["run_id"] for sh in shards for j in sh]
    assert len(ids) == len(set(ids)) == 818
    q = tmp_path / "jobs"
    geo = built["geomeval"][0]
    with pytest.raises(ValueError, match="enqueue that stage first"):
        NRP.resolve_deps(geo, q, verified=lambda rid: False)
    both = NRP.resolve_deps(built["relax"][0] + geo, q, verified=lambda rid: False)
    assert [j["after"] for j in both[818:]] == [j["after"] for j in geo]
    dropped = NRP.resolve_deps(geo, q, verified=lambda rid: True)    # relax results already ingested
    assert all(j["after"] == [] for j in dropped)
    bad = bytearray((tmp_path / "relax.jsonl").read_bytes())
    bad[10] = ord("0") if bad[10] != ord("0") else ord("1")
    (tmp_path / "relax.jsonl").write_bytes(bytes(bad))
    with pytest.raises(ValueError, match="sha256"):
        NRP.load_spec_file(tmp_path / "relax.jsonl")


def test_supervisor_checkout_defaults(tmp_path):
    sp = importlib.util.spec_from_file_location("supervisor_nrp", ROOT / "scripts" / "queue" / "supervisor.py")
    sup = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(sup)
    a = sup.parse(["--checkout", str(tmp_path), "--specs", "specs/cv.jsonl", "--no-launch", "--tranches", "N3"])
    assert a.queue == str(tmp_path / "jobs") and a.marker.startswith(str(tmp_path)) and a.no_launch
    assert a.specs == ["specs/cv.jsonl"] and a.tranches == ("N3",)
