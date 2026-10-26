import json
import subprocess
from importlib import metadata

import pytest

from dftgnn.io.results import DirtyTreeError, write_result


def _repo(path):
    run = lambda *a: subprocess.run(["git", *a], cwd=path, check=True, capture_output=True)
    run("init", "-q", "-b", "main")
    run("config", "user.name", "T")
    run("config", "user.email", "t@example.org")
    run("config", "commit.gpgsign", "false")
    (path / "f.txt").write_text("x")
    run("add", "f.txt")
    run("commit", "-q", "-m", "init", "--no-verify")
    return run


def test_write_result_records_provenance(tmp_path):
    _repo(tmp_path)
    out = write_result("demo", {"mae": 0.1}, repo_root=tmp_path, results_dir=tmp_path / "results")
    rec = json.loads(out.read_text())
    assert out.name == "demo.json"
    assert len(rec["git_sha"]) == 40
    assert rec["git_dirty"] is False
    assert rec["timestamp_utc"].endswith("+00:00")
    assert rec["hostname"]
    assert rec["seeds"] == rec["config"]["split"]["seeds"]
    assert "numpy" in rec["package_versions"]
    assert rec["config"]["stats"]["delta_eV"] == 0.05
    assert rec["payload"] == {"mae": 0.1}


def test_dirty_tree_refused_unless_allowed(tmp_path):
    _repo(tmp_path)
    (tmp_path / "f.txt").write_text("changed")
    with pytest.raises(DirtyTreeError):
        write_result("demo", {}, repo_root=tmp_path, results_dir=tmp_path / "r")
    out = write_result("demo", {}, repo_root=tmp_path, results_dir=tmp_path / "r", allow_dirty=True)
    assert json.loads(out.read_text())["git_dirty"] is True


def test_write_result_records_matgl_version(tmp_path):
    _repo(tmp_path)
    out = write_result("demo", {}, repo_root=tmp_path, results_dir=tmp_path / "results")
    versions = json.loads(out.read_text())["package_versions"]
    assert versions["matgl"] == metadata.version("matgl")
