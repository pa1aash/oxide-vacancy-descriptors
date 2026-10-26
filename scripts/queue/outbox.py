"""Pack finished runs into outbox/<name>.tar.gz for rsync back to the Mac (no git needed).

    python scripts/queue/outbox.py pack [--stage smoke] [--tranche T1]
    python scripts/queue/outbox.py pack-files --name <name> results/x.json [more repo-relative files]
    python scripts/queue/outbox.py unpack outbox/<name>.tar.gz

The tarball holds, for every done job: its result JSON, its predictions parquet, its checkpoint, the artefacts
its payload lists and the done/ job record, at their repo-relative paths, plus OUTBOX_MANIFEST.sha256 listing
every member. ``pack`` also writes ``<name>.tar.gz.sha256`` next to the tarball. ``unpack`` checks the tarball
hash, extracts into the repo root (results/ and the gitignored checkpoints/) and re-verifies every member
against the inner manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path

from _common import OUTBOX, QUEUE, ROOT

MANIFEST = "OUTBOX_MANIFEST.sha256"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def members(queue: Path, stage: str | None, tranche: str | None = None) -> list[Path]:
    out = []
    for f in sorted((queue / "done").glob("*.json")):
        job = json.loads(f.read_text())
        if stage and job.get("stage") != stage:
            continue
        if tranche and job.get("tranche") != tranche:
            continue
        res = Path(job["result"])
        pay = json.loads(res.read_text())["payload"]
        out += [res, f]
        if pay.get("predictions"):
            out.append(ROOT / pay["predictions"]["path"])
        if pay.get("checkpoint"):
            out.append(ROOT / pay["checkpoint"]["path"])
        out += [ROOT / a["path"] for a in pay.get("artifacts", [])]     # Task outputs (relax, embed)
    return out


def pack(queue: Path, outbox: Path, stage: str | None, tranche: str | None = None) -> Path:
    """Snapshot of the finished (done/) runs of a stage and/or tranche; a running job is never included."""
    files = members(queue, stage, tranche)
    if not files:
        raise SystemExit("no done jobs to pack")
    label = "_".join(x for x in (stage, tranche) if x) or "all"
    return _pack_files(files, outbox, f"outbox_{label}_{time.strftime('%Y%m%dT%H%M%S')}.tar.gz")


def pack_files(paths: list[str], outbox: Path, label: str) -> Path:
    """Pack explicit repo-relative files (benchmark and pilot results that are not queue jobs)."""
    files = [ROOT / p for p in paths]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    return _pack_files(files, outbox, f"outbox_{label}_{time.strftime('%Y%m%dT%H%M%S')}.tar.gz")


def arcname(path: Path) -> Path:
    """Member path of a file: repo-relative inside the checkout; ``external/<dir>/<state>/<file>`` for a file outside
    it (the done/ records of a queue that lies outside the checkout), so the name never climbs out with ``..``."""
    for p, root in ((Path(os.path.abspath(path)), Path(os.path.abspath(ROOT))), (Path(path).resolve(), ROOT.resolve())):
        try:
            return p.relative_to(root)
        except ValueError:
            continue
    return Path("external", *Path(path).resolve().parts[-3:])


def _pack_files(files: list[Path], outbox: Path, name: str) -> Path:
    outbox.mkdir(parents=True, exist_ok=True)
    tar_path = outbox / name
    src = [Path(f).resolve() for f in files]
    rel = [arcname(f) for f in src]
    manifest = "".join(f"{_sha(f)}  {r}\n" for f, r in zip(src, rel, strict=True)).encode()
    with tarfile.open(tar_path, "w:gz") as tar:
        for f, r in zip(src, rel, strict=True):
            tar.add(f, arcname=str(r))
        info = tarfile.TarInfo(MANIFEST)
        info.size = len(manifest)
        tar.addfile(info, io.BytesIO(manifest))
    tar_path.with_name(name + ".sha256").write_text(f"{_sha(tar_path)}  {name}\n")
    print(f"{tar_path} ({len(rel)} files)")
    return tar_path


def unpack(tar_path: Path, dest: Path = ROOT) -> int:
    side = tar_path.with_name(tar_path.name + ".sha256")
    if side.exists() and side.read_text().split()[0] != _sha(tar_path):
        raise SystemExit(f"tarball hash mismatch: {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tar:
        names = tar.getnames()
        for n in names:
            if n.startswith("/") or ".." in Path(n).parts:
                raise SystemExit(f"unsafe member path {n}")
        manifest = tar.extractfile(MANIFEST).read().decode()
        tar.extractall(dest, members=[m for m in tar.getmembers() if m.name != MANIFEST], filter="data")
    bad = [r for sha, r in (ln.split("  ", 1) for ln in manifest.splitlines()) if _sha(dest / r) != sha]
    if bad:
        raise SystemExit(f"hash mismatch after unpack: {bad}")
    n = len(manifest.splitlines())
    print(f"unpacked and verified {n} files into {dest}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--stage")
    p.add_argument("--tranche")
    p.add_argument("--queue", default=str(QUEUE))
    p.add_argument("--outbox", default=str(OUTBOX))
    f = sub.add_parser("pack-files")
    f.add_argument("--name", required=True)
    f.add_argument("--outbox", default=str(OUTBOX))
    f.add_argument("paths", nargs="+")
    u = sub.add_parser("unpack")
    u.add_argument("tarball")
    a = ap.parse_args()
    if a.cmd == "pack":
        pack(Path(a.queue), Path(a.outbox), a.stage, a.tranche)
    elif a.cmd == "pack-files":
        pack_files(a.paths, Path(a.outbox), a.name)
    else:
        unpack(Path(a.tarball))


if __name__ == "__main__":
    main()
