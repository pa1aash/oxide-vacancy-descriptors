"""Write or verify data/MANIFEST.sha256 for the pinned Kumagai release and the K25 companion repo.

    python scripts/make_manifest.py            # write
    python scripts/make_manifest.py --verify   # verify (exit 1 on any mismatch)

Line format (tab separated): sha256, size_bytes, path (relative to repo root), source URL.
"""
from __future__ import annotations

import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "MANIFEST.sha256"
DB_SHA = "e310353ecbf698499c85a5ec5416b76e82dcecfd"
ML_SHA = "39db6d2d8f5653d6452c7877ecb2e296f04a7a01"
REPOS = {
    "oxygen_vacancies_db": ("kumagai-group/oxygen_vacancies_db", DB_SHA),
    "ML_charged_defects": ("kumagai-group/ML_charged_defects", ML_SHA),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entries() -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for name, (repo, sha) in REPOS.items():
        archive = RAW / f"{name}-{sha}.tar.gz"
        out.append((archive, f"https://codeload.github.com/{repo}/tar.gz/{sha}"))
        base = RAW / "unpacked" / name
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(base).as_posix()
            if rel.startswith("site_info/"):  # extracted from site_info.tar.gz
                url = f"https://raw.githubusercontent.com/{repo}/{sha}/site_info.tar.gz#{rel}"
            else:
                url = f"https://raw.githubusercontent.com/{repo}/{sha}/{rel}"
            out.append((p, url))
    return out


def write() -> None:
    lines = [
        "# MANIFEST.sha256: Kumagai oxygen-vacancy release and Kiyohara et al. companion repo",
        f"# pinned: kumagai-group/oxygen_vacancies_db@{DB_SHA} (master HEAD, pushed 2022-11-16; no DOI exists)",
        f"# pinned: kumagai-group/ML_charged_defects@{ML_SHA} (main HEAD, pushed 2025-12-16)",
        f"# downloaded_utc: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "# columns (tab separated): sha256, size_bytes, path, source_url",
    ]
    for p, url in entries():
        lines.append(f"{sha256(p)}\t{p.stat().st_size}\t{p.relative_to(ROOT).as_posix()}\t{url}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST} ({len(lines) - 5} files)")


def verify() -> int:
    bad = n = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        h, size, path, _ = line.split("\t")
        p = ROOT / path
        n += 1
        if not p.is_file():
            print(f"MISSING  {path}")
            bad += 1
        elif p.stat().st_size != int(size) or sha256(p) != h:
            print(f"MISMATCH {path}")
            bad += 1
    print(f"verified {n} files, {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(verify())
    write()
