"""Rebuild the 18 CSV tables in tables/ from the stored records and compare them with the shipped copies.

Each table has a source file tables/sources/<name>.json. A cell there is either literal text or an object

    {"text": "{0} [{1}, {2}]", "values": [{"file": "results/x.json", "pointer": "/payload/a/b", "format": "num"}, ...]}

where ``file`` is a record (or configuration) file relative to the repository root, ``pointer`` an RFC 6901 JSON
pointer into it (YAML files are read as the same tree) and ``format`` one of the number formats below. The script
reads every value, formats it, fills the cell text, writes the tables into a temporary directory, compares them cell
by cell with tables/*.csv and prints PASS or FAIL per table. tables/ is left unchanged unless --write is given.

Number formats (plain text):
    num    integers exactly (thousands separated by commas from 10,000 up); other values with magnitude from
           0.001 to below 1000 with three decimals; 1000 or more rounded to an integer with commas; below 0.001
           as m×10^e with two significant figures; 0.0 as 0.000
    round  the nearest integer, then as num (training-host counts that are fold means)
    sig    three significant figures; below 0.01 as m×10^e with three significant figures
    dp3    three fixed decimals; dp4 four fixed decimals (no minus sign on a value that rounds to zero)
    pct    the value times 100 with no decimals
    trim   three decimals with trailing zeros dropped
    word   a boolean as yes or no
    text   a string as stored

    python reproduce/tables.py [--write]
"""
from __future__ import annotations

import os
import sys

# reproduce/numbers.py is named like the standard-library module ``numbers``: keep this folder off the import path.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse
import csv
import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
SOURCES = TABLES / "sources"


# ---------------------------------------------------------------- reading values

@lru_cache(maxsize=None)
def load(rel: str):
    path = ROOT / rel
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def resolve(doc, pointer: str):
    """RFC 6901 JSON pointer."""
    node = doc
    if pointer in ("", "/"):
        return node
    for tok in pointer.lstrip("/").split("/"):
        tok = tok.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            node = node[int(tok)]
        elif tok in node:
            node = node[tok]
        else:  # YAML keys that are numbers
            node = next(v for k, v in node.items() if str(k) == tok)
    return node


# ---------------------------------------------------------------- formatting

def _sci(x: float, digits: int) -> str:
    m, e = f"{abs(x):.{digits}e}".split("e")
    return f"{m}×10^{int(e)}"


def render(x, fmt: str) -> str:
    """Plain-text rendering of one stored value (rules in the module docstring)."""
    if fmt == "word":
        if not isinstance(x, bool):
            raise TypeError(f"{x!r} is not a boolean")
        return "yes" if x else "no"
    if fmt == "text":
        return str(x)
    if isinstance(x, bool):
        raise TypeError(f"{x!r} is a boolean")
    neg = x < 0
    if fmt == "round":
        x, fmt = round(x), "num"
    if fmt == "num":
        if isinstance(x, float) and x.is_integer() and abs(x) < 1e15:
            x = int(x)
        if isinstance(x, int):
            s = f"{abs(x):,}" if abs(x) >= 10000 else str(abs(x))
        elif x == 0:
            s = "0.000"
        elif abs(x) >= 1000:
            s = f"{abs(x):,.0f}"
        elif abs(x) >= 0.001:
            s = f"{abs(x):.3f}"
        else:
            s = _sci(x, 1)
        return ("-" if x < 0 else "") + s
    if fmt == "sig":
        s = f"{abs(x):.3g}" if abs(x) >= 0.01 or x == 0 else _sci(x, 2)
        return ("-" if neg else "") + s
    if fmt in ("dp3", "dp4"):
        s = f"{abs(x):.{fmt[-1]}f}"
        return ("-" if neg and s.strip("0.") else "") + s
    if fmt == "pct":
        s = f"{100 * abs(x):.0f}"
        return ("-" if neg and s != "0" else "") + s
    if fmt == "trim":
        s = f"{abs(x):.3f}".rstrip("0").rstrip(".")
        return ("-" if neg else "") + s
    raise ValueError(f"unknown format {fmt!r}")


def fill(cell) -> str:
    if isinstance(cell, str):
        return cell
    text = cell["text"]
    for i, v in enumerate(cell["values"]):
        text = text.replace("{%d}" % i, render(resolve(load(v["file"]), v["pointer"]), v["format"]))
    return text


def build(src: dict) -> list[list[str]]:
    return [[fill(c) for c in row] for row in [src["header"], *src["rows"]]]


# ---------------------------------------------------------------- driver

def read_csv(path: Path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="overwrite tables/*.csv with the rebuilt tables")
    a = ap.parse_args(argv)
    srcs = sorted(SOURCES.glob("*.json"))
    if not srcs:
        print("no table sources in tables/sources/", file=sys.stderr)
        return 1
    n_fail, n_cells, n_numeric = 0, 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for sp in srcs:
            src = json.loads(sp.read_text())
            name = src["table"]
            rows = build(src)
            write_csv(tmp / f"{name}.csv", rows)
            n_numeric += sum(1 for row in [src["header"], *src["rows"]] for c in row if not isinstance(c, str))
            target = TABLES / f"{name}.csv"
            have = read_csv(target) if target.exists() else []
            diffs = []
            for i in range(max(len(rows), len(have))):
                a_row = rows[i] if i < len(rows) else []
                b_row = have[i] if i < len(have) else []
                for j in range(max(len(a_row), len(b_row))):
                    x = a_row[j] if j < len(a_row) else None
                    y = b_row[j] if j < len(b_row) else None
                    if x != y:
                        diffs.append((i, j, x, y))
            n_cells += sum(len(r) for r in rows)
            if diffs:
                n_fail += 1
                print(f"FAIL {name}.csv: {len(diffs)} cells differ")
                for i, j, x, y in diffs[:5]:
                    print(f"     row {i} column {j}: rebuilt {x!r}, stored {y!r}")
            else:
                print(f"PASS {name}.csv ({len(rows) - 1} rows x {len(rows[0])} columns)")
        print(f"\n{len(srcs) - n_fail} of {len(srcs)} tables identical; {n_cells} cells, {n_numeric} of them filled "
              "from stored records")
        if a.write:
            for sp in srcs:
                name = json.loads(sp.read_text())["table"]
                shutil.copy(tmp / f"{name}.csv", TABLES / f"{name}.csv")
            print(f"wrote {len(srcs)} tables to tables/")
    return 1 if n_fail and not a.write else 0


if __name__ == "__main__":
    sys.exit(main())
