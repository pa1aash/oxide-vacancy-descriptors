"""Regenerate the figures of the study from the stored records into figures/main/ and figures/si/ (PDF).

Each figure script in figures/paper/ reads only stored records under results/ (and figures/main/data/) and writes
one vector PDF. The scripts run in parallel, each in its own process. Figures whose inputs are not in this
repository are listed as SKIPPED with the reason; where a rendered image of such a figure is shipped (PNG), it stays
as it is.

    python reproduce/figures.py [--only NAME ...] [--jobs N]
"""
from __future__ import annotations

import os
import sys

# reproduce/numbers.py is named like the standard-library module ``numbers``: keep this folder off the import path.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# figure -> (script in figures/paper, output relative to the repository root)
FIGURES = {
    "figure2_descriptor_advantage": ("figure2_descriptor_advantage.py", "figures/main/figure2_descriptor_advantage.pdf"),
    "figure3_within_host": ("figure3_within_host.py", "figures/main/figure3_within_host.pdf"),
    "figure_dshell": ("figure_dshell.py", "figures/main/figure_dshell.pdf"),
    "figure4_staging": ("figure4_staging.py", "figures/main/figure4_staging.pdf"),
    "figure5_screening": ("figure5_screening.py", "figures/main/figure5_screening.pdf"),
    "figureE1_within_host_groups": ("figureE1_within_host_groups.py", "figures/si/figureE1_within_host_groups.pdf"),
    "figureF1_sensitivities": ("figureF1_sensitivities.py", "figures/si/figureF1_sensitivities.pdf"),
    "figureH1_probe": ("figureH1_probe.py", "figures/si/figureH1_probe.pdf"),
}
SCHEMATIC = "needs the schematic sources (LaTeX/TikZ drawings) and pdfLaTeX, not in this repository"
SKIPPED = {
    "figure1_models": f"{SCHEMATIC}; the structure render figures/main/fig1_zno_vacancy.png is shipped",
    "figureA1_cascade": SCHEMATIC + " (counts in results/filter_cascade.json and tables/filter_cascade.csv)",
    "figureB1_readout": SCHEMATIC,
    "figureC1_splits": SCHEMATIC,
    "figureI1_sno2_vacancy": "structure render made with Blender from the release supercell; the rendered "
                             "figures/si/figureI1_sno2_vacancy.png is shipped (render settings in its .render.json)",
}


def run(name: str) -> tuple[str, str, str, float]:
    script, out = FIGURES[name]
    path = ROOT / out
    before = path.stat().st_mtime_ns if path.exists() else None
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT),
                                                         os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
           "MPLBACKEND": "Agg"}
    t = time.time()
    proc = subprocess.run([sys.executable, str(ROOT / "figures" / "paper" / script)], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    dt = time.time() - t
    if proc.returncode != 0:
        last = (proc.stderr.strip().splitlines() or ["no message"])[-1]
        return name, out, f"FAILED ({last})", dt
    if not path.exists() or (before is not None and path.stat().st_mtime_ns == before):
        return name, out, "FAILED (no new PDF written)", dt
    return name, out, "OK", dt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES), help="regenerate only these figures")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1), help="parallel processes")
    a = ap.parse_args(argv)
    names = a.only or list(FIGURES)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        rows = list(pool.map(run, names))
    if not a.only:
        rows += [(n, "-", f"SKIPPED: {why}", 0.0) for n, why in SKIPPED.items()]
    w = [max(len(r[i]) for r in rows + [("figure", "output", "", 0)]) for i in range(2)]
    print(f"{'figure'.ljust(w[0])} | {'output'.ljust(w[1])} | status")
    print(f"{'-' * w[0]}-|-{'-' * w[1]}-|-------")
    for name, out, status, dt in rows:
        print(f"{name.ljust(w[0])} | {out.ljust(w[1])} | {status}" + (f" ({dt:.1f} s)" if status == "OK" else ""))
    n_ok = sum(r[2] == "OK" for r in rows)
    n_fail = sum(r[2].startswith("FAILED") for r in rows)
    print(f"\n{n_ok} regenerated, {sum(r[2].startswith('SKIPPED') for r in rows)} skipped, {n_fail} failed; "
          f"{time.time() - t0:.0f} s")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
