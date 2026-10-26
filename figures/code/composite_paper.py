"""Vector composites through pdfLaTeX (M3b): included PDFs stay vector, fonts stay those of the inputs.

``build(body, files, out)`` sets ``body`` in a border-free standalone page (TeX Gyre Heros, the schematics'
family; no creation date, no trailer id, so reruns are byte-stable) with ``files`` (name -> source path) copied
beside it, and writes the PDF to ``out``.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WIDTH_MM = 138.6                  # MDPI text width

PREAMBLE = r"""\pdfinfoomitdate=1
\pdftrailerid{}
\pdfsuppressptexinfo=-1
\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{tgheros}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{graphicx}
\begin{document}
\fontsize{9}{11}\selectfont
"""


def build(body: str, files: dict[str, Path], out: Path, prepare=None) -> Path:
    """Compile ``body``; ``prepare(tmpdir)`` may write further inputs first."""
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        for name, src in files.items():
            shutil.copy(src, t / name)
        if prepare is not None:
            prepare(t)
        (t / "fig.tex").write_text(PREAMBLE + body + "\n\\end{document}\n")
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig.tex"], cwd=t, check=True,
                       capture_output=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(t / "fig.pdf", out)
    return out
