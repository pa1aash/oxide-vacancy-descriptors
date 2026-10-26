"""Figure 1 (M3b): composite of (a) the model pipeline schematic and (b) the ZnO vacancy render.

Layout copies the figure block of (label fig:schematic): (a) at the full text width, (b) centred below it at
0.45 of the text width, each with a bold panel letter centred above it, 2 pt above the panel and 6 pt between
the rows. The composite is exactly 138.6 mm wide, so it drops in at \\linewidth.

    (a) schematics/fig_pipeline.pdf, included as vector (fonts stay TeX Gyre Heros)
    (b) figures/main/fig1_zno_vacancy.png, resampled (Lanczos) to 600 dpi at the panel width and embedded

Panel letters are set in TeX Gyre Heros Bold, the schematic's family, so the PDF holds one font family. The
raster panel is intended: qa_figure.py reports "vector" as failed for this figure only.

    python figures/code/fig1_overview.py [--render figures/main/fig1_zno_vacancy.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import composite_paper as cp

ROOT = cp.ROOT
WIDTH_MM = cp.WIDTH_MM
PANEL_B_FRACTION = 0.45           # : minipage{0.45\linewidth}
DPI = 600
PIPELINE = ROOT / "schematics" / "fig_pipeline.pdf"
RENDER = ROOT / "figures" / "main" / "fig1_zno_vacancy.png"
OUT = ROOT / "figures" / "main" / "fig1_overview.pdf"

BODY = r"""\begin{minipage}{%(w)smm}
\centering
\textbf{(a)}\\[2pt]
\includegraphics[width=%(w)smm]{pipeline.pdf}\\[6pt]
\begin{minipage}{%(wb)smm}
\centering
\textbf{(b)}\\[2pt]
\includegraphics[width=%(wb)smm]{render.png}
\end{minipage}
\end{minipage}"""


def resample(src: Path, dst: Path, width_mm: float, dpi: int) -> tuple[int, int]:
    """Lanczos resample to ``dpi`` at ``width_mm``; RGB, dpi tag set, no other metadata."""
    im = Image.open(src).convert("RGB")
    w = round(width_mm / 25.4 * dpi)
    h = round(im.height * w / im.width)
    im = im.resize((w, h), Image.LANCZOS) if (w, h) != im.size else im
    im.save(dst, dpi=(dpi, dpi), optimize=True)
    return w, h


def main() -> Path:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--render", type=Path, default=RENDER)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    wb = round(WIDTH_MM * PANEL_B_FRACTION, 2)
    px = []
    cp.build(BODY % {"w": WIDTH_MM, "wb": wb}, {"pipeline.pdf": PIPELINE}, a.out,
             prepare=lambda t: px.extend(resample(a.render, t / "render.png", wb, DPI)))
    print(f"panel (b): {px[0]} x {px[1]} px from {a.render.relative_to(ROOT)} ({DPI} dpi at {wb} mm)")
    return a.out


if __name__ == "__main__":
    print(main())
