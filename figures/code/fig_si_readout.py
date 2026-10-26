"""SI readout figure (M3b): schematics/fig_readout.pdf centred, at natural size, on a 138.6 mm wide page.

The readout schematic leaves the main text (Figure 1 now pairs the pipeline with the ZnO render) and becomes an
SI figure. The schematic source stays in schematics/fig_readout.tex; this script only places its PDF, vector,
at the MDPI text width.

    python figures/code/fig_si_readout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import composite_paper as cp

SRC = cp.ROOT / "schematics" / "fig_readout.pdf"
OUT = cp.ROOT / "figures" / "si" / "fig_readout.pdf"
BODY = r"""\begin{minipage}{%smm}
\centering
\includegraphics{readout.pdf}
\end{minipage}""" % cp.WIDTH_MM


if __name__ == "__main__":
    print(cp.build(BODY, {"readout.pdf": SRC}, OUT))
