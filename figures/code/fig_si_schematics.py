"""SI schematic figures (F1b): the filter cascade (Figure S2) and the split designs (Figure S3), each centred, at
natural size, on a 138.6 mm wide page, as fig_si_readout.py does for the readout schematic.

The sources stay in schematics/fig_cascade.tex and schematics/fig_splits.tex; this script only places their
PDFs, vector, at the MDPI text width and writes a caption file beside each, the schematic's caption of record
followed by a source line. The figure number in the caption title is set here; the caption of record keeps the
number it was written with.

    python figures/code/fig_si_schematics.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import composite_paper as cp

OUT_DIR = cp.ROOT / "figures" / "si"
NUMBERS = {"fig_cascade": "S2", "fig_splits": "S3"}
BODY = r"""\begin{minipage}{%smm}
\centering
\includegraphics{%s.pdf}
\end{minipage}"""


def main() -> list[Path]:
    out = []
    for name, num in NUMBERS.items():
        src = cp.ROOT / "schematics" / f"{name}.pdf"
        out.append(cp.build(BODY % (cp.WIDTH_MM, name), {f"{name}.pdf": src}, OUT_DIR / f"{name}.pdf"))
        caption = (cp.ROOT / "schematics" / f"{name}.caption.md").read_text().rstrip()
        caption = re.sub(r"^\*\*Figure S\d+\.\*\*", f"**Figure {num}.**", caption)
        (OUT_DIR / f"{name}.caption.md").write_text(
            f"{caption}\nSource: `schematics/{name}.tex` (caption of record: `schematics/{name}.caption.md`); placed "
            f"at the text width by `figures/code/fig_si_schematics.py`.\n")
    return out


if __name__ == "__main__":
    print(*main(), sep="\n")
