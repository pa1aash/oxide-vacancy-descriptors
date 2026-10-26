"""Write docs/descriptor_taxonomy.csv for the 70 K21 ML-subset descriptor columns."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "data/raw/unpacked/oxygen_vacancies_db/vacancy_formation_energy_ml/charge0.csv"
OUT = ROOT / "docs/descriptor_taxonomy.csv"

LOC_I = "K21 Sec. II I (PRM 5, 123803, p. 3), three descriptor types"
SC = "structural/compositional"
HE = "host-electronic DFT"
SE = "site-electronic DFT"
PU = "pristine unit-cell DFT"

EV_PRISTINE = ("recomputed exactly from pristine unit-cell data in bulk_visual_data.json "
               "(max abs diff {d}, 1745/1745 rows); identical across q=0,+1,+2 CSVs")
EV_Q = ("identical across q=0,+1,+2 CSVs (70/70 columns, max diff <1e-9); K21 classes it as a pristine-host "
        "or on-site property; no pristine counterpart is shipped, so not recomputed")

rows: list[dict] = []


def add(name, definition, units, level, cls, cost, evidence, locator=LOC_I, leak="no"):
    rows.append({"name": name, "definition": definition, "units": units, "level": level, "class": cls,
                 "deployment_cost": cost, "leakage_flag": leak, "evidence": evidence, "locator": locator})


add("ave_ele_diele", "Spherically averaged electronic (ion-clamped) dielectric constant of the pristine host, from DFPT.",
    "dimensionless", "host", HE, PU, EV_PRISTINE.format(d="1.8e-15"))
add("ave_ion_diele", "Spherically averaged ionic dielectric constant of the pristine host, from DFPT.",
    "dimensionless", "host", HE, PU, EV_PRISTINE.format(d="1.4e-14"))
add("band_gap", "Band gap of the pristine host from the PBEsol(+U) calculation; not the nsc-dd hybrid gap.",
    "eV", "host", HE, PU, "equals bulk_visual_data gga_band_gap exactly (max abs diff 8.9e-16, 1745/1745 rows); "
    "differs from the hybrid band_gap field by up to 4.8 eV; identical across q",
    "K21 Sec. II I (PBEsol(+U) band gap); K21 Sec. III A for the hybrid gap")
add("formation_energy", "Host formation energy per atom retrieved from the Materials Project database "
    "(empirically corrected). NOT the vacancy target.", "eV/atom", "host", HE,
    "tabulated pristine-DFT value (lookup) or pristine unit-cell total energies", EV_Q,
    "K21 Sec. II A and II I (retrieved from the MPD)", "naming hazard: same word as the target, host property only")
for edge, long_ in (("vbm", "valence-band maximum"), ("cbm", "conduction-band minimum")):
    for orb in "spdf":
        add(f"{edge}_{orb}", f"Fraction of {orb}-orbital character in the pristine host's {long_} state "
            "(projection ratio).", "dimensionless", "host", HE, PU, EV_Q, LOC_I + "; ratios of i-orbital components at VBM and CBM")
add("ave_bec", "Spherically averaged Born effective charge (trace/3) of the O atom to be removed, in the pristine host (DFPT).",
    "e (elementary charge); negative for O", "site", SE, PU, EV_Q)
add("bader_charge", "Bader charge of the O atom to be removed, from the pristine host unit-cell all-electron density; "
    "net ionic charge, negative for the anion.", "e", "site", SE, PU,
    EV_PRISTINE.format(d="2.2e-16") + "; equals the pristine unit-cell Bader charge of the O site (not a defect-cell quantity)",
    "K21 Sec. II I (second type; Bader code of Henkelman et al., refs 50-51)")
add("bader_volume", "Bader (zero-flux) volume of the O atom to be removed, from the pristine host all-electron density.",
    "A^3", "site", SE, PU, EV_Q, "K21 Sec. II I (second type)")
add("o2p_center_from_vbm", "Centroid of the O-2p projected density of states of the O site, measured as its distance below "
    "the VBM (Deml et al. descriptor).", "eV", "site", SE, PU,
    "reproduced from the pristine unit-cell site-projected DOS shipped in bulk_visual_data.json "
    "(1745 rows; max dev 0.13 eV, integration window not documented); identical across q",
    "K21 Sec. II I (second type; Deml et al., ref. 7)")
NN = ("Neighbours are the periodic Voronoi neighbours of the O site in the pristine unit cell; weights are the "
      "Voronoi solid-angle fractions (sum 1).")
for stat, sdef in (("ave", "solid-angle-weighted mean"), ("max", "maximum"), ("min", "minimum")):
    ev = ("weighted mean recomputed exactly from pristine unit cell + per-atom pristine Bader charges (max abs diff 5.7e-15)"
          if stat == "ave" else
          "built from pristine unit-cell neighbours; max/min selection rule (weight > 1/12) reproduced for "
          + {"max": "99.8% of rows", "min": "72.8% of rows"}[stat] + ", exact rule not documented; identical across q")
    add(f"nn_{stat}_bader_charge", f"{sdef.capitalize()} of the Bader charge over the Voronoi neighbours of the O site. {NN}",
        "e", "site", SE, PU, ev, LOC_I + "; third type (neighbour information)")
for stat, sdef in (("ave", "solid-angle-weighted mean"), ("max", "maximum"), ("min", "minimum")):
    add(f"nn_{stat}_ave_bec", f"{sdef.capitalize()} of the spherically averaged Born effective charge over the Voronoi neighbours. {NN}",
        "e", "site", SE, PU, EV_Q, LOC_I + "; third type (neighbour information)")
for stat, sdef in (("ave", "solid-angle-weighted mean"), ("max", "maximum"), ("min", "minimum")):
    ev = ("weighted mean recomputed exactly from pristine structure and Pauling electronegativities (max abs diff 2.2e-15)"
          if stat == "ave" else
          "built from pristine structure and Pauling electronegativities; max/min selection reproduced for "
          + {"max": "74.3% of rows", "min": "97.7% of rows"}[stat] + ", exact rule not documented; identical across q")
    add(f"nn_{stat}_eleneg", f"{sdef.capitalize()} of the Pauling electronegativity over the Voronoi neighbours "
        f"(cations and O). {NN}", "Pauling units (dimensionless)", "site", SC, "none", ev,
        LOC_I + "; third type (neighbour information)")

cols = [c for c in pd.read_csv(ML, index_col=0, nrows=1).columns if c not in ("formula", "full_name", "vacancy_formation_energy")]
elements = [c for c in cols if c not in {r["name"] for r in rows}]
for el in elements:
    add(el, f"Fraction w_{el} of the solid angle of the O-site Voronoi cell (pristine unit cell) that faces {el} atoms.",
        "dimensionless (fraction, sums to 1 over elements)", "site", SC, "none",
        "recomputed exactly from the pristine unit-cell Voronoi tessellation (max abs diff 8.3e-16, 1745/1745 rows); identical across q",
        LOC_I + "; weights w_X of surfaces pointing to neighbouring X atoms")
assert [r["name"] for r in rows if r["name"] in cols] and sorted(r["name"] for r in rows) == sorted(cols), \
    (set(cols) ^ {r["name"] for r in rows})
rows.sort(key=lambda r: cols.index(r["name"]))
OUT.parent.mkdir(exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "descriptors;", sum(r["class"] == "DEFECT-DERIVED" for r in rows), "defect-derived")
