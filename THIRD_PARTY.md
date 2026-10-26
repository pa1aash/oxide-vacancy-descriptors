# Third-party data, weights and licences

This file lists what in or around this repository comes from third parties, under which licence, and where the
licence texts are. The code of this repository is MIT licensed (`LICENSE`); the data and results derived here are
released under CC BY 4.0 with the attribution below (`LICENSES/CC-BY-4.0.txt`).

## Oxygen-vacancy database of Kumagai et al. (source of every DFT value)

- Source: Y. Kumagai, N. Tsunoda, A. Takahashi, F. Oba, "Insights into oxygen vacancies from high-throughput
  first-principles calculations", Phys. Rev. Materials 5, 123803 (2021), doi:10.1103/PhysRevMaterials.5.123803.
- Repository: https://github.com/kumagai-group/oxygen_vacancies_db at commit
  `e310353ecbf698499c85a5ec5416b76e82dcecfd`. The repository has no LICENSE file; the licence statement is line 24
  of its README:

  > All the data is licensed under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

  Verified by fetching
  https://raw.githubusercontent.com/kumagai-group/oxygen_vacancies_db/e310353ecbf698499c85a5ec5416b76e82dcecfd/README.md
  on 2026-10-26 (7,100 bytes, sha256 `a61a3c2fdd2996ea3a3de3b18097047f7acee7f5fa737cc504dc9a2ee7469cbc`, the value
  listed for that file in `data/MANIFEST.sha256`).
- Licence: CC BY 4.0. Legal code in `LICENSES/CC-BY-4.0.txt`.
- Not redistributed here. `data/MANIFEST.sha256` lists the source URL, size and sha256 of every downloaded file.
- The 818 release supercell CIFs (`site_info/<formula>/supercell.cif` of the release) are inputs of the training
  stages. They are not in this repository; they are redistributed unchanged, under CC BY 4.0, with the processed
  inputs, available from the corresponding author.

## Derived data (this work)

The processed tables, graph-store manifests, splits (except `splits/kiyohara.json`, below), per-site prediction
tables, result records under `results/`, the CSV tables under `tables/` and the small tables under `data/` are
derived from the database above. They are released under CC BY 4.0. Attribution: cite Kumagai et al. (above) when
you use any of them. The job specifications (`specs/`) and the checkpoint hash lists are outputs of the code and are
released under its MIT licence.

## Kiyohara split (`splits/kiyohara.json`)

Host-ID lists reconstructed from S. Kiyohara, C. Shibui, S. Bae, Y. Kumagai, Phys. Rev. Lett. 135, 246101 (2025),
doi:10.1103/h66h-y5k6. The repository of that work (https://github.com/kumagai-group/ML_charged_defects, commit
`39db6d2d8f5653d6452c7877ecb2e296f04a7a01`, pinned in `data/MANIFEST.sha256`) states no licence, so no licence is
asserted for this file. Cite Kiyohara et al. when you use it.

## MACE-MP-0 foundation-model weights

- Weights: MACE-MP-0 medium, `2023-12-03-mace-128-L1_epoch-199.model` (release `mace_mp_0`), used unchanged for the
  host re-relaxation check. sha256 `01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845` (identity
  check in `results/mace_model.json`). Not in this repository.
- Licence: MIT License of ACEsuit/mace-foundations (copyright 2024 MACE-MP). Text in `LICENSES/MACE-MP-0.txt`.

## Licence texts in `LICENSES/`

| File | Source URL | Fetched (UTC) | Bytes | sha256 |
|---|---|---|---:|---|
| `LICENSES/CC-BY-4.0.txt` | https://creativecommons.org/licenses/by/4.0/legalcode.txt | 2026-10-26 | 18,657 | `9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411` |
| `LICENSES/MACE-MP-0.txt` | https://raw.githubusercontent.com/ACEsuit/mace-foundations/main/LICENSE | 2026-10-26 | 1,064 | `31ea0ccf7bc19797081bff51c7eff3a3927c8cb6c1d7a726dec3d157b436da1c` |

Both files are stored unchanged, byte for byte as fetched. The body of `LICENSE` (MIT) equals the SPDX MIT text
(https://raw.githubusercontent.com/spdx/license-list-data/main/text/MIT.txt, fetched 2026-10-26) apart from line
wrapping, with the copyright line "Copyright (c) 2026 Palaash Gang".

## Computing resources

This work used resources available through the National Research Platform (NRP) at the University of California, San Diego. NRP has been developed, and is supported in part, by funding from National Science Foundation, from awards 1730158, 1540112, 1541349, 1826967, 2112167, 2100237, and 2120019, as well as additional funding from community partners.

Weitzel et al., "The National Research Platform: Stretched, Multi-Tenant, Scientific Kubernetes Cluster",
PEARC '25, doi:10.1145/3708035.3736060.
