#!/usr/bin/env bash
# Reproduce the data download (headless, curl only), unpack, and verify data/MANIFEST.sha256.
# Usage: scripts/fetch_data.sh [--verify-only]
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw
UNP=$RAW/unpacked
DB_SHA=e310353ecbf698499c85a5ec5416b76e82dcecfd
ML_SHA=39db6d2d8f5653d6452c7877ecb2e296f04a7a01
DB_TGZ=$RAW/oxygen_vacancies_db-$DB_SHA.tar.gz
ML_TGZ=$RAW/ML_charged_defects-$ML_SHA.tar.gz

if [[ "${1:-}" != "--verify-only" ]]; then
  mkdir -p "$UNP/oxygen_vacancies_db" "$UNP/ML_charged_defects"
  [[ -f $DB_TGZ ]] || curl -fL --retry 3 -o "$DB_TGZ" \
    "https://codeload.github.com/kumagai-group/oxygen_vacancies_db/tar.gz/$DB_SHA"
  [[ -f $ML_TGZ ]] || curl -fL --retry 3 -o "$ML_TGZ" \
    "https://codeload.github.com/kumagai-group/ML_charged_defects/tar.gz/$ML_SHA"
  [[ -d $UNP/oxygen_vacancies_db/oxygen_vacancies_db_data ]] || \
    tar -xzf "$DB_TGZ" -C "$UNP/oxygen_vacancies_db" --strip-components=1
  [[ -d $UNP/ML_charged_defects/cgcnn ]] || \
    tar -xzf "$ML_TGZ" -C "$UNP/ML_charged_defects" --strip-components=1
  # site_info.tar.gz -> site_info/<formula>/{supercell.cif,cell_info.txt}
  [[ -d $UNP/oxygen_vacancies_db/site_info ]] || \
    tar -xzf "$UNP/oxygen_vacancies_db/site_info.tar.gz" -C "$UNP/oxygen_vacancies_db"
  # per-formula archives -> oxygen_vacancies_db_data/<formula>/ (inner per-defect archives stay packed)
  for t in "$UNP"/oxygen_vacancies_db/oxygen_vacancies_db_data/*.tar.gz; do
    [[ -d "${t%.tar.gz}" ]] || tar -xzf "$t" -C "$(dirname "$t")"
  done
fi
python3 scripts/make_manifest.py --verify
echo "manifest verification: OK"
