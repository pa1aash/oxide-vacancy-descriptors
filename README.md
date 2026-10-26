# oxide-vacancy-descriptors

An open framework for predicting the formation energy of the neutral oxygen vacancy in non-magnetic oxides with
graph networks, and for measuring what explicit electronic descriptors of the host add to the crystal structure as
the number of training oxides grows. All models share a MEGNet backbone and read the pristine supercell with the
vacancy oxygen flagged:

- **S** reads the structure only;
- **D** also receives the density-functional electronic descriptors of the pristine host and site; **D\*** is the
  same descriptor network after a wider hyperparameter search;
- **P** is a staged network: a descriptor predictor (P1) estimates the descriptors from the structure and passes
  them to D;
- baselines: random forests on the released descriptors and a two-parameter physics floor (oxide formation energy
  and band gap, B0).

Test and training oxides never share a formula, learning curves are measured against the number of training hosts,
and both the training subsets and the seeds are resampled. This repository holds the package (`src/dftgnn`), the
analysis code, the job specifications and splits of every run, and the stored result records with the per-site
predictions of every run. From these it reproduces the numbers, tables and figures of the associated study by
re-derivation from the stored records: no retraining, no GPU. No new DFT calculations were performed.

## Quick start

```
make env                 # conda env "dftgnn" from env/environment-local.yml, package installed in editable mode
conda activate dftgnn
make all                 # env, test, specs-check, numbers, tables, figures
make help                # the list below
```

| Target | What it does | Runtime (laptop with an 8-core Apple-silicon CPU; no GPU) |
|---|---|---|
| `make env` | create or update the conda environment, `pip install -e . --no-deps` | about 10 min (measured; depends on download speed) |
| `make test` | unit tests of the package and the analysis code (`tests/`, `analysis/tests/`) | about 3 to 5 min |
| `make specs-check` | every job spec against its manifest, the splits, the configs and the run records; run ids recomputed | about 25 s |
| `make numbers` | re-derives the headline quantities from the per-site predictions and compares each with its stored record | about 1 min |
| `make tables` | rebuilds the 18 CSV tables in `tables/` from the stored records, cell by cell | about 1 s |
| `make figures` | regenerates the figures from the stored records as PDF into `figures/main/` and `figures/si/` | about 10 s |

`make numbers` prints one table (quantity, value, 95% interval, stored value, match) and exits with status 1 if any
value differs from its stored record by more than 1e-9. `make tables` prints PASS or FAIL per table (`python
reproduce/tables.py --write` rewrites them). `make figures` lists each figure as OK or SKIPPED with the reason: the
schematic figures need drawing sources that are not here, and the two structure renders are shipped as PNG.

## Layout

```
src/dftgnn/        package: data, graphs, models (MEGNet, CGCNN), train, tune, split, stats, baselines, probe, mlip
analysis/          analysis code that reads the stored predictions (with its tests in analysis/tests/)
scripts/           entry points that (re)generate results; scripts/queue/ writes job specs and runs workers
reproduce/         numbers.py, tables.py, figures.py, check_specs.py (the make targets above)
configs/           config.yaml (protocol constants) and the tuned hyperparameters (tuned_*.yaml)
splits/ specs/     formula-grouped splits; job specifications of every run (<stage>.jsonl + manifest)
results/           result records (JSON, with config, code commit, package versions, seeds) and per-site
                   prediction tables (parquet), one directory per stage; named analysis records
tables/            the 18 tables as CSV; tables/sources/ maps every numeric cell to its record and JSON pointer
figures/           paper/ figure scripts, code/ helpers; main/ and si/ figure inputs and renders
data/              small tracked tables and manifests (data/MANIFEST.sha256: every downloaded source file)
checkpoints/       checkpoint hash lists (the checkpoints are not in this repository)
env/               conda environments, lock files and the GPU container
LICENSES/          licence texts of third-party material (see THIRD_PARTY.md)
```

## Data source and attribution

Every formation energy, descriptor and structure comes from the oxygen-vacancy database released with
Y. Kumagai, N. Tsunoda, A. Takahashi and F. Oba, "Insights into oxygen vacancies from high-throughput first-principles
calculations", Phys. Rev. Materials 5, 123803 (2021)
(https://github.com/kumagai-group/oxygen_vacancies_db, CC BY 4.0). Cite that article when you use any derived file.
The database itself is not redistributed; `data/MANIFEST.sha256` pins every downloaded file by URL and sha256.

## Checkpoints and processed inputs

The trained checkpoints are not on GitHub. Their sha256 lists are in `checkpoints/` (stages without a cluster
manifest) and `results/nrp/artefacts/<stage>.sha256`; each result record also names its checkpoint and hash. The
processed inputs (`data/processed`: universe table and graph store) are not in this repository either. Both are
available from the corresponding author, Palaash Gang (palaash.gang@indusschoolpune.com); check every file you
receive against its listed sha256.

## Rerunning training

Retraining is not needed to reproduce the results. To rerun a stage, use the GPU environment
(`env/environment-gpu.yml` constrained by `env/lock-gpu.txt`, or the container `env/Dockerfile.gpu`), place the
processed inputs under `data/processed`, and drive the runs with `scripts/queue/`: `make_specs.py` writes or checks
the job specifications (`python scripts/queue/make_specs.py --check` regenerates all 18 spec files byte for byte; it
also needs `data/processed`), `enqueue.py` queues a stage and `worker.py` trains the queued runs, each writing a
result record that can be compared with the stored one. Retraining needs a CUDA GPU; on the same GPU class it is
bitwise repeatable, across GPU classes it is not.

## Licences

Code: MIT (`LICENSE`). Data, splits and results: CC BY 4.0 (`LICENSES/CC-BY-4.0.txt`), with the attribution above.
`splits/kiyohara.json` and the MACE-MP-0 weights used for the re-relaxation check carry their own terms; see
`THIRD_PARTY.md`.

## How to cite

See `CITATION.cff` (also used by the "Cite this repository" button on GitHub), and cite the data source above.

## Authors

Palaash Gang (Indus International School, Pune, India; ORCID 0009-0006-7448-9488; corresponding author),
Aiden Lee (Westview High School, San Diego, CA, USA; ORCID 0009-0007-3281-8420) and
Qi Wu (University of California, San Diego, Department of Chemistry & Biochemistry; ORCID 0009-0006-1874-6478).
