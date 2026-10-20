# Optimization runbook

## Setup (once)

```bash
python -m venv .venv
.venv/bin/pip install requests rdkit==2024.9.4 scikit-learn scipy
.venv/bin/pip freeze > requirements.lock
.venv/bin/python -c "import rdkit; print(rdkit.__version__)"   # 2024.09.4
export NVIDIA_API_KEY="nvapi-..."
```

Free NGC key: https://build.nvidia.com/mit/boltz2 (40 RPM, 10k/day).

## Phase 1 — Baseline and evaluation

### Step 1 — Build the synthetic library (no owner bundle needed)

```bash
.venv/bin/python make_synthetic_library.py synthetic-library
```

### Step 2 — Verify demo_pool resolves

```bash
.venv/bin/python - <<'PY'
import csv
from chemistry import Library
with Library("synthetic-library") as lib:
    ok = sum(lib.resolve(r["product_id"]) is not None
             for r in csv.DictReader(open("demo_pool.csv")))
print(f"{ok}/12 resolve")
PY
```

### Step 3 — Generate the SMILES pool

```bash
.venv/bin/python generate_pool.py --library synthetic-library --reaction rxn:1 \
    --count 100 --attempts 500 --seed 42 --output pool-100.csv
```

### Step 4 — NIM smoke test

```bash
.venv/bin/python - <<'PY'
import os
from chemistry import Library
from nim_oracle import NimOracle
HIV = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
with Library("synthetic-library") as lib:
    print(NimOracle(os.environ["NVIDIA_API_KEY"], lib, HIV).score(["rxn:1:1001:2001"]))
PY
```

Expect `{'rxn:1:1001:2001': 5.xx}`. If `NaN` or `NIM error`, supply an MSA.

### Step 5 — MSA (recommended)

```bash
.venv/bin/python make_nim_msa.py \
    "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF" msa.a3m
```

### Step 6 — Baseline (random strategy, matches Phase 2 budget)

```bash
mkdir -p results
.venv/bin/python search_nim.py --library synthetic-library --pool pool-100.csv \
    --target-sequence "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF" \
    --msa msa.a3m --strategy random --seed 42 \
    --probes 24 --batch-size 4 --required 3 --max-sim 0.6 \
    --output results/search-random-s42.json
```

## Phase 2 — Search optimization

### Step 7 — Ablation at fixed budget

```bash
for strat in greedy ucb ei; do
  .venv/bin/python search_nim.py --library synthetic-library --pool pool-100.csv \
      --target-sequence "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF" \
      --msa msa.a3m --strategy $strat --seed 42 \
      --probes 24 --batch-size 4 --required 3 --max-sim 0.6 \
      --output results/search-${strat}-s42.json
done
```

### Step 8 — Compare

```bash
.venv/bin/python metrics.py results/*.json --pool pool-100.csv \
    --csv-out results/nim-ablation.csv
```

Primary axis: `best_per_pair`. Report `validation_valid` and `div_max_sim` separately.

### Step 9 — Seed sweep (evidence for improvement claim)

```bash
for strat in random ei; do
  for seed in 42 43 44 45 46; do
    .venv/bin/python search_nim.py --library synthetic-library --pool pool-100.csv \
        --target-sequence "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF" \
        --msa msa.a3m --strategy $strat --seed $seed \
        --probes 24 --batch-size 4 --required 3 --max-sim 0.6 \
        --output results/search-${strat}-s${seed}.json
  done
done
.venv/bin/python metrics.py results/*.json --pool pool-100.csv \
    --csv-out results/phase2-seeds.csv
```

## Phase 3 — Validation and handover

```bash
sha256sum pool-100.csv synthetic-library/* > artifact-hashes.txt
```

Ship `results/`, `artifact-hashes.txt`, `requirements.lock`, `phase3-report.md`.

**Report must open with:**

> Oracle scores are real NVIDIA NIM Boltz-2 affinities. Set-level validation uses a local reimplementation of the frozen library's rules (`local_eval.py`), not the owner's evaluation service. Candidate quality against the owner's historical database and internal entropy gate is unverified pending receipt of the owner's frozen library bundle.

## Acceptance criteria

| Criterion | Threshold |
|---|---|
| `pairs_charged` | Equal to baseline at same probes |
| `validation_valid` | `True` on ≥ 4/5 EI seeds |
| `best_per_pair` | EI mean ≥ random mean + 2·SEM over 5 seeds |
| `div_max_sim` | ≤ 0.6 on ≥ 4/5 EI seeds |
| Reproduction | Rerun from scratch reproduces `phase2-seeds.csv` given seeds |

## Migration to owner's bundle

Two changes:
1. Replace `synthetic-library/` with owner's `library.json` + `molecules.sqlite`.
2. Rerun Steps 2–9 with the owner's pool generated from their library.

Code unchanged. That is the point of the design.