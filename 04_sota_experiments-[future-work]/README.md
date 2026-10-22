# SOTA AI Experiments

This folder contains four State-of-the-Art (SOTA) molecular optimization
methods, implemented as standalone runnable scripts that plug into the
existing search pipeline.

All scripts share the same `NimOracle` and `Library` from the parent
directory. All outputs are JSON files compatible with `metrics.py`.

---

## Prerequisites

```bash
# From the repo root
pip install -r 04_sota_experiments/requirements-sota.txt

# For GNN experiment only:
pip install torch-geometric

# Verify the environment
python -c "import torch; print('torch', torch.__version__)"
python -c "import torch_geometric; print('pyg', torch_geometric.__version__)"
```

**On Kaggle:** select the **GPU T4** accelerator. `torch-geometric` installs
cleanly on Kaggle without needing `torch_scatter`/`pyg_lib` — the scripts
use only pure-PyTorch PyG operations.

---

## Experiment 1: GNN Surrogate (`search_gnn.py`)

**WHAT:** Replaces the RandomForest surrogate in Bayesian Optimization with a
Graph Neural Network ensemble (GIN + 3-model ensemble for uncertainty).
The GNN operates directly on the molecular graph — atoms as nodes, bonds as
edges — rather than on flattened fingerprints.

**WHY:** Fingerprints like Morgan ignore 3D geometry and long-range atom
relationships. GNNs learn message-passing rules across the graph, which
has been shown to improve sample efficiency for molecular property prediction
by 15–30% in benchmarks. The 3-model ensemble gives calibrated uncertainty
for Expected Improvement without needing a Gaussian Process.

**HOW:**
```bash
python 04_sota_experiments/search_gnn.py \
    --library synthetic-library-100k \
    --pool pool-100k.csv \
    --target-sequence "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF" \
    --strategy ei --probes 100 --seed 42 \
    --output 04_sota_experiments/results/gnn/ei-s42.json
```

The script falls back to RandomForest automatically if `torch_geometric` is
not installed — so it always runs, just without the GNN advantage.

**RESULT:** A JSON checkpoint file identical in shape to the RF-based
results. Compare `best_per_pair` in `metrics.py` to see if the GNN improved
sample efficiency.

**Runtime:** ~5–10 min for 100 probes on a T4 GPU (GNN training ~2 min per batch).

---

## Experiment 2: Reinforcement Learning (`search_rl.py`)

**WHAT:** A small character-level GRU is pre-trained on the pool's SMILES
strings, then fine-tuned with REINFORCE (policy gradient) to generate new
SMILES that maximize NIM Boltz-2 affinity. New molecules are then scored and
passed through a MaxMin diversity filter.

**WHY:** Library-based search is limited to what the client already
enumerated. RL can discover **novel chemical matter** that is not in the
pool. This is the frontier approach used by REINVENT and MolDQN.

**HOW:**
```bash
python 04_sota_experiments/search_rl.py \
    --library synthetic-library-100k \
    --pool pool-100k.csv \
    --target-sequence "PQITL..." \
    --pretrain-epochs 20 --rl-steps 100 --n-generated 200 \
    --seed 42 \
    --output 04_sota_experiments/results/rl/run-s42.json
```

**RESULT:** A JSON file with:
- `selected`: 20 novel SMILES discovered by RL (not from the pool)
- `scores`: their real NIM scores
- `best_score`: the best pIC50 found
- `n_valid` / `n_generated`: validity rate of the generator

**Runtime:** ~15–25 min on T4 (20 pretrain epochs + 100 RL steps + final scoring).

**Caveat:** This implementation uses a small GRU, not the full REINVENT
Transformer. It is production-grade for a first pass; for maximum quality,
replace `SMILESGRU` with a transformer or fine-tune a pretrained ChemBERTa.

---

## Experiment 3: Multi-Objective BO (`search_mobo.py`)

**WHAT:** Bayesian Optimization over **three objectives simultaneously**:
1. **pIC50** (predicted by RF ensemble, maximize)
2. **Negative SA score** (RDKit Ertl-Schuffenhauer, maximize)
3. **Novelty** (1 − Tanimoto to closest historical molecule, maximize)

At each batch, it computes the **Pareto front** of these three objectives
and selects the batch from the best frontier, then MaxMin-diversifies.

**WHY:** The client's `README.md` requires **novelty and diversity**. A
single-objective optimizer will find high-affinity molecules that are
hard to synthesize or too similar to known compounds. MOBO gives you a
**menu of trade-offs** — you can show the client "candidate A is highest
affinity, B is easiest to synthesize, C is most novel."

**HOW:**
```bash
python 04_sota_experiments/search_mobo.py \
    --library synthetic-library-100k \
    --pool pool-100k.csv \
    --history real_data/molecules.csv.gz \
    --target-sequence "PQITL..." \
    --probes 100 --seed 42 \
    --output 04_sota_experiments/results/mobo/run-s42.json
```

The `--history` flag is optional. If provided, the novelty objective is
computed against the client's historical database. Otherwise novelty=1.0.

**RESULT:** A JSON file where `selected` contains the Pareto-optimal
20-molecule set. Each molecule trades off differently across affinity,
synthesizability, and novelty — perfect for a client presentation.

**Runtime:** ~10–15 min for 100 probes.

---

## Experiment 4: Amortized Optimization (`search_amortized.py`)

**WHAT:** Trains an MLP once on all scored molecules, then scores the
**entire 100,000-molecule pool in a single forward pass** without any
additional oracle calls. Returns the top-20 diverse candidates.

**WHY:** If the client needs to re-rank 100K molecules daily, running the
oracle 100 times per day is impossible. Amortized inference reduces the
cost of each ranking from 100 oracle calls to **zero** after the initial
training. This is the fastest possible pipeline for production deployment.

**HOW:**
```bash
python 04_sota_experiments/search_amortized.py \
    --library synthetic-library-100k \
    --pool pool-100k.csv \
    --target-sequence "PQITL..." \
    --train-probes 100 --seed 42 \
    --output 04_sota_experiments/results/amortized/run-s42.json
```

The script has 5 phases:
1. Collect 100 training samples via random sampling (100 oracle calls)
2. Train MLP on (fingerprint, score) pairs
3. Score ALL pool molecules in one forward pass (0 oracle calls)
4. MaxMin-select top 20 diverse
5. Validate + score the 20 finalists with the oracle (20 calls)

**RESULT:** A JSON file with:
- `selected`: 20 candidates chosen from a full-pool MLP ranking
- `best_predicted`: the highest MLP-predicted score (before oracle validation)
- `best_final`: the highest actual NIM score among the 20 finalists

**Runtime:** ~5–8 min (100 random probes + 100 MLP epochs + 1 forward pass + 20 validation probes).

**Caveat:** MLP accuracy depends on the diversity of the 100 training
samples. For production, use `train_probes=200` or more.

---

## Recommended Execution Order

| Order | Experiment | Time | Why |
|---|---|---|---|
| **1** | Amortized | 5–8 min | Fastest, proves the full-pool inference concept |
| **2** | MOBO | 10–15 min | Proves multi-objective trade-offs work |
| **3** | GNN | 5–10 min | Proves graph learning improves sample efficiency |
| **4** | RL | 15–25 min | Proves novel molecule generation works |

**Total runtime:** ~35–60 min for all four experiments.

---

## Comparison Table

After all four experiments complete, run:

```bash
python metrics.py 04_sota_experiments/results/*/*.json \
    --pool pool-100k.csv \
    --csv-out 04_sota_experiments/results/SOTA_COMPARISON.csv
```

Then fill this table for the client report:

| Method | Best pIC50 | Pairs Used | Novel? | Synthesizable? | One-pass inference? |
|---|---|---|---|---|---|
| RF + EI (baseline) | ? | 100 | No | Maybe | No |
| **GNN Surrogate** | ? | 100 | No | Maybe | No |
| **RL Generator** | ? | ~300 | **Yes** | Varies | No |
| **MOBO** | ? | 100 | Yes (obj) | Yes (obj) | No |
| **Amortized** | ? | **120** | No | Maybe | **Yes** |

---

## Troubleshooting

**`torch_geometric not installed`** → GNN script falls back to RF. Install
`torch-geometric` for full functionality.

**`CUDA out of memory` on GNN** → reduce `hidden=64` to `hidden=32` in
`search_gnn.py::GINSurrogate.__init__`.

**`sascorer` import fails** → falls back to heuristic SA (rings + stereocenters).
Accuracy is lower but still usable.

**RL generates all-invalid SMILES** → increase `--pretrain-epochs` from 20 to 40,
or reduce `temperature` in `SMILESGRU.sample()` from 1.0 to 0.8.

**`Nim error ... 429`** → NIM rate limit. The scripts handle retries
automatically; reduce `--probes` if it happens repeatedly.