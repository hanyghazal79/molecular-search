"""SOTA Experiment 3: Multi-Objective Bayesian Optimization.

Optimizes 3 objectives simultaneously:
  - pIC50 (maximize, predicted by RF ensemble)
  - Negative SA score (maximize, so easier to synthesize)
  - Novelty (maximize, Tanimoto distance to closest molecule in history)

Finds the Pareto front of trade-offs and selects a diverse final batch.

Usage:
    python 04_sota_experiments/search_mobo.py \\
        --library synthetic-library-100k \\
        --pool pool-100k.csv \\
        --target-sequence "PQITL..." \\
        --probes 100 --seed 42 \\
        --output 04_sota_experiments/results/mobo/run-s42.json
"""

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor

from sota_utils import (
    set_seed, morgan_np, sa_score, pareto_rank,
)
from chemistry import Library
from nim_oracle import NimOracle
from local_eval import validate_set, final_set

_MORGAN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def compute_objectives(pids, smiles_map, library, history_fps=None):
    """Compute (pIC50, -SA, novelty) for each molecule. Returns arrays + valid pids."""
    n = len(pids)
    scores, sas, novelties = [], [], []
    valid_pids, valid_smiles = [], []

    for pid in pids:
        smi = smiles_map.get(pid) or library.resolve(pid)
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = _MORGAN.GetFingerprint(mol)
        sa = sa_score(smi)
        if history_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fp, history_fps)
            novelty = 1.0 - max(sims) if sims else 1.0
        else:
            novelty = 1.0
        valid_pids.append(pid)
        valid_smiles.append(smi)
        sas.append(-sa)  # maximize negative SA = minimize SA
        novelties.append(novelty)

    return valid_pids, np.array(sas), np.array(novelties)


def maxmin_pick(pids, k, library):
    if len(pids) <= k:
        return pids[:k]
    fps, valid = {}, []
    for p in pids:
        smi = library.resolve(p)
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps[p] = _MORGAN.GetFingerprint(mol)
        valid.append(p)
    if len(valid) <= k:
        return valid
    chosen, remaining = [valid[0]], valid[1:]
    while len(chosen) < k and remaining:
        best_pid, best_sim = None, 2.0
        for p in remaining:
            sim = max(DataStructs.TanimotoSimilarity(fps[p], fps[c]) for c in chosen)
            if sim < best_sim:
                best_pid, best_sim = p, sim
        chosen.append(best_pid)
        remaining.remove(best_pid)
    return chosen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(library_dir, pool_path, target_sequence, seed=42, probes=100, batch_size=10,
        required=20, max_sim=0.6, history_path=None, msa=None,
        cache='nim_cache.json', output='mobo_result.json'):
    set_seed(seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state_path = output.with_suffix('.state.json')

    # Load pool
    pids, smiles_map = [], {}
    with open(pool_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('product_id') and row.get('smiles'):
                pids.append(row['product_id'])
                smiles_map[row['product_id']] = row['smiles']
    random.Random(seed).shuffle(pids)

    # Load history fingerprints (optional)
    history_fps = []
    if history_path and Path(history_path).exists():
        import gzip
        try:
            with gzip.open(history_path, 'rt', encoding='utf-8') as f:
                for i, row in enumerate(csv.DictReader(f)):
                    if i >= 5000:
                        break
                    smi = row.get('smiles', '')
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        history_fps.append(_MORGAN.GetFingerprint(mol))
            print(f"Loaded {len(history_fps)} history fingerprints (sampled)")
        except Exception as e:
            print(f"History load failed: {e}, using novelty=1.0")

    with Library(library_dir) as library:
        if state_path.exists():
            state = json.loads(state_path.read_text())
            print(f"Resuming: {len(state['scored'])} scored")
        else:
            state = {'seed': seed, 'scored': [], 'attempted': [],
                     'selected': None, 'validation': None, 'final': None}

        oracle = NimOracle(api_key=os.environ['NVIDIA_API_KEY'], library=library,
                           target_sequence=target_sequence, cache_path=cache, msa_path=msa)

        attempted = set(state['attempted'])
        remaining = [p for p in pids if p not in attempted]

        while len(state['scored']) < probes and remaining:
            if len(state['scored']) < 5:
                rng = random.Random(seed + len(state['scored']))
                batch = rng.sample(remaining, min(batch_size, len(remaining)))
            else:
                # Train surrogate on scored molecules
                scored_pids = [p for p, _ in state['scored']]
                y = np.array([s for _, s in state['scored']], dtype=np.float32)
                X_train = []
                for p in scored_pids:
                    fp = morgan_np(smiles_map[p])
                    if fp is not None:
                        X_train.append(fp)
                X_train = np.stack(X_train)
                rf = RandomForestRegressor(n_estimators=100, random_state=seed,
                                           n_jobs=-1).fit(X_train, y)

                # Predict on remaining + compute other objectives
                valid_rem, sa_arr, nov_arr = compute_objectives(
                    remaining, smiles_map, library, history_fps)
                X_rem = []
                for p in valid_rem:
                    fp = morgan_np(smiles_map[p])
                    X_rem.append(fp if fp is not None else np.zeros(2048, dtype=np.float32))
                X_rem = np.stack(X_rem)

                preds = np.array([t.predict(X_rem) for t in rf.estimators_])
                mu_pIC50 = preds.mean(axis=0)

                # 3-objective Pareto ranking
                objectives = np.stack([mu_pIC50, sa_arr, nov_arr], axis=1)
                ranks = pareto_rank([row.tolist() for row in objectives])

                # Select from best front with diversity
                front_0 = [valid_rem[i] for i in range(len(valid_rem)) if ranks[i] == 0]
                if len(front_0) < batch_size:
                    front_1 = [valid_rem[i] for i in range(len(valid_rem)) if ranks[i] == 1]
                    front_0 += front_1
                batch = maxmin_pick(front_0[:200], min(batch_size, len(front_0)), library)

            batch = [p for p in batch if p not in attempted][:batch_size]
            if not batch:
                break

            scores = oracle.score(batch)
            new = 0
            for pid in batch:
                attempted.add(pid)
                s = scores.get(pid, float('nan'))
                if not math.isnan(s):
                    state['scored'].append((pid, s))
                    new += 1
            state['attempted'] = sorted(attempted)
            remaining = [p for p in remaining if p not in attempted]
            state_path.write_text(json.dumps(state, indent=2))
            best = max(s for _, s in state['scored']) if state['scored'] else float('nan')
            print(f"  +{new} | {len(state['scored'])}/{probes} | best={best:.2f}")

        # Final selection: Pareto front of actual scores
        if state['selected'] is None:
            scored_pids = [p for p, _ in state['scored']]
            _, sa_f, nov_f = compute_objectives(scored_pids, smiles_map, library, history_fps)
            y = np.array([s for _, s in state['scored']], dtype=np.float32)
            valid_scored = [(p, s) for (p, s) in state['scored']
                            if p in set(smiles_map)]
            if len(valid_scored) == len(sa_f):
                objectives = np.stack([y, sa_f, nov_f], axis=1)
                ranks = pareto_rank([row.tolist() for row in objectives])
                front_pids = [valid_scored[i][0] for i in range(len(valid_scored)) if ranks[i] == 0]
                if len(front_pids) < required:
                    front_pids += [valid_scored[i][0] for i in range(len(valid_scored)) if ranks[i] == 1]
                state['selected'] = maxmin_pick(front_pids, required, library)
            else:
                state['scored'].sort(key=lambda x: -x[1])
                state['selected'] = maxmin_pick([p for p, _ in state['scored']], required, library)
            state_path.write_text(json.dumps(state, indent=2))

        ok, reasons, _ = validate_set(state['selected'], library, max_sim)
        state['validation'] = {'valid': ok, 'reasons': reasons}
        state_path.write_text(json.dumps(state, indent=2))
        if not ok:
            output.write_text(json.dumps(state, indent=2))
            return 1

        state['final'] = final_set(state['selected'], library, oracle, max_sim)
        state_path.write_text(json.dumps(state, indent=2))
        output.write_text(json.dumps(state, indent=2))
        best = max(s for _, s in state['scored']) if state['scored'] else float('nan')
        print(f"Done. best={best:.2f} -> {output}")
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True)
    p.add_argument('--pool', required=True)
    p.add_argument('--target-sequence', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--probes', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=10)
    p.add_argument('--required', type=int, default=20)
    p.add_argument('--max-sim', type=float, default=0.6)
    p.add_argument('--history', help='Path to molecules.csv.gz for novelty objective')
    p.add_argument('--msa')
    p.add_argument('--cache', default='nim_cache.json')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    return run(
        library_dir=args.library, pool_path=args.pool, target_sequence=args.target_sequence,
        seed=args.seed, probes=args.probes, batch_size=args.batch_size,
        required=args.required, max_sim=args.max_sim, history_path=args.history,
        msa=args.msa, cache=args.cache, output=args.output,
    )


if __name__ == '__main__':
    raise SystemExit(main())