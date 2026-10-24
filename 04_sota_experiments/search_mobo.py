"""SOTA Experiment 3: Multi-Objective BO (subsampled Pareto).

Optimizes pIC50, -SA, novelty. Pareto-ranks only the top-N by predicted
pIC50, then MaxMin-selects a batch — O(N²) on N=2000 instead of O(100000²).
"""

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor

RDLogger.DisableLog('rdApp.*')

from sota_utils import set_seed, morgan_np, sa_score, pareto_rank
from chemistry import Library
from nim_oracle import NimOracle
from local_eval import validate_set, final_set

_MORGAN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)


def compute_objectives_for_subset(pids, smiles_map, library, history_fps):
    """Compute (SA, novelty) only for the given subset. Returns (pids, sa_arr, nov_arr)."""
    valid_pids, sas, novs = [], [], []
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
        sas.append(-sa)
        novs.append(novelty)
    return valid_pids, np.array(sas, dtype=np.float32), np.array(novs, dtype=np.float32)


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


def run(library_dir, pool_path, target_sequence, seed=42, probes=100, batch_size=10,
        required=20, max_sim=0.6, history_path=None, msa=None,
        cache='nim_cache.json', output='mobo_result.json',
        pareto_subsample=2000):
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

    # Load history fingerprints once
    history_fps = []
    if history_path and Path(history_path).exists():
        import gzip
        try:
            with gzip.open(history_path, 'rt', encoding='utf-8') as f:
                for i, row in enumerate(csv.DictReader(f)):
                    if i >= 3000:            # was 5000, lowered for speed
                        break
                    mol = Chem.MolFromSmiles(row.get('smiles', ''))
                    if mol:
                        history_fps.append(_MORGAN.GetFingerprint(mol))
            print(f"Loaded {len(history_fps)} history fingerprints")
        except Exception as e:
            print(f"History load failed: {e}, novelty=1.0")

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
            # --- Initial random batch ---
            if len(state['scored']) < 5:
                rng = random.Random(seed + len(state['scored']))
                batch = rng.sample(remaining, min(batch_size, len(remaining)))
            else:
                # --- Predict pIC50 for the full remaining pool via RF ---
                scored_pids = [p for p, _ in state['scored']]
                y = np.array([s for _, s in state['scored']], dtype=np.float32)
                X_train = np.stack([morgan_np(smiles_map[p]) for p in scored_pids])
                rf = RandomForestRegressor(n_estimators=100, random_state=seed,
                                           n_jobs=-1).fit(X_train, y)

                X_rem = np.stack([morgan_np(smiles_map[p]) for p in remaining])
                mu_pIC50 = np.array([t.predict(X_rem) for t in rf.estimators_]).mean(axis=0)

                # --- Subsample top-N by pIC50 before expensive Pareto ranking ---
                n_sub = min(pareto_subsample, len(remaining))
                top_idx = np.argsort(-mu_pIC50)[:n_sub]
                sub_pids = [remaining[i] for i in top_idx]

                # --- Compute SA + novelty only on the subsample ---
                valid_pids, sa_arr, nov_arr = compute_objectives_for_subset(
                    sub_pids, smiles_map, library, history_fps)

                if len(valid_pids) < batch_size:
                    batch = sub_pids[:batch_size]
                else:
                    # Re-align mu to the valid subset
                    valid_idx = [top_idx[sub_pids.index(p)] for p in valid_pids]
                    mu_valid = mu_pIC50[valid_idx]

                    # --- Pareto rank on the 2000-molecule subsample (fast) ---
                    objectives = np.stack([mu_valid, sa_arr, nov_arr], axis=1)
                    ranks = pareto_rank([row.tolist() for row in objectives])

                    front_0 = [valid_pids[i] for i in range(len(valid_pids)) if ranks[i] == 0]
                    if len(front_0) < batch_size:
                        front_1 = [valid_pids[i] for i in range(len(valid_pids)) if ranks[i] == 1]
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

        # Final selection
        if state['selected'] is None:
            scored_pids = [p for p, _ in state['scored']]
            _, sa_f, nov_f = compute_objectives_for_subset(scored_pids, smiles_map, library, history_fps)
            y = np.array([s for _, s in state['scored']], dtype=np.float32)
            if len(sa_f) == len(y):
                objectives = np.stack([y, sa_f, nov_f], axis=1)
                ranks = pareto_rank([row.tolist() for row in objectives])
                front_pids = [scored_pids[i] for i in range(len(scored_pids)) if ranks[i] == 0]
                if len(front_pids) < required:
                    front_pids += [scored_pids[i] for i in range(len(scored_pids)) if ranks[i] == 1]
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
    p.add_argument('--history')
    p.add_argument('--msa')
    p.add_argument('--cache', default='nim_cache.json')
    p.add_argument('--output', required=True)
    p.add_argument('--pareto-subsample', type=int, default=2000)
    args = p.parse_args()
    return run(
        library_dir=args.library, pool_path=args.pool, target_sequence=args.target_sequence,
        seed=args.seed, probes=args.probes, batch_size=args.batch_size,
        required=args.required, max_sim=args.max_sim, history_path=args.history,
        msa=args.msa, cache=args.cache, output=args.output,
        pareto_subsample=args.pareto_subsample,
    )


if __name__ == '__main__':
    raise SystemExit(main())
