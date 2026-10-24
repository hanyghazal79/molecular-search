"""SOTA Experiment 4: Amortized Optimization.

Trains a single MLP surrogate once on all previously scored molecules.
At inference time, scores the ENTIRE pool in one forward pass (no oracle calls),
then returns the top-K diverse candidates.

This is "amortized" because the training cost is paid once and reused across
many inference calls. Useful for production where the client needs fast
re-ranking after each new scoring batch.

Usage:
    python 04_sota_experiments/search_amortized.py \\
        --library synthetic-library-100k \\
        --pool pool-100k.csv \\
        --target-sequence "PQITL..." \\
        --train-probes 100 --seed 42 \\
        --output 04_sota_experiments/results/amortized/run-s42.json
"""

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sota_utils import (
    set_seed, morgan_np, AmortizedMLP,
)
from chemistry import Library
from nim_oracle import NimOracle
from local_eval import validate_set, final_set

_MORGAN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)


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


def train_mlp(scored, smiles_map, epochs=100, lr=1e-3, device='cpu'):
    """Train MLP on (fingerprint, score) pairs. Returns trained model + scaler stats."""
    X, y, valid_pids = [], [], []
    for pid, score in scored:
        fp = morgan_np(smiles_map[pid])
        if fp is not None:
            X.append(fp)
            y.append(score)
            valid_pids.append(pid)
    if len(X) < 5:
        raise RuntimeError(f"Need at least 5 scored molecules to train MLP, have {len(X)}")

    X = torch.tensor(np.stack(X), dtype=torch.float, device=device)
    y = torch.tensor(y, dtype=torch.float, device=device)
    y_mean, y_std = y.mean(), y.std() + 1e-6
    y_norm = (y - y_mean) / y_std

    model = AmortizedMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(X)
        loss = F.mse_loss(pred, y_norm)
        loss.backward()
        opt.step()
    model.eval()
    return model, (y_mean.item(), y_std.item())


@torch.no_grad()
def score_all(model, scaler, smiles_map, device='cpu'):
    """Score ALL pool molecules in one forward pass. Returns (pids, scores)."""
    pids, X = [], []
    for pid, smi in smiles_map.items():
        fp = morgan_np(smi)
        if fp is not None:
            pids.append(pid)
            X.append(fp)
    if not X:
        return [], []
    X = torch.tensor(np.stack(X), dtype=torch.float, device=device)
    y_mean, y_std = scaler
    preds = (model(X) * y_std + y_mean).cpu().numpy()
    return pids, preds


def run(library_dir, pool_path, target_sequence, seed=42, train_probes=100,
        required=20, max_sim=0.6, msa=None, cache='nim_cache.json',
        output='amortized_result.json'):
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load pool
    pids, smiles_map = [], {}
    with open(pool_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('product_id') and row.get('smiles'):
                pids.append(row['product_id'])
                smiles_map[row['product_id']] = row['smiles']
    random.Random(seed).shuffle(pids)

    with Library(library_dir) as library:
        oracle = NimOracle(api_key=os.environ['NVIDIA_API_KEY'], library=library,
                           target_sequence=target_sequence, cache_path=cache, msa_path=msa)

        # Phase 1: collect training data via random sampling
        print(f"Phase 1: Collecting {train_probes} training samples via random sampling...")
        train_pids = pids[:train_probes]
        train_scores = oracle.score(train_pids)
        scored = [(p, s) for p, s in train_scores.items() if not math.isnan(s)]
        print(f"  Collected {len(scored)} valid scores")

        # Phase 2: train amortized model
        print(f"Phase 2: Training amortized MLP...")
        model, scaler = train_mlp(scored, smiles_map, epochs=100, device=device)

        # Phase 3: single-forward-pass inference over ENTIRE pool
        print(f"Phase 3: Scoring all {len(smiles_map)} molecules in one forward pass...")
        all_pids, all_scores = score_all(model, scaler, smiles_map, device=device)
        print(f"  Scored {len(all_pids)} molecules without additional oracle calls")

        # Phase 4: select top-K diverse candidates
        ranked = sorted(zip(all_pids, all_scores), key=lambda x: -x[1])
        top_200 = [p for p, _ in ranked[:200]]
        selected = maxmin_pick(top_200, required, library)

        # Phase 5: validate + final score with oracle (only 20 calls)
        print(f"Phase 5: Validating and scoring final {len(selected)} candidates...")
        ok, reasons, _ = validate_set(selected, library, max_sim)
        validation = {'valid': ok, 'reasons': reasons}

        final_result = {'valid': False, 'reasons': reasons, 'scores': {}}
        if ok:
            final_scores = oracle.score(selected)
            final_result = {'valid': True, 'scores': final_scores,
                            'mean_pIC50': float(np.nanmean(list(final_scores.values())))}

        result = {
            'method': 'amortized_mlp',
            'seed': seed,
            'train_samples': len(scored),
            'scored_all': len(all_pids),
            'selected': selected,
            'validation': validation,
            'final': final_result,
            'best_predicted': float(all_scores.max()) if len(all_scores) else float('nan'),
            'best_final': max((v for v in final_result['scores'].values()
                              if not math.isnan(v)), default=float('nan')),
        }
        output.write_text(json.dumps(result, indent=2))
        print(f"Done. best_final={result['best_final']:.2f} -> {output}")
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True)
    p.add_argument('--pool', required=True)
    p.add_argument('--target-sequence', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-probes', type=int, default=100)
    p.add_argument('--required', type=int, default=20)
    p.add_argument('--max-sim', type=float, default=0.6)
    p.add_argument('--msa')
    p.add_argument('--cache', default='nim_cache.json')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    return run(
        library_dir=args.library, pool_path=args.pool,
        target_sequence=args.target_sequence, seed=args.seed,
        train_probes=args.train_probes, required=args.required,
        max_sim=args.max_sim, msa=args.msa, cache=args.cache, output=args.output,
    )


if __name__ == '__main__':
    raise SystemExit(main())
