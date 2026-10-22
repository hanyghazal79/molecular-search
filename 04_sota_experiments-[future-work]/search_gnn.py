"""SOTA Experiment 1: GNN Surrogate for Bayesian Optimization.

Replaces the RandomForest surrogate with a Graph Neural Network that
operates on molecular graphs directly.

Falls back to RandomForest automatically if torch_geometric is not installed.

Usage:
    python 04_sota_experiments/search_gnn.py \\
        --library synthetic-library-100k \\
        --pool pool-100k.csv \\
        --target-sequence "PQITL..." \\
        --strategy ei --probes 100 --seed 42 \\
        --output 04_sota_experiments/results/gnn/ei-s42.json
"""

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.stats import norm

from sota_utils import (
    HAS_PYG, set_seed, smiles_to_graph, morgan_np,
)
from chemistry import Library
from nim_oracle import NimOracle
from local_eval import validate_set, final_set

_MORGAN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)


# ---------------------------------------------------------------------------
# GNN Model
# ---------------------------------------------------------------------------

if HAS_PYG:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GINConv, global_add_pool

    class GINSurrogate(nn.Module):
        """GIN-based surrogate: predicts scalar score from molecular graph."""

        def __init__(self, in_dim, hidden=64, layers=3):
            super().__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            cur_dim = in_dim
            for _ in range(layers):
                mlp = nn.Sequential(
                    nn.Linear(cur_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                )
                self.convs.append(GINConv(mlp))
                self.bns.append(nn.BatchNorm1d(hidden))
                cur_dim = hidden
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
            # Global pooling
            x = global_add_pool(x, batch)
            return self.head(x).squeeze(-1)

    class GNNEnsemble(nn.Module):
        """Ensemble of GIN models for uncertainty estimation."""

        def __init__(self, in_dim, hidden=64, layers=3, n_models=3):
            super().__init__()
            self.models = nn.ModuleList([
                GINSurrogate(in_dim, hidden, layers) for _ in range(n_models)
            ])

        def fit(self, data_list, y, epochs=30, lr=1e-3, device='cpu'):
            y_tensor = torch.tensor(y, dtype=torch.float, device=device)
            for model in self.models:
                model.to(device)
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                for _ in range(epochs):
                    optimizer.zero_grad()
                    batch = Batch.from_data_list(data_list).to(device)
                    pred = model(batch)
                    loss = F.mse_loss(pred, y_tensor)
                    loss.backward()
                    optimizer.step()
            return self

        def predict_with_std(self, data_list, device='cpu'):
            preds = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    batch = Batch.from_data_list(data_list).to(device)
                    preds.append(model(batch).cpu().numpy())
            preds = np.array(preds)  # (n_models, n_samples)
            return preds.mean(axis=0), preds.std(axis=0) + 1e-9


# ---------------------------------------------------------------------------
# Fallback: RandomForest surrogate
# ---------------------------------------------------------------------------

def _rf_predict(scored_pids, y, remaining_pids, library):
    from sklearn.ensemble import RandomForestRegressor
    X_train = np.stack([morgan_np(library.resolve(p)) for p in scored_pids])
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                               random_state=42, n_jobs=-1).fit(X_train, y)
    X_rem, valid_pids = [], []
    for p in remaining_pids:
        smi = library.resolve(p)
        if not smi:
            continue
        fp = morgan_np(smi)
        if fp is not None:
            X_rem.append(fp)
            valid_pids.append(p)
    preds = np.array([t.predict(np.stack(X_rem)) for t in rf.estimators_])
    return valid_pids, preds.mean(axis=0), preds.std(axis=0) + 1e-9


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

def compute_acquisition(mu, sigma, best, strategy):
    if strategy == 'greedy':
        return mu
    if strategy == 'ucb':
        return mu + 2.0 * sigma
    # ei
    z = (mu - best) / sigma
    return (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)


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
# Main loop
# ---------------------------------------------------------------------------

def run(library_dir, pool_path, target_sequence, strategy='ei', seed=42, probes=100,
        batch_size=10, required=20, max_sim=0.6, msa=None, cache='nim_cache.json',
        output='gnn_result.json', device=None):
    set_seed(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
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

    with Library(library_dir) as library:
        # Load or init state
        if state_path.exists():
            state = json.loads(state_path.read_text())
            print(f"Resuming: {len(state['scored'])} scored")
        else:
            state = {
                'strategy': strategy, 'seed': seed, 'scored': [], 'attempted': [],
                'selected': None, 'validation': None, 'final': None,
            }

        oracle = NimOracle(api_key=os.environ['NVIDIA_API_KEY'], library=library,
                           target_sequence=target_sequence, cache_path=cache, msa_path=msa)

        attempted = set(state['attempted'])
        remaining = [p for p in pids if p not in attempted]
        started = time.monotonic()

        while len(state['scored']) < probes and remaining:
            # Select batch
            if len(state['scored']) < 2 * batch_size or strategy == 'random':
                rng = random.Random(seed + len(state['scored']))
                batch = rng.sample(remaining, min(batch_size, len(remaining)))
            else:
                scored_pids = [p for p, _ in state['scored']]
                y = np.array([s for _, s in state['scored']], dtype=np.float32)

                if HAS_PYG and len(scored_pids) >= 5:
                    try:
                        graphs = []
                        valid_idx = []
                        for i, p in enumerate(scored_pids):
                            g = smiles_to_graph(smiles_map[p])
                            if g is not None:
                                graphs.append(g)
                                valid_idx.append(i)
                        y_filtered = y[valid_idx]
                        in_dim = graphs[0].x.size(1)
                        ens = GNNEnsemble(in_dim=in_dim, hidden=64, layers=2, n_models=3)
                        ens.fit(graphs, y_filtered, epochs=25, lr=1e-3, device=device)

                        rem_graphs, rem_pids = [], []
                        for p in remaining:
                            g = smiles_to_graph(smiles_map[p])
                            if g is not None:
                                rem_graphs.append(g)
                                rem_pids.append(p)
                        mu, sigma = ens.predict_with_std(rem_graphs, device=device)
                        valid_rem = rem_pids
                    except Exception as e:
                        print(f"  GNN failed ({e}), falling back to RF")
                        valid_rem, mu, sigma = _rf_predict(scored_pids, y, remaining, library)
                else:
                    valid_rem, mu, sigma = _rf_predict(scored_pids, y, remaining, library)

                acq = compute_acquisition(mu, sigma, float(y.max()), strategy)
                ranked = [valid_rem[i] for i in np.argsort(-acq)]
                shortlist = ranked[:max(50, batch_size * 10)]
                batch = maxmin_pick(shortlist, min(batch_size, len(shortlist)), library)

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
            state['scored'].sort(key=lambda x: -x[1])
            scored_pids = [p for p, _ in state['scored']]
            selected = maxmin_pick(scored_pids, required, library)
            state['selected'] = selected
            state_path.write_text(json.dumps(state, indent=2))

        ok, reasons, _ = validate_set(state['selected'], library, max_sim)
        state['validation'] = {'valid': ok, 'reasons': reasons}
        state_path.write_text(json.dumps(state, indent=2))
        if not ok:
            print("Validation failed:", reasons)
            output.write_text(json.dumps(state, indent=2))
            return 1

        state['final'] = final_set(state['selected'], library, oracle, max_sim)
        state_path.write_text(json.dumps(state, indent=2))
        output.write_text(json.dumps(state, indent=2))
        best = max(s for _, s in state['scored']) if state['scored'] else float('nan')
        print(f"Done. best={best:.2f} pairs={len(state['scored'])} -> {output}")
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True)
    p.add_argument('--pool', required=True)
    p.add_argument('--target-sequence', required=True)
    p.add_argument('--strategy', choices=['random', 'greedy', 'ucb', 'ei'], default='ei')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--probes', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=10)
    p.add_argument('--required', type=int, default=20)
    p.add_argument('--max-sim', type=float, default=0.6)
    p.add_argument('--msa')
    p.add_argument('--cache', default='nim_cache.json')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    return run(
        library_dir=args.library, pool_path=args.pool, target_sequence=args.target_sequence,
        strategy=args.strategy, seed=args.seed, probes=args.probes,
        batch_size=args.batch_size, required=args.required, max_sim=args.max_sim,
        msa=args.msa, cache=args.cache, output=args.output,
    )


if __name__ == '__main__':
    raise SystemExit(main())