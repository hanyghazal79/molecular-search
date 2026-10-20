"""Surrogate-guided search using NVIDIA NIM Boltz-2 as the oracle.

Budget is counted in NIM oracle calls (one per probe). Local validation is free.

Requires:
    pip install requests rdkit==2024.9.4 scikit-learn scipy
    export NVIDIA_API_KEY="nvapi-..."
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
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

from chemistry import Library
from local_eval import validate_set, final_set
from nim_oracle import NimOracle


def _load_pool(path):
    with open(path, newline="") as stream:
        return [row["product_id"] for row in csv.DictReader(stream) if row.get("product_id")]


def _fp_matrix(pids, library):
    X = np.zeros((len(pids), 2048), dtype=np.float32)
    for i, pid in enumerate(pids):
        smi = library.resolve(pid)
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048), X[i])
    return X


def _maxmin(pids, k, library):
    if len(pids) <= k:
        return pids[:k]
    fps, valid = {}, []
    for pid in pids:
        smi = library.resolve(pid)
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps[pid] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        valid.append(pid)
    if len(valid) <= k:
        return valid
    chosen, remaining = [valid[0]], valid[1:]
    while len(chosen) < k and remaining:
        best, best_sim = None, 2.0
        for pid in remaining:
            sim = max(DataStructs.TanimotoSimilarity(fps[pid], fps[c]) for c in chosen)
            if sim < best_sim:
                best, best_sim = pid, sim
        chosen.append(best); remaining.remove(best)
    return chosen


def _select_batch(state, remaining, library, batch_size):
    strategy, scored = state["strategy"], state["scored"]
    if strategy == "random" or len(scored) < 2 * batch_size:
        rng = random.Random(state["seed"] + len(scored))
        return rng.sample(remaining, min(batch_size, len(remaining)))

    pids = [p for p, _ in scored]
    y = np.array([s for _, s in scored], dtype=np.float64)
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                               random_state=state["seed"], n_jobs=-1).fit(
        _fp_matrix(pids, library), y)
    X_rem = _fp_matrix(remaining, library)
    preds = np.array([t.predict(X_rem) for t in rf.estimators_])
    mu, sigma = preds.mean(axis=0), preds.std(axis=0) + 1e-9

    if strategy == "greedy":
        acq = mu
    elif strategy == "ucb":
        acq = mu + 2.0 * sigma
    else:  # ei
        best = y.max()
        z = (mu - best) / sigma
        acq = (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)

    ranked = [remaining[i] for i in np.argsort(-acq)]
    return _maxmin(ranked[:max(50, batch_size * 10)], batch_size, library)


def _state_path(output, strategy, seed):
    return Path(str(output) + f".{strategy}.s{seed}.state")


def _save(path, state):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def run(args):
    output = Path(args.output)
    state_path = _state_path(output, args.strategy, args.seed)

    if args.resume and not state_path.exists():
        raise SystemExit(
            f"--resume requested but {state_path} does not exist. "
            f"Check --strategy={args.strategy} and --seed={args.seed} "
            f"match the original run.")

    with Library(args.library) as library:
        if state_path.exists():
            state = json.loads(state_path.read_text())
            mismatches = []
            for k, arg in [("strategy", args.strategy), ("seed", args.seed),
                           ("batch_size", args.batch_size),
                           ("probes_planned", args.probes)]:
                if k in state and state[k] != arg:
                    mismatches.append(f"{k}: state={state[k]} arg={arg}")
            if mismatches:
                raise SystemExit(f"Cannot resume: {mismatches}")
            print(f"Resuming {state_path}: {len(state['scored'])} scored, "
                  f"{len(state.get('attempted', []))} attempted")
        else:
            if output.exists():
                raise SystemExit(f"{output} exists; use --resume or a new --output")
            pids = _load_pool(args.pool)
            random.Random(args.seed).shuffle(pids)
            state = {
                "seed": args.seed, "strategy": args.strategy,
                "batch_size": args.batch_size, "probes_planned": args.probes,
                "max_sim": args.max_sim, "target_sequence": args.target_sequence,
                "pool": pids, "scored": [], "attempted": [],
                "selected": None, "validation": None, "final": None,
                "history": [], "elapsed": 0.0,
            }
            _save(state_path, state)

        oracle = NimOracle(
            api_key=os.environ["NVIDIA_API_KEY"], library=library,
            target_sequence=state["target_sequence"],
            cache_path=args.cache, msa_path=args.msa)

        attempted = set(state.get("attempted", []))
        remaining = [p for p in state["pool"] if p not in attempted]

        started = time.monotonic()
        previous = state.get("elapsed", 0.0)

        while len(state["scored"]) < state["probes_planned"] and remaining:
            batch = [p for p in _select_batch(state, remaining, library, state["batch_size"])
                     if p not in attempted]
            if not batch:
                break
            scores = oracle.score(batch)
            new = 0
            for pid in batch:
                attempted.add(pid)
                s = scores.get(pid, float("nan"))
                if not math.isnan(s):
                    state["scored"].append((pid, s)); new += 1
            state["attempted"] = sorted(attempted)
            remaining = [p for p in remaining if p not in attempted]
            if state["scored"]:
                state["history"].append(max(s for _, s in state["scored"]))
            state["elapsed"] = previous + time.monotonic() - started
            _save(state_path, state)
            best = max(s for _, s in state["scored"]) if state["scored"] else float("nan")
            print(f"  +{new} valid | {len(state['scored'])}/{state['probes_planned']} "
                  f"probes | best={best:.2f}")

        if state["selected"] is None:
            state["scored"].sort(key=lambda x: -x[1])
            state["selected"] = [p for p, _ in state["scored"][:args.required]]
            _save(state_path, state)

        ok, reasons, _ = validate_set(state["selected"], library, args.max_sim)
        state["validation"] = {"valid": ok, "reasons": reasons}
        _save(state_path, state)
        if not ok:
            print("Validation failed:", reasons)
            return 1

        state["final"] = final_set(state["selected"], library, oracle, args.max_sim)
        _save(state_path, state)
        output.write_text(json.dumps(state, indent=2))

        best = max(s for _, s in state["scored"]) if state["scored"] else float("nan")
        print(f"\nDone. best_pIC50={best:.2f} pairs={len(state['scored'])} "
              f"selected={len(state['selected'])} → {output}")
        return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--target-sequence", required=True)
    p.add_argument("--strategy", choices=["random", "greedy", "ucb", "ei"], default="ei")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probes", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--required", type=int, default=3)
    p.add_argument("--max-sim", type=float, default=0.6)
    p.add_argument("--msa")
    p.add_argument("--cache", default="nim_cache.json")
    p.add_argument("--output", default="search-nim.json")
    p.add_argument("--resume", action="store_true")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())