import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

RDLogger.DisableLog('rdApp.*')
from chemistry import Library
from local_eval import validate_set, final_set
from nim_oracle import NimOracle

_MORGAN_GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)

def _load_pool(path):
    pids, smiles_map = [], {}
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            pid = row.get("product_id")
            smi = row.get("smiles")
            if pid and smi:
                pids.append(pid)
                smiles_map[pid] = smi
    return pids, smiles_map

def _maxmin(pids, k, fps_cache):
    if len(pids) <= k:
        return pids[:k]
    valid = [pid for pid in pids if pid in fps_cache]
    if len(valid) <= k:
        return valid
    chosen, remaining = [valid[0]], valid[1:]
    while len(chosen) < k and remaining:
        best, best_sim = None, 2.0
        for pid in remaining:
            sim = max(DataStructs.TanimotoSimilarity(fps_cache[pid], fps_cache[c]) for c in chosen)
            if sim < best_sim:
                best, best_sim = pid, sim
        chosen.append(best); remaining.remove(best)
    return chosen

def _select_batch(state, remaining, batch_size, fps_cache):
    strategy, scored = state["strategy"], state["scored"]
    if strategy == "random" or len(scored) < 2 * batch_size:
        rng = random.Random(state["seed"] + len(scored))
        sample_size = min(len(remaining), batch_size * 10)
        shortlist = rng.sample(remaining, sample_size)
        return _maxmin(shortlist, batch_size, fps_cache)

    pids = [p for p, _ in scored if p in fps_cache]
    y = np.array([s for p, s in scored if p in fps_cache], dtype=np.float64)
    
    X_train = np.zeros((len(pids), 2048), dtype=np.float32)
    for i, pid in enumerate(pids):
        DataStructs.ConvertToNumpyArray(fps_cache[pid], X_train[i])
        
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                               random_state=state["seed"], n_jobs=-1).fit(X_train, y)
    
    X_rem = np.zeros((len(remaining), 2048), dtype=np.float32)
    valid_rem = []
    for i, pid in enumerate(remaining):
        if pid in fps_cache:
            DataStructs.ConvertToNumpyArray(fps_cache[pid], X_rem[i])
            valid_rem.append(pid)
            
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

    ranked = [valid_rem[i] for i in np.argsort(-acq)]
    shortlist = ranked[:max(50, batch_size * 10)]
    return _maxmin(shortlist, batch_size, fps_cache)

def _state_path(output, strategy, seed):
    return Path(str(output) + f".{strategy}.s{seed}.state")

def _save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)

def run(library_dir, pool_path, target_sequence, strategy="ei", seed=42, probes=24,
        batch_size=4, required=3, max_sim=0.6, msa=None, cache="nim_cache.json",
        output="search-nim.json", resume=False):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state_path = _state_path(output, strategy, seed)

    if resume and not state_path.exists():
        raise SystemExit(f"--resume requested but {state_path} does not exist.")

    with Library(library_dir) as library:
        if state_path.exists():
            state = json.loads(state_path.read_text())
            print(f"Resuming {state_path}: {len(state['scored'])} scored")
        else:
            if output.exists():
                raise SystemExit(f"{output} exists; use --resume or a new --output")
            pids, smiles_map = _load_pool(pool_path)
            random.Random(seed).shuffle(pids)
            state = {
                "seed": seed, "strategy": strategy,
                "batch_size": batch_size, "probes_planned": probes,
                "max_sim": max_sim, "target_sequence": target_sequence,
                "pool": pids, "scored": [], "attempted": [],
                "smiles_map": smiles_map,
                "selected": None, "validation": None, "final": None,
                "history": [], "elapsed": 0.0,
            }
            _save(state_path, state)

        print("Precomputing fingerprints...")
        fps_cache = {}
        smiles_map = state.get("smiles_map", {})
        for pid in state["pool"]:
            smi = smiles_map.get(pid)
            if smi:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    fps_cache[pid] = _MORGAN_GEN.GetFingerprint(mol)
        print(f"Cached {len(fps_cache)} fingerprints.")

        oracle = NimOracle(
            api_key=os.environ["NVIDIA_API_KEY"], library=library,
            target_sequence=state["target_sequence"],
            cache_path=cache, msa_path=msa)

        attempted = set(state.get("attempted", []))
        remaining = [p for p in state["pool"] if p not in attempted]

        started = time.monotonic()
        previous = state.get("elapsed", 0.0)

        while len(state["scored"]) < state["probes_planned"] and remaining:
            batch = [p for p in _select_batch(state, remaining, state["batch_size"], fps_cache)
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
            scored_pids = [p for p, _ in state["scored"]]
            selected = []
            for pid in scored_pids:
                if pid not in fps_cache:
                    continue
                fp = fps_cache[pid]
                is_diverse = True
                for s in selected:
                    if DataStructs.TanimotoSimilarity(fp, fps_cache[s]) > max_sim:
                        is_diverse = False
                        break
                if is_diverse:
                    selected.append(pid)
                    if len(selected) == required:
                        break
            if len(selected) < required:
                for pid in scored_pids:
                    if pid not in selected:
                        selected.append(pid)
                        if len(selected) == required:
                            break
            state["selected"] = selected
            _save(state_path, state)

        ok, reasons, _ = validate_set(state["selected"], library, max_sim)
        state["validation"] = {"valid": ok, "reasons": reasons}
        _save(state_path, state)
        if not ok:
            print("Validation failed:", reasons)
            return 1

        state["final"] = final_set(state["selected"], library, oracle, max_sim)
        _save(state_path, state)
        output.write_text(json.dumps(state, indent=2))

        best = max(s for _, s in state["scored"]) if state["scored"] else float("nan")
        print(f"\nDone. best_pIC50={best:.2f} pairs={len(state['scored'])} "
              f"selected={len(state['selected'])} → {output}")
        return 0

def search_main(library_dir, pool_path, target_sequence, strategy="ei", seed=42, probes=24,
         batch_size=4, required=3, max_sim=0.6, msa=None, cache="nim_cache.json",
         output="search-nim.json", resume=False):
    return run(
        library_dir=library_dir, pool_path=pool_path, target_sequence=target_sequence,
        strategy=strategy, seed=seed, probes=probes, batch_size=batch_size,
        required=required, max_sim=max_sim, msa=msa, cache=cache,
        output=output, resume=resume
    )
