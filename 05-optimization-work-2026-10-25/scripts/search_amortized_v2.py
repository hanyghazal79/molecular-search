"""SOTA Experiment 4 v2: Amortized optimization with pooled labels + HistGB + descriptors.

Item 15-17 upgrades over v1:
  - Pools all available Boltz-2 labels from disk (218+ from prior runs,
    295 after GNN fix) instead of only 100 fresh probes.
  - Swaps the MLP for HistGradientBoosting (or XGBoost if installed).
  - Concats ~210 RDKit descriptors to the 2048-bit Morgan fingerprint.
  - Excludes training SMILES from the candidate pool so final selections
    are genuinely held out.

Zero behavior change to the scoring/validation path — uses the same
NimOracle, local_eval.validate_set, and local_eval.final_set as v1.

Usage:
    python 04_sota_experiments/search_amortized_v2.py \\
        --library synthetic-library-100000 \\
        --pool pool-100k.csv \\
        --target-sequence "PQITL..." \\
        --required 20 --seed 42 \\
        --output 04_sota_experiments/results/amortized/run-s42-v2.json
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import HistGradientBoostingRegressor

RDLogger.DisableLog("rdApp.*")

from chemistry import Library
from nim_oracle import NimOracle
from local_eval import validate_set, final_set

_MORGAN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)

# SOTA_ROOT = the directory containing this script (04_sota_experiments).
# Its parent (/kaggle/working) is where chemistry.py, nim_oracle.py, etc. live
# and where sys.path must point for imports to work.
_SCRIPT_DIR  = Path(__file__).resolve().parent          # 04_sota_experiments
_WORKING_DIR = _SCRIPT_DIR.parent                       # /kaggle/working

if str(_WORKING_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKING_DIR))

SOTA_ROOT = _SCRIPT_DIR


def fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(_MORGAN.GetFingerprint(mol), arr)
    return arr


def descriptors_vec(mol):
    if mol is None:
        return np.zeros(len(Descriptors._descList), dtype=np.float32)
    vals = []
    for _, fn in Descriptors._descList:
        try:
            v = fn(mol)
            if v is None or not np.isfinite(v):
                v = 0.0
            else:
                v = float(np.clip(v, -1e6, 1e6))
            vals.append(v)
        except Exception:
            vals.append(0.0)
    return np.array(vals, dtype=np.float32)


def featurize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    f = fp(smi)
    d = descriptors_vec(mol)
    return np.concatenate([f, d])


def load_pool_labels():
    """Reconstruct the pooled Boltz-2 label set from disk (same as Cell C)."""
    pid_to_smiles = {}
    pool_path = Path("/kaggle/working/pool-100k.csv")
    with pool_path.open() as f:
        for row in csv.DictReader(f):
            pid_to_smiles[row["product_id"]] = row["smiles"]

    scored = {}
    for state_path in glob.glob(str(SOTA_ROOT / "results" / "*" / "*.state.json")):
        state = json.loads(Path(state_path).read_text())
        smi_map = state.get("smiles_map") or pid_to_smiles
        for pid, score in state.get("scored", []):
            smi = smi_map.get(pid, pid_to_smiles.get(pid))
            if smi and np.isfinite(score):
                scored[smi] = float(score)
    for path in glob.glob(str(SOTA_ROOT / "results" / "amortized" / "*.json")):
        data = json.loads(Path(path).read_text())
        for k, score in (data.get("final", {}).get("scores") or {}).items():
            if not np.isfinite(score):
                continue
            smi = pid_to_smiles.get(k, k)
            if smi and Chem.MolFromSmiles(smi):
                scored[smi] = float(score)
    return scored


def train_booster(X, y):
    """Train HistGradientBoosting on (fp + descriptors)."""
    y_mean = float(np.mean(y))
    y_std = float(np.std(y) + 1e-6)
    y_norm = (y - y_mean) / y_std
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=None,
        min_samples_leaf=15, l2_regularization=1.0, random_state=42,
    )
    model.fit(X, y_norm)
    return model, y_mean, y_std


def score_pool(model, y_mean, y_std, candidate_pids, pid_to_smiles):
    """Score every candidate pid. Returns (pids, scores) with NaN filtered."""
    pids, feats = [], []
    for pid in candidate_pids:
        smi = pid_to_smiles.get(pid)
        if not smi:
            continue
        v = featurize(smi)
        if v is None:
            continue
        pids.append(pid)
        feats.append(v)
    if not feats:
        return [], []
    preds = model.predict(np.stack(feats)) * y_std + y_mean
    return pids, preds


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


def run(library_dir, pool_path, target_sequence, seed=42, required=20,
        max_sim=0.6, msa=None, cache="nim_cache.json",
        output="amortized_v2.json"):
    random.seed(seed); np.random.seed(seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pid_to_smiles = {}
    with open(pool_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("product_id") and row.get("smiles"):
                pid_to_smiles[row["product_id"]] = row["smiles"]

    # ---------- Phase 1: pool all labels from disk ----------
    train_scored = load_pool_labels()
    print(f"[v2] pooled labels from disk: {len(train_scored)}")
    if len(train_scored) < 100:
        raise RuntimeError(f"need ≥100 pooled labels, have {len(train_scored)}")

    train_smiles = list(train_scored.keys())
    train_y = np.array([train_scored[s] for s in train_smiles], dtype=np.float32)

    # ---------- Phase 2: featurize training labels ----------
    print(f"[v2] featurizing {len(train_smiles)} training molecules...")
    X_tr, y_tr = [], []
    for smi, yv in zip(train_smiles, train_y):
        v = featurize(smi)
        if v is not None:
            X_tr.append(v); y_tr.append(yv)
    X_tr = np.stack(X_tr); y_tr = np.array(y_tr, dtype=np.float32)
    print(f"[v2] training matrix: {X_tr.shape}")

    # ---------- Phase 3: train booster ----------
    print("[v2] training HistGradientBoosting...")
    model, y_mean, y_std = train_booster(X_tr, y_tr)

    # ---------- Phase 4: score pool minus training molecules ----------
    train_set = set(train_smiles)
    candidates = [pid for pid, smi in pid_to_smiles.items() if smi not in train_set]
    print(f"[v2] scoring {len(candidates)} candidates (excluding {len(train_set)} training)...")
    all_pids, all_scores = score_pool(model, y_mean, y_std, candidates, pid_to_smiles)
    print(f"[v2] scored {len(all_pids)} candidates")

    # ---------- Phase 5: top-200 → MaxMin → 20 ----------
    ranked = sorted(zip(all_pids, all_scores), key=lambda x: -x[1])
    top_200 = [p for p, _ in ranked[:200]]
    with Library(library_dir) as library:
        selected = maxmin_pick(top_200, required, library)

        ok, reasons, _ = validate_set(selected, library, max_sim)
        validation = {"valid": ok, "reasons": reasons}

        oracle = NimOracle(api_key=os.environ["NVIDIA_API_KEY"], library=library,
                           target_sequence=target_sequence, cache_path=cache, msa_path=msa)

        final_result = {"valid": False, "reasons": reasons, "scores": {}}
        if ok:
            fs = oracle.score(selected)
            final_result = {
                "valid": True,
                "scores": fs,
                "mean_pIC50": float(np.nanmean(list(fs.values()))),
            }

    best_final = max((v for v in final_result["scores"].values()
                      if not math.isnan(v)), default=float("nan"))

    result = {
        "method": "amortized_booster",
        "version": 2,
        "seed": seed,
        "train_samples": len(y_tr),
        "scored_all": len(all_pids),
        "selected": selected,
        "validation": validation,
        "final": final_result,
        "best_predicted": float(np.max(all_scores)) if len(all_scores) else float("nan"),
        "best_final": best_final,
    }
    output.write_text(json.dumps(result, indent=2))
    print(f"[v2] done. best_final={best_final:.2f} → {output}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--target-sequence", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--required", type=int, default=20)
    p.add_argument("--max-sim", type=float, default=0.6)
    p.add_argument("--msa")
    p.add_argument("--cache", default="nim_cache.json")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    return run(
        library_dir=args.library, pool_path=args.pool,
        target_sequence=args.target_sequence, seed=args.seed,
        required=args.required, max_sim=args.max_sim, msa=args.msa,
        cache=args.cache, output=args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
