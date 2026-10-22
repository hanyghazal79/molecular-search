"""Compare checkpoints from baseline.py, search.py, or search_nim.py.

Handles both schemas:
- baseline.py / search.py: state["probes"] = [{result:{items:[{product,score}], backend}}]
- search_nim.py: state["scored"] = [[product, score], ...]
"""
import os
import csv
import json
import sys
from pathlib import Path


def _normalize(state):
    """Return [(product_id, score)] regardless of source schema."""
    if state.get("scored"):
        return [(p, s) for p, s in state["scored"]]
    out = []
    for job in state.get("probes", []):
        res = job.get("result") or {}
        backend = res.get("backend", "reference")
        for item in res.get("items", []):
            pid = item.get("product")
            score = item.get("demo_value") if backend == "demo" else item.get("score")
            if pid and isinstance(score, (int, float)):
                out.append((pid, float(score)))
    return out


def _validation(state):
    v = state.get("validation") or {}
    return v.get("result") or v


def _final_scores(state):
    """Return final-set scores (list of floats) or [] if not run."""
    final = state.get("final") or {}
    # search_nim.py schema
    if "scores" in final:
        return [float(v) for v in final["scores"].values()
                if isinstance(v, (int, float)) and v == v]
    # baseline.py schema
    res = final.get("result") or {}
    backend = res.get("backend", "reference")
    out = []
    for item in res.get("items", []):
        v = item.get("demo_value") if backend == "demo" else item.get("score")
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def compute(state, fps_by_pid=None):
    case = state.get("case", {}) or {}
    target_count = case.get("target_count", 1)
    required = case.get("required_count", 3)
    probes = _normalize(state)
    selected = state.get("selected") or []
    validation = _validation(state)
    final_scores = _final_scores(state)
    scores = [s for _, s in probes]

    # search_nim.py does not pay per-pair for final (cache hit), baseline does
    if state.get("scored") is not None:
        pairs_charged = len(probes)  # one NIM call per probe
    else:
        pairs_charged = (len(state.get("probes", [])) + (required if final_scores else 0)) * target_count

    metrics = {
        "checkpoint_strategy": state.get("strategy", "baseline"),
        "case": case.get("id", "synthetic-demo"),
        "seed": state.get("seed"),
        "n_probes": len(probes),
        "n_unique_scored": len(set(p for p, _ in probes)),
        "n_selected": len(selected),
        "required_count": required,
        "pairs_charged": pairs_charged,
        "validation_valid": validation.get("valid"),
        "final_valid": (state.get("final") or {}).get("valid"),
        "status": state.get("status", "completed"),
    }
    if scores:
        top = sorted(scores, reverse=True)
        metrics["best_probe_score"] = top[0]
        metrics["mean_top_k_probe"] = sum(top[:required]) / max(required, 1)
        metrics["mean_all_probe"] = sum(scores) / len(scores)
        metrics["best_per_pair"] = top[0] / pairs_charged if pairs_charged else 0.0
    if final_scores:
        metrics["best_final_score"] = max(final_scores)
        metrics["mean_final_score"] = sum(final_scores) / len(final_scores)

    if fps_by_pid and len(selected) >= 2:
        from rdkit import DataStructs
        fps = [fps_by_pid[p] for p in selected if p in fps_by_pid]
        if len(fps) >= 2:
            sims = [DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    for i in range(len(fps)) for j in range(i + 1, len(fps))]
            metrics["div_mean_sim"] = sum(sims) / len(sims)
            metrics["div_max_sim"] = max(sims)
    return metrics


def load_fps(pool_path):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        print("WARN: rdkit not installed; diversity metrics skipped", file=sys.stderr)
        return None
    fps = {}
    with open(pool_path, newline="") as stream:
        for row in csv.DictReader(stream):
            pid = (row.get("product_id") or "").strip()
            smi = (row.get("smiles") or "").strip()
            if not pid or not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fps[pid] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    return fps


def main(checkpoints, pool=None, csv_out=None):
    """Core logic with parametrized signatures."""
    fps = load_fps(pool) if pool else None
    rows = []
    for path in checkpoints:
        try:
            state = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARN: cannot read {path}: {exc}", file=sys.stderr); continue
        row = compute(state, fps)
        row["checkpoint"] = Path(path).name
        rows.append(row)

    if not rows:
        print("No checkpoints read", file=sys.stderr); return 1

    columns = ["checkpoint", "case", "checkpoint_strategy", "seed", "n_probes",
               "n_unique_scored", "n_selected", "pairs_charged",
               "best_probe_score", "mean_top_k_probe", "best_per_pair",
               "best_final_score", "validation_valid", "final_valid",
               "div_mean_sim", "div_max_sim"]
    print("\t".join(columns))
    for row in rows:
        print("\t".join(str(row.get(c, "")) for c in columns))

    if csv_out:
        with open(csv_out, "w", newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"\nSaved {csv_out}", file=sys.stderr)
    return 0

RESULTS_DIR = "/kaggle/working/results"
POOL_PATH = "pool.csv"
CSV_OUT = f"{RESULTS_DIR}/phase2-seeds.csv"

result_files = [
    os.path.join(RESULTS_DIR, f) 
    for f in os.listdir(RESULTS_DIR) 
    if f.endswith(".json") and f.startswith("search-")
]

# metrics_main(result_files, pool=POOL_PATH, csv_out=CSV_OUT)
