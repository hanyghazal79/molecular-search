"""Surrogate-guided search over the owner's evaluation service.

Shares checkpoint and idempotency contract with baseline.py.
Requires a pool CSV with a smiles column (see generate_pool.py).

Do not run until the owner supplies the frozen library bundle and grants
your token access to an affinity_configured case.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

from baseline import output_lock, checkpoint, error_record, evaluate_pending
from client import EvaluationClient, EvaluationError


def _load_pool(path):
    with open(path, newline="") as stream:
        return [{"product_id": r["product_id"].strip(),
                 "smiles": (r.get("smiles") or "").strip()}
                for r in csv.DictReader(stream) if r.get("product_id")]


def _fingerprints(pool):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return {}, False
    fps = {}
    for row in pool:
        if not row.get("smiles"):
            return {}, False
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            return {}, False
        fps[row["product_id"]] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    return fps, True


def _maxmin(pids, k, fps):
    from rdkit import DataStructs
    if len(pids) <= k or not fps:
        return pids[:k]
    chosen, remaining = [pids[0]], pids[1:]
    while len(chosen) < k and remaining:
        best, best_sim = None, 2.0
        for pid in remaining:
            if pid not in fps:
                return chosen + remaining[:k - len(chosen)]
            sim = max(DataStructs.TanimotoSimilarity(fps[pid], fps[c]) for c in chosen)
            if sim < best_sim:
                best, best_sim = pid, sim
        chosen.append(best); remaining.remove(best)
    return chosen


def _scored(state):
    pairs = []
    for job in state.get("probes", []):
        res = job.get("result") or {}
        backend = res.get("backend", "reference")
        for item in res.get("items", []):
            if backend == "demo":
                pairs.append((item["product"], float(item["demo_value"])))
            elif res.get("valid"):
                pairs.append((item["product"], float(item["score"])))
    return pairs


def _probed(state):
    return {item["product"] for job in state.get("probes", [])
            for item in (job.get("result") or {}).get("items", [])}


def _select_batch(state, remaining, fps, batch_size):
    strategy, scored = state["strategy"], _scored(state)
    if strategy == "random" or not fps or len(scored) < 2 * batch_size:
        return random.Random(state["seed"] + len(scored)).sample(
            remaining, min(batch_size, len(remaining)))
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from scipy.stats import norm
    X = np.zeros((len(scored), 2048), dtype=np.float32)
    from rdkit import DataStructs
    for i, (pid, _) in enumerate(scored):
        if pid not in fps:
            return random.Random(state["seed"] + len(scored)).sample(
                remaining, min(batch_size, len(remaining)))
        DataStructs.ConvertToNumpyArray(fps[pid], X[i])
    y = np.array([s for _, s in scored])
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                               random_state=state["seed"], n_jobs=-1).fit(X, y)
    X_rem = np.zeros((len(remaining), 2048), dtype=np.float32)
    for i, pid in enumerate(remaining):
        if pid in fps:
            DataStructs.ConvertToNumpyArray(fps[pid], X_rem[i])
    preds = np.array([t.predict(X_rem) for t in rf.estimators_])
    mu, sigma = preds.mean(axis=0), preds.std(axis=0) + 1e-9
    if strategy == "greedy":
        acq = mu
    elif strategy == "ucb":
        acq = mu + 2.0 * sigma
    else:
        best = y.max()
        z = (mu - best) / sigma
        acq = (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)
    ranked = [remaining[i] for i in np.argsort(-acq)]
    return _maxmin(ranked[:max(50, batch_size * 10)], batch_size, fps)


def run_experiment(args, output):
    client = EvaluationClient()
    if args.resume:
        state = json.loads(output.read_text())
        if state.get("checkpoint_version") != 1 or not state.get("run_id"):
            raise EvaluationError("Not a resumable search checkpoint")
        if state.get("endpoint") != client.base:
            raise EvaluationError("EVAL_URL must match checkpoint endpoint",
                                  run_id=state["run_id"])
        run = client.run(state["run_id"])
        if run["case_id"] != state["case"]["id"] or run["case_digest"] != state["case_digest"]:
            raise EvaluationError("Checkpoint does not match service's frozen run",
                                  run_id=state["run_id"])
        pool = _load_pool(args.pool)
    else:
        if output.exists():
            raise EvaluationError("Output exists; use --resume or new --output")
        cases = {c["id"]: c for c in client.cases()}
        if args.case not in cases:
            raise EvaluationError("Case not available")
        case = cases[args.case]
        if case["backend"] == "reference" and not case.get("affinity_configured"):
            raise EvaluationError("Validation-only case; request affinity case")
        required, target, max_pairs = case["required_count"], case["target_count"], case["max_pairs_per_run"]
        if args.probes < required:
            raise EvaluationError("--probes must be >= required_count")
        if (args.probes + required) * target > max_pairs:
            raise EvaluationError(f"Budget exceeds max_pairs_per_run={max_pairs}")
        pool = _load_pool(args.pool)
        if len(pool) < args.probes:
            raise EvaluationError(f"Pool {len(pool)} < probes {args.probes}")
        run = client.create_run(args.case, f"search-{args.strategy}-seed-{args.seed}",
                                args.revision)
        state = {
            "checkpoint_version": 1, "endpoint": client.base,
            "run_id": run["id"], "case_digest": run["case_digest"],
            "case": case, "seed": args.seed, "revision": args.revision,
            "strategy": args.strategy, "batch_size": args.batch_size,
            "max_sim": args.max_sim, "probes_planned": args.probes,
            "probes": [], "selected": None, "validation": None, "final": None,
            "pending": None, "status": "running", "client_wall_seconds": 0.0,
        }
        checkpoint(output, state)

    started, previous = time.monotonic(), state.get("client_wall_seconds", 0.0)

    def save():
        state["client_wall_seconds"] = previous + time.monotonic() - started
        checkpoint(output, state)

    try:
        fps, have_smiles = _fingerprints(pool)
        case = state["case"]
        required = case["required_count"]

        while len(state["probes"]) < state["probes_planned"]:
            remaining = [r["product_id"] for r in pool if r["product_id"] not in _probed(state)]
            if not remaining:
                break
            batch = _select_batch(state, remaining, fps if have_smiles else {},
                                  state["batch_size"])
            for pid in batch:
                if len(state["probes"]) >= state["probes_planned"]:
                    break
                job = evaluate_pending(client, state, "oracle", [pid], save)
                state["probes"].append(job)
                state["pending"] = None
                save()

        if state["selected"] is None:
            scored = sorted(_scored(state), key=lambda x: -x[1])
            state["selected"] = [p for p, _ in scored[:required]]
            save()

        if state["validation"] is None:
            state["validation"] = evaluate_pending(client, state, "validate",
                                                    state["selected"], save)
            state["pending"] = None
            save()

        if (state["validation"].get("result") or {}).get("valid") is False:
            state["status"] = "final_set_invalid"
        else:
            if state["final"] is None:
                state["final"] = evaluate_pending(client, state, "final",
                                                   state["selected"], save)
                state["pending"] = None
                save()
            state["status"] = "completed"

        state["ledger"] = client.run(state["run_id"])
        state.pop("error", None)
        save()
    except EvaluationError as exc:
        exc.with_context(run_id=state["run_id"])
        state["status"] = "interrupted" if exc.retryable else "failed"
        state["error"] = error_record(exc)
        save()
        raise

    print(f"Saved {output}; status={state['status']}")
    return state


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probes", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--strategy", choices=["random", "greedy", "ucb", "ei"], default="ei")
    p.add_argument("--max-sim", type=float, default=0.6)
    p.add_argument("--revision", default="search-al-v1")
    p.add_argument("--output", default="search-result.json")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    output = Path(a.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        return run_experiment(a, output)


if __name__ == "__main__":
    try:
        main()
    except EvaluationError as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        raise SystemExit(1)