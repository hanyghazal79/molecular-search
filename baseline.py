"""Simple pool-sampling integration baseline, not a competitive search algorithm."""
import argparse
import csv
import json
import os
import random
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from client import EvaluationClient, EvaluationError


@contextmanager
def output_lock(path):
    """The OS releases this exclusive checkpoint lock even if the process dies."""
    lock = Path(str(path) + ".lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise EvaluationError("Another baseline process is using this checkpoint") from None
        else:
            import fcntl
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise EvaluationError("Another baseline process is using this checkpoint") from None
        yield
    finally:
        lock.close()


def checkpoint(path, state):
    """Replace a checkpoint only after its complete JSON has reached disk."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(state, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def error_record(exc):
    return {"message": str(exc), "job_id": exc.job_id, "run_id": exc.run_id,
            "retryable": exc.retryable, "status": exc.status}


def evaluate_pending(client, state, operation, products, save):
    pending = state.get("pending")
    if pending is None:
        pending = {"operation": operation, "products": products,
                   "request_id": uuid.uuid4().hex, "job_id": None}
        state["pending"] = pending
        save()
    if pending["operation"] != operation or pending["products"] != products:
        raise EvaluationError("Checkpoint pending request does not match the saved experiment",
                              run_id=state["run_id"], job_id=pending.get("job_id"))
    try:
        if not pending.get("job_id"):
            # A resumed ambiguous POST uses the same persisted request ID. The
            # service returns its original job without reserving allowance again.
            job = client.submit(state["run_id"], products, operation, request_id=pending["request_id"])
            pending["job_id"] = job["id"]
            save()
        return client.wait(pending["job_id"], run_id=state["run_id"])
    except EvaluationError as exc:
        exc.with_context(run_id=state["run_id"], job_id=pending.get("job_id"))
        pending["error"] = error_record(exc)
        if exc.job is not None:
            pending["job"] = exc.job
        save()
        raise


def continue_run(client, state, save):
    case = state["case"]
    count = case["required_count"]
    while len(state["probes"]) < len(state["candidates"]):
        product = state["candidates"][len(state["probes"])]
        # Query individually so one invalid candidate does not reject a batch.
        job = evaluate_pending(client, state, "oracle", [product], save)
        state["probes"].append(job)
        state["pending"] = None
        save()
    if state["selected"] is None:
        items = []
        for job in state["probes"]:
            response = job["result"]
            if response["backend"] == "demo" or response.get("valid"):
                items.extend(response["items"])
        key = "demo_value" if case["backend"] == "demo" else "score"
        reverse = case.get("public", {}).get("score_direction", "max") != "min"
        selected = [item["product"] for item in sorted(items, key=lambda v: v[key], reverse=reverse)[:count]]
        if len(selected) < count:
            raise EvaluationError("Not enough valid candidates; the completed probes are saved. "
                                  "Discuss a larger pool/probe budget with the owner.", run_id=state["run_id"])
        state["selected"] = selected
        save()
    if state["validation"] is None:
        state["validation"] = evaluate_pending(client, state, "validate", state["selected"], save)
        state["pending"] = None
        save()
    validation = state["validation"]["result"]
    if validation["backend"] == "reference" and not validation.get("valid"):
        status = "final_set_invalid"
    else:
        if state["final"] is None:
            state["final"] = evaluate_pending(client, state, "final", state["selected"], save)
            state["pending"] = None
            save()
        status = "completed"
    state["ledger"] = client.run(state["run_id"])
    state["status"] = status
    state.pop("error", None)
    save()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="demo")
    p.add_argument("--pool", default="examples/demo_pool.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probes", type=int, default=8)
    p.add_argument("--revision", default="integration-baseline-v1")
    p.add_argument("--output", default="baseline-result.json")
    p.add_argument("--resume", help="Continue the exact run saved in this checkpoint; other experiment options are ignored")
    a = p.parse_args(argv)
    output = Path(a.resume or a.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        return run_experiment(a, output)


def run_experiment(a, output):
    client = EvaluationClient()
    if a.resume:
        state = json.loads(output.read_text())
        if state.get("checkpoint_version") != 1 or not state.get("run_id"):
            raise EvaluationError("This file is not a resumable baseline checkpoint")
        if state.get("endpoint") != client.base:
            raise EvaluationError("EVAL_URL must match the endpoint in the checkpoint", run_id=state["run_id"])
        run = client.run(state["run_id"])
        if run["case_id"] != state["case"]["id"] or run["case_digest"] != state["case_digest"]:
            raise EvaluationError("Checkpoint does not match the service's frozen run", run_id=state["run_id"])
    else:
        if output.exists():
            raise EvaluationError("Output already exists; use --resume to continue it or choose a new --output")
        cases = {c["id"]: c for c in client.cases()}
        if a.case not in cases:
            raise EvaluationError("Case is not available to this token")
        case = cases[a.case]
        if case["backend"] == "reference" and not case.get("affinity_configured"):
            raise EvaluationError("This case supports validation only; ask the owner for an affinity-configured case "
                                  "(for example nnmt-rxn1-gpu) before running the scoring baseline")
        count = case["required_count"]
        with open(a.pool, newline="") as stream:
            pool = list(dict.fromkeys(row["product_id"] for row in csv.DictReader(stream)))
        if len(pool) < max(count, a.probes) or a.probes < count:
            raise EvaluationError("Pool and --probes must each contain at least the final required count")
        requested = (a.probes + count) * case["target_count"]
        if requested > case["max_pairs_per_run"]:
            raise EvaluationError("Probes plus final reevaluation would exceed this run's pair allowance")
        candidates = random.Random(a.seed).sample(pool, a.probes)
        run = client.create_run(a.case, f"pool-sampling-seed-{a.seed}", a.revision)
        state = {"checkpoint_version": 1, "endpoint": client.base, "run_id": run["id"],
                 "case_digest": run["case_digest"], "case": case, "seed": a.seed, "revision": a.revision,
                 "candidates": candidates, "probes": [], "selected": None, "validation": None,
                 "final": None, "pending": None, "status": "running", "client_wall_seconds": 0.0}
        checkpoint(output, state)
    started = time.monotonic()
    previous_seconds = state.get("client_wall_seconds", 0.0)

    def save():
        state["client_wall_seconds"] = previous_seconds + time.monotonic() - started
        checkpoint(output, state)

    try:
        if state["case"].get("case_digest") != state["case_digest"]:
            raise EvaluationError("Case changed while creating this run; no further jobs will be submitted. "
                                  "Choose a new output to start with the updated case.", run_id=state["run_id"])
        if state["status"] not in ("completed", "final_set_invalid") or "ledger" not in state:
            state["status"] = "running"
            continue_run(client, state, save)
    except EvaluationError as exc:
        exc.with_context(run_id=state["run_id"])
        state["status"] = "interrupted" if exc.retryable else "failed"
        state["error"] = error_record(exc)
        save()
        raise
    except (Exception, KeyboardInterrupt):
        state["status"] = "interrupted"
        save()
        raise
    print(f"Saved {output}; run={state['run_id']}; status={state['status']}")
    if state["case"]["backend"] == "demo":
        print("DEMO ONLY: synthetic values cannot assess molecular-search quality.")
    return state


if __name__ == "__main__":
    try:
        main()
    except EvaluationError as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        raise SystemExit(1)
