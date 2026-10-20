# Molecule Search — developer starter kit

Build and benchmark a search pipeline that selects a strong, valid molecular
set from a reaction-based chemical space. This folder is the developer handoff.
The owner runs the separate service and supplies your endpoint and token.

## What you receive from the owner

- An HTTPS evaluation URL (or a local/private-network URL for development).
- Your own bearer token and the permitted challenge IDs.
- A frozen library bundle containing `library.json` and `molecules.sqlite`, or
  a candidate pool for the initial integration task. Ask for its snapshot digest.
- A target-molecule-pair allowance and agreed compute spending, where applicable.

The CPU demo contains invented product IDs and synthetic values. **It is not a
chemistry benchmark and must not be used to assess scientific performance.**

## Browser workspace

Open the owner's evaluation URL in your browser and sign in with your token.
Your workspace shows assigned cases, remaining allowances, experiments and job
results. Create an experiment, paste product IDs or import a candidate CSV, and
submit validation or scoring jobs. Check the displayed pair cost before submitting.
Results update as the worker completes each job; records can be downloaded.

The Resources page provides this kit and any published frozen libraries. Download
both `library.json` and `molecules.sqlite` into the same directory. Use the scripts
below when your search needs automation. The browser and scripts share your
allowance and experiment history. Sign out on shared machines; your token also
continues to work with the Python client.

### Historical molecule data

The Resources page also provides a versioned **Historical molecules** dataset.
Download `history.json` and `molecules.csv.gz` into the same directory. This
collection contains previous molecular structures across several targets, with
the columns `target_id`, `historical_id`, `smiles`, and `inchi_key`. The manifest
records the row count, targets, snapshot ID and compressed-file SHA-256. Its
`quality` section records the source row count and per-target exclusions for
records missing a structure or chemical identifier. The downloadable CSV
contains only complete records; existing duplicate records are retained.

Use it to explore chemical space, test similarity searches and develop novelty
filters. It contains no affinity scores or experimental activity labels.
Historical identifiers can use older formats and are not necessarily accepted
product IDs for your current case. Reusing an archived molecule or a close
analogue can fail the evaluator's novelty/similarity checks. Each experiment
continues to use its own frozen rules and history; this general dataset can
cover other targets or a different snapshot.

You can inspect the compressed CSV using only Python's standard library, without
loading the entire dataset into memory:

```python
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path

folder = Path("/path/to/history")
manifest = json.loads((folder / "history.json").read_text())
metadata = next(item for item in manifest["files"] if item["name"] == "molecules.csv.gz")
with (folder / "molecules.csv.gz").open("rb") as stream:
    if hashlib.file_digest(stream, "sha256").hexdigest() != metadata["sha256"]:
        raise ValueError("Historical dataset checksum mismatch")
print(manifest["total_rows"], "records across", manifest["target_count"], "targets")
with gzip.open(folder / "molecules.csv.gz", "rt", encoding="utf-8", newline="") as stream:
    for row in itertools.islice(csv.DictReader(stream), 5):
        print(row)
```

Keep the manifest with your experiment notes so that your data preparation can
be reproduced. New snapshots appear when the owner service refreshes the data;
previously downloaded snapshots stay useful for reproducing earlier work.

## Quick integration check — no third-party Python packages required

Use Python 3.12. Run from this folder:

```bash
export EVAL_URL=http://127.0.0.1:8000
read -rs EVAL_TOKEN
export EVAL_TOKEN
python baseline.py --case demo --output demo-result.json
```

Paste the supplied token after `read`; it is not echoed. For a remote service,
use the owner's HTTPS URL. The example makes eight synthetic oracle queries,
validates three selected identifiers, and requests a synthetic final result.
It reserves eleven target-molecule pairs. The output saves progress after every
operation, including individual probes, selected products, and the final ledger.

## Real challenge workflow

```python
from client import EvaluationClient
client = EvaluationClient()
print(client.cases())
run = client.create_run("nnmt-rxn1", "baseline-seed-42", code_revision="your-git-commit")
# Product IDs must come from the owner's frozen reaction database, not demo_pool.csv.
job = client.evaluate(run["id"], your_products, operation="validate")
print(job["result"])
```

There are three operations, all submitted as jobs:

| Operation | Purpose | Charge |
|---|---|---|
| `validate` | Check a complete final set using frozen chemistry, novelty and diversity rules | No pair charge; at most 50 requests per run, plus an account-wide allowance |
| `oracle` | Score a small search batch; within-set entropy and final set count are relaxed | Requested products × targets |
| `final` | Validate and score a complete final set | Requested products × targets again |

Oracle batches still face reaction, chemical property, Boltz safety, historical
novelty, similarity and chemical-duplicate checks. **One invalid candidate rejects
the batch.** A successful probe does not establish that a final set is diverse.
The complete set must pass `validate` and `final` separately.

Allowances count accepted scoring requests, including invalid, failed and timed
out jobs. There is no free cross-run cache. The client retries temporary polling
failures until its deadline, but never automatically retries a submission. Ask
the owner for an allowance increase after a service-side failure if appropriate.
The default account allowances are 200 validation requests and 200 runs; the
owner can configure different limits. Starting a new run or rotating a token
does not reset those allowances or the total scoring-pair allowance.

### Resume interrupted work

Keep the output file: it is also the experiment checkpoint. After a network
failure, timeout, or interruption, use the same endpoint and developer identity:

```bash
python baseline.py --resume demo-result.json
```

Resume uses the saved case, seed, candidate order, completed probes, and pending
job. Other experiment options are ignored. It polls an accepted job again. If a
submission response was lost, it repeats the exact saved request ID and payload;
the service returns the original job without reserving allowance again. Completed
experiments are not repeated. A failed worker job remains failed and is saved
with its error; resume does not replace it with another paid job. Start a separate
experiment with a new output filename only when you intend to spend again.

The baseline refuses to overwrite an existing output or run concurrently against
the same checkpoint. Checkpoints contain no API token. Run creation itself is
not retried: if its response is lost before a checkpoint exists, inspect your
runs with the owner; that empty run has not submitted any scoring jobs.

For a custom search, persist a unique `request_id` before calling
`client.submit(run_id, products, operation="oracle", request_id=request_id)`.
Reusing that ID with identical products and operation recovers the original job
within that run. A different payload under the same ID is rejected. Save the
returned job ID before `client.wait(job_id, run_id=run_id)`. `EvaluationError`
provides `job_id`, `run_id`, `retryable`, and HTTP `status` where available;
`retryable` is not permission to create a new scoring request.

The reference set size is obtained from the case; do not hardcode an old value.
The service accepts product identifiers, for example
`rxn:2:123:456` or `rxn:2:123:456:k1`, and resolves the actual SMILES itself.
Your submitted SMILES, score, model, target, threshold or shell command cannot
override its evaluation. There is no code-upload or execution endpoint.

## Small random pool generator

If the owner supplies the frozen library bundle:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-optional-chemistry.txt
.venv/bin/python generate_pool.py --library /path/to/library --reaction rxn:1 --count 200 --output pool.csv
.venv/bin/python baseline.py --case nnmt-rxn1-gpu --pool pool.csv --probes 100 --output real-baseline-result.json
```

The generator checks the library snapshot and applies its reaction and property
rules. It is only a simple random starting point. Use the allowed reaction from
the case. The evaluation service performs the authoritative complete checks.
The sampling baseline chooses top individual scores, so its final set may fail
entropy or chemical uniqueness. Such a failure is reported rather than hidden.
Improving generation, search and set selection is the actual engineering work.

The scoring baseline requires a case with `affinity_configured: true`; a
validation-only case cannot produce oracle or final affinity scores. The example
case name `nnmt-rxn1-gpu` must be granted to your token by the owner. You can still
use `validate` directly on a validation-only case such as `nnmt-rxn1`.

## First paid assignment

1. Confirm source/data/evaluator versions and reproduce the integration baseline.
2. Produce at least one full chemically valid final set on a reference case.
3. Record bottlenecks, evaluation throughput, rejected candidates and compute use.
4. Test one focused improvement using the same cases and requested-pair budget.
5. Deliver the code, exact commands, result JSON files, and a short interpretation.

Use several fixed search seeds and the same input snapshot. Keep a separate
held-out set of cases for owner review when available. A successful integration
run alone does not establish improved scientific performance.

## Daily update template

Completed work; experiment/run IDs; valid final sets and scores; reserved pairs;
runtime and compute spend; blockers; next step. State clearly whether each
number came from demo, frozen reference validation, or real model inference.

## Measurement limits

The service measures its own requested scoring work, queue time and worker
runtime. Generation on your computer, extra GPU/API calls outside the service,
and your code revision are self-reported. Therefore this kit alone does not
enforce an identical total CPU/GPU budget for every search program. The owner
should rerun shortlisted code in one controlled environment for that comparison.
Frozen local scores do not establish novelty against newer data or laboratory
efficacy. Confirm any claimed improvement on held-out cases and frozen inputs.
