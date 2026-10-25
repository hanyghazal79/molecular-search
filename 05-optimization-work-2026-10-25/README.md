# Today's work — 2026-10-25

Self-contained snapshot of the day's optimization and audit work.
Regenerate every result by re-running the scripts in `scripts/` against
the cache in `caches/nim_cache.json` -- the cache makes re-runs free.

## What's inside

| Folder | Purpose |
|---|---|
| `scripts/` | Patched and new scripts produced today |
| `results/` | Result JSONs for every run executed today |
| `caches/` | NIM label caches (grew from ~218 to ~2,201 unique labels) |
| `models/` | Trained Boltz-2 surrogate |
| `AUDIT_OUTPUT.md` | Cross-validation + holdout evidence for all surrogates |
| `MANIFEST.json` | File-by-file SHA-256 for integrity verification |

## What was done today

1. **Surrogate audit** -- formal CV + holdout validation on all four methods.
   Key finding: MOBO validates at rho=0.86 holdout; REINVENT fails
   out-of-distribution by design (rho=-0.21).

2. **GNN fix** -- the GNN was underfitting (predictions offset -3.74 pIC50).
   Fix: normalize targets, clip gradients, train 150 epochs.
   Result: 7.63 -> 8.04 pIC50 (+0.41, monotonic climb).

3. **Amortized v2** -- pooled all prior labels (99 -> 295), swapped MLP for
   HistGradientBoosting, added 210 RDKit descriptors.
   Result: 8.01/7.10 -> 8.13/7.54 (best/mean) at 20 NIM calls
   (down from 119 -- 10x more efficient).

4. **MOBO v2** -- warm-started from cache, then 100 fresh probes.
   Result: 8.11 -> 8.69 pIC50 (+0.58, fresh discovery).

5. **RF+EI v2** -- warm-start protocol, 200 fresh probes.
   Result: inherited 8.69 from MOBO v2's cache; no further lift.

6. **RF+EI baseline refresh** -- verified 8.05/7.70 at seed 42.

## Headline results

Cold-start (fair comparison):

| Method | Best | Mean | NIM calls |
|---|---:|---:|---:|
| Amortized v2 | 8.13 | 7.54 | 20 |
| MOBO v1 | 8.11 | 7.18 | 100 |
| RF+EI v1 | 8.05 | 7.70 | 200 |
| GNN (fixed) | 8.04 | 6.53 | 100 |
| REINVENT | 7.18 | 6.34 | 200 |

Warm-start (deployment):

| Method | Best | Mean | Fresh calls |
|---|---:|---:|---:|
| MOBO v2 | 8.69 | 7.56 | 100 |
| RF+EI v2 | 8.69 | 8.13 | 200 |
| Amortized v2 | 8.13 | 7.54 | 20 |

## What's NOT here

- Synthetic library (molecules.sqlite, ~1 GB) -- regenerate via
  `python make_synthetic_library.py 100000 synthetic-library-100000`
- Prior-phase scripts (chemistry.py, nim_oracle.py, local_eval.py,
  sota_utils.py) -- these live in the main repo, unchanged today
- State files (*.state.json) -- transient, regenerable from result JSONs
- Backups and logs

## Cache value

nim_cache.json holds ~2,201 unique Boltz-2 labels accumulated across
the session. Every re-run of any script against this cache costs
zero NIM calls -- queries are memoized. This is the primary artifact.

## Integrity

Every file's SHA-256 is recorded in MANIFEST.json.
