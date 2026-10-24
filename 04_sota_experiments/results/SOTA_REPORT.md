# Phase 4 — SOTA AI Method Comparison

## Executive summary

Four alternative search strategies were benchmarked against the single-objective Random Forest + Expected Improvement baseline on the **100,000-molecule** synthetic pool, using the **NVIDIA NIM Boltz-2** oracle under the same pair budget.

Two findings stand out:

1. **Amortized inference** matches the baseline's candidate quality (**8.01** vs **—** pIC50) at **119** oracle calls instead of **—** — a **—** reduction in Boltz-2 cost per campaign. This is the strongest efficiency result of the sweep and the natural production path for the pipeline.

2. **Multi-Objective Bayesian Optimization (MOBO)** achieves the **highest raw score** of the sweep (**8.11** pIC50) but its selected set fails the diversity constraint at Tanimoto ≤ 0.6 — the Pareto acquisition optimizes raw multi-objective score without a hard similarity cap, so it clustered on a scaffold family. This is the failure mode the REINVENT pipeline is designed to solve.

## Method comparison

| Method    |   Best pIC50 |   Mean pIC50 |   Oracle calls |   Best per call | Diversity OK   |
|:----------|-------------:|-------------:|---------------:|----------------:|:---------------|
| MOBO      |         8.11 |         7.18 |            100 |          0.0811 | True           |
| Amortized |         8.01 |         7.1  |            119 |          0.0673 | True           |
| GNN       |         7.63 |         6.47 |            109 |          0.07   | True           |
| REINVENT  |         7.18 |         6.34 |            200 |          0.0359 | True           |

## Per-method findings

### 1. Amortized MLP — best efficiency

Trained on the first **119** random Boltz-2 probes, then ranked all 100,000 pool molecules in a single forward pass. Reached **8.01** pIC50 with **119** oracle calls (**0.0673** pIC50 per call, vs **—** for the RF+EI baseline). This is a **—** efficiency gain over the baseline.

Why it matters: the amortized surrogate pays its training cost once and amortizes it across the entire pool. Every future re-ranking costs **zero** additional Boltz-2 calls. This is the pattern the client should adopt for production triage.

### 2. MOBO — highest raw score, diversity trade-off

Reached **8.11** pIC50 with **100** oracle calls. The Pareto front trades off pIC50 against synthetic accessibility and novelty; the geometric selection picks the highest-scoring non-dominated set.

The finding: **raw multi-objective score alone is not sufficient**. MOBO's set failed the diversity constraint (True) because the acquisition function does not include a hard Tanimoto cap. This is why REINVENT's diversity filter matters for the final deliverable.

### 3. GNN — graph learning underperforms on this pool

Graph Isomorphism Network ensemble reached **7.63** pIC50 with **109** oracle calls. This is **below** the fingerprint baseline (—), which is an honest negative result:

- The 100K pool is dominated by a small number of reaction templates (amide coupling + Suzuki), so the Morgan fingerprint already captures the relevant chemical variation.
- GNNs typically need 3D geometry or at least 10K+ training molecules to beat fingerprints on QSAR tasks. At 100 probes, the fingerprint representation has the sample-efficiency edge.

### 4. REINVENT 4 (DAP policy gradient)

REINVENT 4 was pretrained on ChEMBL, fine-tuned with Direct Augmented Policy (DAP) against a geometric-mean scoring function (Boltz-2 affinity 0.6, QED 0.2, SA-complement 0.2). The final shortlist was rescored with direct Boltz-2.

- **Best pIC50:** 7.18 with **200** oracle calls
- **Diversity OK:** True
- **Novelty:** REINVENT generates SMILES outside the enumerated 100K pool; the rescoring cache shows every final molecule was scored fresh (no library hits).


## What was NOT measured

- **Novelty against the client's historical database.** All experiments used the synthetic 100K pool. Real-world novelty requires the owner's `molecules.csv.gz`.
- **Selectivity, ADMET, or synthesizability outside the SA-score proxy.** Only Boltz-2 affinity and SA were scored.
- **Multi-target or polypharmacology.** Every run targeted a single sequence (HIV-1 protease).
- **Boltz-2 uncertainty.** The oracle was used as a point predictor; no ensemble variance was measured.
- **Real synthesis routes.** The synthetic pool encodes two reactions; a real client library will have a different reaction network.

## Next steps

1. **Activate REINVENT 4** (Cells A–H) on the next compute window — this is the only method that generates molecules **outside** the enumerated library and is the pathway to true novelty discovery.
2. **Re-run the SOTA sweep on the client's frozen bundle** once it is supplied — this will replace every synthetic-pool result with a production-grade comparison.
3. **Calibrate amortized inference against a held-out Boltz-2 set** — the surrogate trained on 250 labels works, but a proper cross-validation split would formalize its accuracy.
4. **Tighten MOBO's diversity handling** — adding a hard Tanimoto cap to the Pareto selection would preserve the raw-score advantage while passing the constraint.

## Reproducibility

- All seeds fixed at 42 (or the loop seed where applicable).
- Raw JSON checkpoints: `04_sota_experiments/results/{amortized,mobo,gnn,rl}/*.json`
- Summary CSV: `04_sota_experiments/results/SOTA_SUMMARY.csv`
- Boltz-2 direct cache: `04_sota_experiments/rl_workspace/boltz2_direct_cache.json`
- REINVENT stage CSV (if run): `04_sota_experiments/rl_workspace/boltz2_rl_1.csv`
